import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


# 1. Scalar Routers (unchanged)
class EqualWeights(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    @register_to_config
    def __init__(self, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        # ensure at least one buffer on the module
        self.register_buffer("_dummy_eq", torch.zeros(1))

    def forward(self, inputs=None):
        if isinstance(inputs, torch.Tensor):
            device = inputs.device
        else:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = torch.zeros([1, self.num_experts], device=device)
        return logits


class SimpleWeights(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    @register_to_config
    def __init__(self, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.wg = nn.Linear(1, self.num_experts, bias=False)
        # ensure at least one buffer on the module
        self.register_buffer("_dummy_sw", torch.zeros(1))

    def forward(self, inputs=None):
        device = next(self.wg.parameters()).device
        dtype = next(self.wg.parameters()).dtype
        x = torch.ones([1, 1], device=device, dtype=dtype)
        return self.wg(x)


# 2. Individual MLP_k (Feature Extractor)
# Now requires explicit in_channels at init to avoid mismatch.
class PatchLevelFeatExtractor(nn.Module):
    """
    3-layer MLP_k: takes one ControlNet feature W_k (C_k channels)
    and produces a single logit map [B, 1, H, W].

    Structure:
      Conv2d(total_in, hidden_dim, 1)
      GELU
      Conv2d(hidden_dim, hidden_dim, 1)
      GELU
      Conv2d(hidden_dim, 1, 1)

    Supports optional timestep embedding (t_dim) and cond embedding (cond_dim).
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128,
                 t_dim: Optional[int] = None, cond_dim: Optional[int] = None, image_dim: Optional[int] = None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.t_dim = t_dim
        self.cond_dim = cond_dim
        self.image_dim = image_dim 

        # optional projections
        self.t_proj = nn.Linear(t_dim, hidden_dim) if t_dim is not None else None
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim is not None else None
        self.image_proj = nn.Linear(image_dim, hidden_dim) if image_dim is not None else None

        # compute additional channels from embeddings
        extra = 0
        if self.t_proj is not None:
            extra += self.hidden_dim
        if self.cond_proj is not None:
            extra += self.hidden_dim
        if self.image_proj is not None: 
            extra += hidden_dim

        total_in = self.in_channels + extra

        # 3-layer Conv MLP (Patch MLP)
        self.mlp_in = nn.Conv2d(total_in, hidden_dim, kernel_size=1)
        self.mlp_mid = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.mlp_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        # buffer to keep module alive on device
        self.register_buffer("_dummy_feat", torch.zeros(1))

    def forward(self, feat: torch.Tensor, t_emb: Optional[torch.Tensor] = None, cond_emb: Optional[torch.Tensor] = None, image_emb=None):
        if feat is None or feat.dim() != 4:
            raise ValueError("PatchLevelFeatExtractor expects a 4D feature tensor 'feat'.")
        B, C, H, W = feat.shape
        if C != self.in_channels:
            # helpful error pointing to mismatch
            raise RuntimeError(f"PatchLevelFeatExtractor expected in_channels={self.in_channels} but got feat with C={C}")

        device = feat.device
        dtype = feat.dtype

        pieces = [feat]

        def process_emb(x, proj):
            if x is None or proj is None:
                return None
            if not isinstance(x, torch.Tensor):
                raise ValueError("Embeddings must be torch.Tensor or None")
            x = x.to(device=device, dtype=dtype)
            # if extra dims, collapse them
            if x.dim() > 2:
                x = x.flatten(1, -2).mean(dim=1)
            out = proj(x)[:, :, None, None]  # [B, hidden, 1, 1]
            return out.expand(B, self.hidden_dim, H, W)

        t = process_emb(t_emb, self.t_proj)
        c = process_emb(cond_emb, self.cond_proj)
        img = process_emb(image_emb, self.image_proj)

        if t is not None:
            pieces.append(t)
        if c is not None:
            pieces.append(c)
        if img is not None:
            pieces.append(img)

        merged = torch.cat(pieces, dim=1)

        # 3-layer Conv MLP
        h = F.gelu(self.mlp_in(merged))
        h = F.gelu(self.mlp_mid(h))
        logits = self.mlp_out(h)      # [B,1,H,W]

        return logits


# Patch-Level MoE Gate (per-control MLPs)
# Each MLP_k uses same in_channels (per-block control channel).
class PatchLevelMoEGate(ModelMixin, ConfigMixin):
    """
    For a given block, there are N controls (num_experts).
    Each control i has a feature W_i with in_channels = control_in_channels.
    We create N independent PatchLevelFeatExtractor, each with in_channels=control_in_channels.
    Forward: takes list of control features [W1, W2, ..., WN] and returns weights [B, N, H, W].
    """
    @register_to_config
    def __init__(self,
                 num_experts: int,
                 control_in_channels: int,
                 hidden_dim: int = 128,
                 t_dim: Optional[int] = None,
                 cond_dim: Optional[int] = None,
                 image_dim: Optional[int] = None):
        super().__init__()
        self.num_experts = int(num_experts)
        self.control_in_channels = int(control_in_channels)
        self.hidden_dim = hidden_dim
        self.t_dim = t_dim
        self.cond_dim = cond_dim
        self.image_dim = image_dim 

        # create N independent MLP_k, each accepts control_in_channels
        self.expert_mlps = nn.ModuleList([
            PatchLevelFeatExtractor(in_channels=self.control_in_channels, hidden_dim=self.hidden_dim, t_dim=self.t_dim, cond_dim=self.cond_dim, image_dim=self.image_dim )
            for _ in range(self.num_experts)
        ])

        # ensure top-level has at least one buffer for HF/diffusers device inference
        self.register_buffer("_dummy_gate", torch.zeros(1))

    def forward(self, cn_feats_list: List[torch.Tensor], t_emb: Optional[torch.Tensor] = None, cond_emb: Optional[torch.Tensor] = None, image_emb: Optional[torch.Tensor] = None):
        if len(cn_feats_list) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} control features (one per expert), but got {len(cn_feats_list)}")

        all_logits = []
        for k, feat in enumerate(cn_feats_list):
            # each MLP_k processes its own control feature
            logits_k = self.expert_mlps[k](feat, t_emb=t_emb, cond_emb=cond_emb, image_emb=image_emb)  # [B,1,H,W]
            all_logits.append(logits_k)

        logits = torch.cat(all_logits, dim=1)  # [B, N, H, W]
        weights = F.softmax(logits, dim=1)
        return weights


# Router Container (final)
class newControlNetRouter(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 num_experts: int = 2,
                 router_type: str = 'simple_weights',
                 embedding_dim: Optional[int] = None,
                 control_channels_per_block: Optional[List[int]] = None,
                 mid_block_channel: Optional[int] = None,
                 num_routers: int = 12,
                 add_mid_block_router: bool = True,
                 cond_dim: Optional[int] = None,
                 t_dim: Optional[int] = None,
                 backbone_model_name: Optional[str] = None,
                 use_sparsemax: bool = False):
        """
        Args:
          num_experts: number of controls (ControlNet count)
          router_type: 'equal_weights' | 'simple_weights' | 'patch_mlp*' (we treat patch_mlp* as per-control MLPs)
          embedding_dim: if provided as int, used as fallback in_channels for every block (not recommended for SDXL)
          control_channels_per_block: Optional[List[int]] length == num_routers, giving in_channels per block (recommended)
          mid_block_channel: in_channels for mid block (if add_mid_block_router True)
        """
        super().__init__()

        # FIX: ensure top-level buffer so HF/diffusers can infer device even if something odd happens
        self.register_buffer("_dummy_router", torch.zeros(1))

        self.num_experts = int(num_experts)
        self.num_routers = int(num_routers)
        self.router_type = router_type
        self.embedding_dim = embedding_dim
        self.cond_dim = cond_dim
        self.t_dim = t_dim
        self.backbone_model_name = backbone_model_name
        self.use_sparsemax = use_sparsemax
        self.add_mid_block_router = add_mid_block_router

        # prepare per-block in_channels
        if control_channels_per_block is None:
            if embedding_dim is None:
                raise ValueError("Either control_channels_per_block or embedding_dim must be provided.")
            # fallback: repeat embedding_dim across blocks (user-supplied fallback; not ideal for SDXL)
            control_channels_per_block = [embedding_dim for _ in range(self.num_routers)]
        if len(control_channels_per_block) != self.num_routers:
            raise ValueError(f"control_channels_per_block length ({len(control_channels_per_block)}) must equal num_routers ({self.num_routers})")
        self.control_channels_per_block = control_channels_per_block

        # mid block channel
        if self.add_mid_block_router:
            if mid_block_channel is None:
                # fallback to last down block channel
                mid_block_channel = self.control_channels_per_block[-1]
            self.mid_block_channel = mid_block_channel

        # Build per-block routers
        self.down_blocks_router = nn.ModuleList([])
        for i in range(self.num_routers):
            if self.router_type == 'equal_weights':
                self.down_blocks_router.append(EqualWeights(self.num_experts))
            elif self.router_type == 'simple_weights':
                self.down_blocks_router.append(SimpleWeights(self.num_experts))
            elif self.router_type.startswith('patch_mlp'):
                # For block i, control feature per-control channels = control_channels_per_block[i]
                in_ch = self.control_channels_per_block[i]
                self.down_blocks_router.append(PatchLevelMoEGate(
                    num_experts=self.num_experts,
                    control_in_channels=in_ch,
                    hidden_dim=128,
                    t_dim=self.t_dim if 'timestep' in self.router_type else None,
                    cond_dim=self.cond_dim if 'embedding' in self.router_type else None,
                    image_dim=self.image_dim if 'image' in self.router_type else None   # ★ 新增
                ))
            else:
                raise ValueError(f"Unknown router_type: {self.router_type}")

        # mid block router
        if self.add_mid_block_router:
            if self.router_type == 'equal_weights':
                self.mid_block_router = EqualWeights(self.num_experts)
            elif self.router_type == 'simple_weights':
                self.mid_block_router = SimpleWeights(self.num_experts)
            elif self.router_type.startswith('patch_mlp'):
                self.mid_block_router = PatchLevelMoEGate(
                    num_experts=self.num_experts,
                    control_in_channels=self.mid_block_channel,
                    hidden_dim=128,
                    t_dim=self.t_dim if 'timestep' in self.router_type else None,
                    cond_dim=self.cond_dim if 'embedding' in self.router_type else None,
                    image_dim=self.image_dim if 'image' in self.router_type else None   # ★ 新增
                )
            else:
                self.mid_block_router = None
        else:
            self.mid_block_router = None

    def forward(self,
                cn_feats_down: Optional[Union[Dict[str, List[torch.Tensor]], List[torch.Tensor]]] = None,
                cn_feats_mid: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
                router_input: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                sparse_mask=None):
        """
        Returns:
          down_weights: list length num_routers; each element:
              - if scalar router: tensor [1, num_experts]
              - if patch router: tensor [B, num_experts, H, W]
          mid_weights: similar shape or None
        """
        down_weights = []
        mid_weights = None

        # parse router_input embeddings
        t_emb, cond_emb = None, None
        if isinstance(router_input, list):
            if len(router_input) == 3:
                t_emb, cond_emb, image_emb = router_input
            elif len(router_input) == 2:
                cond_emb, t_emb = router_input
            elif len(router_input) == 1:
                cond_emb = router_input[0]
        else:
            cond_emb = router_input

        # iterate routers per down block
        for i, router in enumerate(self.down_blocks_router):
            if self.router_type in ['equal_weights', 'simple_weights']:
                logits = router(router_input)
                w = F.softmax(logits, dim=-1)
                down_weights.append(w)
                continue

            # patch-level: collect per-control features for block i as list
            cn_list_at_i = []

            # expected cn_feats_down as dict[str -> list[tensor]] (multiple controls)
            if cn_feats_down is not None and isinstance(cn_feats_down, dict):
                for feat_list in cn_feats_down.values():
                    if i < len(feat_list):
                        cn_list_at_i.append(feat_list[i])
            elif cn_feats_down is not None and isinstance(cn_feats_down, list):
                # single-control scenario: cn_feats_down is list of blocks
                # but for patch_mlp we expect num_experts controls; allow degenerate case when num_experts==1
                if self.num_experts == 1:
                    cn_list_at_i.append(cn_feats_down[i])
                else:
                    # if user passed list-of-lists incorrectly, raise helpful error
                    raise ValueError("cn_feats_down is a list but router expects multiple control features per block (dict).")

            if len(cn_list_at_i) != self.num_experts:
                raise ValueError(f"Patch-Level Router {i} requires exactly {self.num_experts} ControlNet features for block {i}, got {len(cn_list_at_i)}.")

            # compute weights (patch gate)
            w = router(cn_list_at_i, t_emb=t_emb, cond_emb=cond_emb)  # [B, num_experts, H, W]
            down_weights.append(w)

        # mid block
        if self.mid_block_router is not None:
            if self.router_type in ['equal_weights', 'simple_weights']:
                logits = self.mid_block_router(router_input)
                mid_weights = F.softmax(logits, dim=-1)[0]
            else:
                mid_feats_list = []
                if cn_feats_mid is not None and isinstance(cn_feats_mid, dict):
                    for feat in cn_feats_mid.values():
                        mid_feats_list.append(feat)
                elif cn_feats_mid is not None and not isinstance(cn_feats_mid, dict):
                    if self.num_experts == 1:
                        mid_feats_list.append(cn_feats_mid)

                if len(mid_feats_list) != self.num_experts:
                    raise ValueError(f"Mid block router requires exactly {self.num_experts} ControlNet features, got {len(mid_feats_list)}.")

                mid_weights = self.mid_block_router(mid_feats_list, t_emb=t_emb, cond_emb=cond_emb)

        return down_weights, mid_weights

