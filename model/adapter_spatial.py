import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional

from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise separable 2D convolution:
    standard Conv2d(in, out, k, s, p) is replaced by
    depthwise Conv2d(in->in, groups=in) + pointwise 1x1 Conv2d(in->out).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SEBlock2d(nn.Module):
    """
    Squeeze-and-Excitation block for 2D feature maps (B, C, H, W).
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DSResnetBlock2D(nn.Module):
    """
    Lightweight ResNet block for images:
      - standard Conv2d layers are replaced by DepthwiseSeparableConv2d
      - an SEBlock2d is inserted after the second conv

    Interface is compatible with the ResnetBlock2D usage in AdapterSpatioTemporal:
      forward(x, temb=None, output_size=None)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        temb_channels: int = 0,
        eps: float = 1e-6,
        use_in_shortcut: bool = True,
        up: bool = False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps
        self.up = up

        # first norm + nonlinearity
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps)
        self.act1 = nn.SiLU()

        # first depthwise separable conv
        self.conv1 = DepthwiseSeparableConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        # time embedding projection
        self.temb_channels = temb_channels
        if temb_channels is not None and temb_channels > 0:
            self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            self.use_scale_shift_norm = True
        else:
            self.time_emb_proj = None
            self.use_scale_shift_norm = False

        # second norm + nonlinearity
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps)
        self.act2 = nn.SiLU()

        # second depthwise separable conv
        self.conv2 = DepthwiseSeparableConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        # SE attention
        self.se = SEBlock2d(out_channels, reduction=4)

        # shortcut projection if needed
        if in_channels != out_channels or not use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = nn.Identity()

        # optional upsampling (kept to be compatible with original ResnetBlock2D)
        if self.up:
            self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        else:
            self.upsample = None

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        output_size: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            temb: (B, temb_channels), time embedding
            output_size: if provided and up=True, use it as target spatial size
        """
        residual = x

        # handle upsampling first so that residual and main path match in size
        if self.upsample is not None:
            if output_size is not None:
                x = F.interpolate(x, size=output_size, mode="nearest")
                residual = F.interpolate(residual, size=output_size, mode="nearest")
            else:
                x = self.upsample(x)
                residual = self.upsample(residual)

        # main branch
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        scale = shift = None
        if self.time_emb_proj is not None and temb is not None:
            # project temb to (B, 2*C_out) for scale & shift
            temb_out = self.time_emb_proj(temb)  # (B, 2*C_out)
            scale, shift = temb_out.chunk(2, dim=1)

        h = self.norm2(h)
        if scale is not None and shift is not None:
            # FiLM-style modulation
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        elif shift is not None:
            h = h + shift[:, :, None, None]

        h = self.act2(h)
        h = self.conv2(h)

        # SE attention
        h = self.se(h)

        # residual connection
        residual = self.conv_shortcut(residual)
        return h + residual


class AdapterSpatioTemporal(nn.Module):
    """
    Lightweight image-only adapter module.

    - Spatial path: DSResnetBlock2D (+ optional upsampling) + optional BasicTransformerBlock.
    - Temporal path: not implemented; this module is intended for num_frames=1 (images).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,

        # which modules to activate
        add_spatial_resnet: bool = True,
        add_temporal_resnet: bool = True,
        add_spatial_transformer: bool = True,
        add_temporal_transformer: bool = True,

        # resnet arguments
        eps: float = 1e-6,
        temporal_eps: float = None,
        merge_factor: float = 0.5,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
        up_sampling_scale: float = 1.0,

        # transformer arguments
        cross_attention_dim: int = 1024,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.up_sampling_scale = up_sampling_scale

        self.add_spatial_resnet = add_spatial_resnet
        self.add_temporal_resnet = add_temporal_resnet
        self.add_spatial_transformer = add_spatial_transformer
        self.add_temporal_transformer = add_temporal_transformer

        # This implementation is focused on images (num_frames=1).
        # Temporal branches can be added later if needed.
        if self.add_temporal_resnet or self.add_temporal_transformer:
            raise NotImplementedError(
                "Temporal (video) components are not implemented in this lightweight image-only adapter."
            )

        temb_channels = in_channels

        # Time embedding for spatial ResNet blocks
        if self.add_spatial_resnet:
            self.resnet_time_proj = Timesteps(in_channels, True, downscale_freq_shift=0)
            self.resnet_time_embedding = TimestepEmbedding(in_channels, temb_channels)

        # Spatial transformer setup
        if self.add_spatial_transformer:
            self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps)
            self.inner_dim = num_attention_heads * attention_head_dim
            self.num_attention_heads = num_attention_heads
            self.attention_head_dim = attention_head_dim

            self.proj_in = nn.Linear(in_channels, self.inner_dim)
            self.proj_out = nn.Linear(self.inner_dim, in_channels)

        spatial_resnets = []
        spatial_attentions = []

        for i in range(self.num_layers):
            # Spatial ResNet blocks (depthwise separable + SE)
            if self.add_spatial_resnet:
                spatial_resnets.append(
                    DSResnetBlock2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=eps,
                        use_in_shortcut=True,
                        up=True if (i == 0 and self.up_sampling_scale > 1.0) else False,
                    )
                )

            # Spatial transformer blocks
            if self.add_spatial_transformer:
                from diffusers.models.transformers.transformer_temporal import BasicTransformerBlock

                spatial_attentions.append(
                    BasicTransformerBlock(
                        self.inner_dim,
                        self.num_attention_heads,
                        attention_head_dim,
                        cross_attention_dim=cross_attention_dim,
                    )
                )

        if self.add_spatial_resnet:
            self.spatial_resnets = nn.ModuleList(spatial_resnets)
        if self.add_spatial_transformer:
            self.spatial_attentions = nn.ModuleList(spatial_attentions)

    def _prepare_timestep(
        self,
        timestep: Optional[torch.Tensor],
        batch_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Normalize various timestep formats to a 1D tensor of length batch_frames.
        """
        if timestep is None:
            timestep = torch.zeros(batch_frames, device=device, dtype=dtype)
        elif isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=device, dtype=dtype).repeat(batch_frames)
        elif isinstance(timestep, torch.Tensor):
            if timestep.dim() == 0:
                timestep = timestep[None].repeat(batch_frames)
            elif timestep.dim() == 1:
                if timestep.shape[0] == 1:
                    timestep = timestep.repeat(batch_frames)
                elif timestep.shape[0] != batch_frames:
                    # best effort: tile / crop to match batch_frames
                    repeats = (batch_frames + timestep.shape[0] - 1) // timestep.shape[0]
                    timestep = timestep.repeat(repeats)[:batch_frames]
            elif timestep.dim() == 2:
                timestep = timestep.squeeze()
                if timestep.shape[0] != batch_frames:
                    repeats = (batch_frames + timestep.shape[0] - 1) // timestep.shape[0]
                    timestep = timestep.repeat(repeats)[:batch_frames]
            else:
                timestep = timestep.reshape(-1)[:batch_frames]
        else:
            raise TypeError(f"Unsupported timestep type: {type(timestep)}")

        return timestep.to(dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        timestep: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        sparsity_masking=None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states: (B * num_frames, C, H, W); for images, num_frames=1 so B = batch size.
            num_frames: number of frames; this implementation is for num_frames=1 (images).
            timestep: scalar or tensor time step(s).
            encoder_hidden_states: optional context for cross-attention, shape (B, L, C_ctx) or (1, L, C_ctx).
            sparsity_masking: reserved for future use (ignored here).
        """
        batch_frames, channels, height, width = hidden_states.shape

        if num_frames != 1:
            # We do a best-effort reshape but this is not meant for true video processing.
            batch_size = batch_frames // num_frames
        else:
            batch_size = batch_frames

        # 0. process timestep
        timestep = self._prepare_timestep(
            timestep,
            batch_frames=batch_frames,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # 1. prepare ResNet time embedding (if needed)
        if self.add_spatial_resnet:
            resnet_temb = self.resnet_time_proj(timestep)  # (bf, emb_dim)
            resnet_temb = self.resnet_time_embedding(resnet_temb)  # (bf, temb_channels)
            resnet_temb = resnet_temb.to(dtype=hidden_states.dtype)
        else:
            resnet_temb = None

        # main stack
        for i in range(self.num_layers):
            # 1.1 spatial resnet
            if self.add_spatial_resnet:
                _, _, height, width = hidden_states.shape
                output_size = (
                    (int(height * self.up_sampling_scale), int(width * self.up_sampling_scale))
                    if (i == 0 and self.up_sampling_scale != 1.0)
                    else None
                )
                hidden_states = self.spatial_resnets[i](
                    hidden_states, resnet_temb, output_size=output_size
                )  # (bf, C, H, W)
                _, _, height, width = hidden_states.shape  # update in case of upsampling

            # 2. spatial transformer
            if self.add_spatial_transformer:
                residual = hidden_states  # (bf, C, H, W)

                h = self.norm(hidden_states)
                bf, c, hgt, wdt = h.shape
                h = h.permute(0, 2, 3, 1).reshape(bf, hgt * wdt, c)  # (bf, HW, C)
                h = self.proj_in(h)  # (bf, HW, inner_dim)

                # prepare context for cross-attention
                context = encoder_hidden_states
                if context is not None:
                    # allow (B, C_ctx) or (B, L, C_ctx) or (1, L, C_ctx)
                    if context.dim() == 2:
                        context = context.unsqueeze(1)  # (B, 1, C_ctx)
                    if context.shape[0] == 1 and bf > 1:
                        context = context.repeat(bf, 1, 1)

                h = self.spatial_attentions[i](h, encoder_hidden_states=context)  # (bf, HW, inner_dim)
                h = self.proj_out(h)  # (bf, HW, C)
                h = h.reshape(bf, hgt, wdt, c).permute(0, 3, 1, 2)  # (bf, C, H, W)

                hidden_states = residual + h

        return hidden_states


if __name__ == "__main__":
    # Simple sanity check: count parameters for a SDXL-like configuration.
    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    dims = [(64, 64), (64, 64), (64, 64), (32, 32), (32, 32), (32, 32), (16, 16), (16, 16), (16, 16)]
    channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280]

    total_params = 0
    for (h, w), ch in zip(dims, channels):
        adapter = AdapterSpatioTemporal(
            in_channels=ch,
            out_channels=ch,
            num_layers=1,
            add_spatial_resnet=True,
            add_temporal_resnet=False,
            add_spatial_transformer=True,
            add_temporal_transformer=False,
            cross_attention_dim=2048,
            up_sampling_scale=1.0,
        )
        total_params += count_params(adapter)

    print(f"Total trainable parameters across all adapters: {total_params}")
