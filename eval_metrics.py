import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import torch
from torchvision import transforms
from torchvision.models import inception_v3
from scipy import linalg
from datasets import load_dataset

def get_image_list(dir_path):
    return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def load_and_preprocess_image(path, size=(256, 256)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    return np.array(img)

def calc_mse_ssim(real_path, fake_path):
    real = load_and_preprocess_image(real_path)
    fake = load_and_preprocess_image(fake_path)
    real_gray = np.dot(real[...,:3], [0.2989, 0.5870, 0.1140])
    fake_gray = np.dot(fake[...,:3], [0.2989, 0.5870, 0.1140])
    m = mse(real_gray, fake_gray)
    s = ssim(real_gray, fake_gray, data_range=fake_gray.max() - fake_gray.min())
    return m, s

# --- FID ---
def get_inception_model():
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

def extract_features(img_paths, model, device):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for p in img_paths:
            img = Image.open(p).convert('RGB')
            img_t = preprocess(img).unsqueeze(0).to(device)
            feat = model(img_t).cpu().numpy()
            feats.append(feat)
    return np.concatenate(feats, axis=0)

def calculate_fid(real_feats, fake_feats):
    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def main(real_dir, fake_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    real_files = get_image_list(real_dir)
    fake_files = get_image_list(fake_dir)
    
    real_set = set(real_files)
    fake_set = set(fake_files)
    common = sorted(real_set & fake_set)
    assert len(common) > 0, "There are no pictures with the same name in the two directories!"
    print(f"fine {len(common)} pairs of images with the same name")

    mses, ssims = [], []
    for f in common:
        m, s = calc_mse_ssim(os.path.join(real_dir, f), os.path.join(fake_dir, f))
        mses.append(m)
        ssims.append(s)
    print(f"Average MSE:  {np.mean(mses):.4f}")
    print(f"Average SSIM: {np.mean(ssims):.4f}")

    model = get_inception_model().to(device)
    real_paths = [os.path.join(real_dir, f) for f in common]
    fake_paths = [os.path.join(fake_dir, f) for f in common]
    
    real_feats = extract_features(real_paths, model, device)
    fake_feats = extract_features(fake_paths, model, device)
    fid = calculate_fid(real_feats, fake_feats)
    print(f"FID: {fid:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True, help="real figures directory")
    parser.add_argument("--fake_dir", type=str, required=True, help="fake figures directory")
    args = parser.parse_args()
    main(args.real_dir, args.fake_dir)