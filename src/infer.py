import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import ConvAutoEncoder


def load_image(path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return image, tensor


def infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = ConvAutoEncoder().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    original_pil, image_tensor = load_image(args.image, args.image_size)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        reconstructed = model(image_tensor)

    original = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    recon = reconstructed.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    error = np.mean((original - recon) ** 2, axis=2)
    anomaly_score = float(error.mean())

    heatmap = (error - error.min()) / (error.max() - error.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    original_uint8 = np.uint8(original * 255)
    recon_uint8 = np.uint8(recon * 255)

    original_bgr = cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR)
    recon_bgr = cv2.cvtColor(recon_uint8, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

    out = np.hstack([original_bgr, recon_bgr, heatmap_color, overlay])

    output_path = Path(args.output_dir) / "anomaly_result.jpg"
    cv2.imwrite(str(output_path), out)

    print(f"Anomaly score: {anomaly_score:.6f}")
    print(f"Saved result to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())
