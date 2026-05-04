import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ConvAutoEncoder
from dataset import ImageFolderDataset


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset = ImageFolderDataset(args.train_dir, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = ConvAutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for images, _ in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}")

        torch.save(
            model.state_dict(),
            Path(args.output_dir) / f"autoencoder_epoch_{epoch}.pth"
        )

    torch.save(model.state_dict(), Path(args.output_dir) / "autoencoder_final.pth")
    print("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
