from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.root_dir = Path(root_dir)
        self.image_paths = sorted(
            list(self.root_dir.glob("*.jpg")) +
            list(self.root_dir.glob("*.jpeg")) +
            list(self.root_dir.glob("*.png"))
        )

        if not self.image_paths:
            raise ValueError(f"No images found in {self.root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)
