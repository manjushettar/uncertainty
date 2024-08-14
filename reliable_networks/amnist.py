import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms
torch.manual_seed(19)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

new_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
    transforms.RandomChoice([
        transforms.ColorJitter(10), 
        transforms.GaussianBlur(9), 
        transforms.RandomInvert(1), 
        transforms.RandomErasing(1)
        ]),
])

class AMNIST(Dataset):
    def __init__(self, original_data, transformed = False) -> None:
        self.data = original_data
        self.transformed = transformed
        self.transform = new_transform if transformed else transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.transform(img)
        return img, label