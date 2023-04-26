import os
from torch import stack
from torch.utils.data import Dataset
from torchvision.io import read_image

class BSDDataset(Dataset):

    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.items = next(os.walk(self.img_dir))[1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        blurred_path = os.path.join(self.img_dir, self.items[idx], 'Blur', 'RGB')
        sharp_path = os.path.join(self.img_dir, self.items[idx], 'Sharp', 'RGB')
        
        blurred_frames = next(os.walk(blurred_path))[2]
        sharp_frames = next(os.walk(sharp_path))[2]

        blurred_frames.sort()
        sharp_frames.sort()

        blurred_video_tensor = stack([read_image(os.path.join(blurred_path, bf)) for bf in blurred_frames])
        sharp_video_tensor = stack([read_image(os.path.join(sharp_path, sf)) for sf in sharp_frames])
        return blurred_video_tensor, sharp_video_tensor

