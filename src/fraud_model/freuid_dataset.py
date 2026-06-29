from torch.utils.data import Dataset
from PIL import Image
import os

class FreuidDataset(Dataset):
    def __init__(self, dataset_path,  data, transform=None):
        super().__init__()
        self.labels = data['labels'].tolist()
        self.image_paths = data['paths'].tolist()
        self.regions = data['regions'].tolist()
        self.transform = transform
        self.dataset_path = dataset_path
        self._region_ids: dict[str, int] = {
            'BENIN/DL': 1,
            'EGYPT/DL': 2,
            'GUINEA/DL': 3,
            'MOZAMBIQUE/DL': 4,
            'MAURITIUS/ID': 5
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = 1 if self.labels[index] else 0
        img = Image.open(self.dataset_path + '/train/' + img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label