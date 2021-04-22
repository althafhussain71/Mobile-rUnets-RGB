import os
import glob
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

compose = transforms.Compose([
   transforms.ToTensor(),
])

class DirDataset(Dataset):
    # Referred from: https://github.com/hiepph/unet-lightning
    def __init__(self, img_dir, mask_dir, transforms = compose):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        try:
            self.ids = [s.split('.')[0] for s in os.listdir(self.img_dir)]
        except FileNotFoundError:
            self.ids = []


    # Referred from: https://github.com/hiepph/unet-lightning
    def __len__(self):
        return len(self.ids)


    # Referred from: https://github.com/hiepph/unet-lightning
    def __getitem__(self, i):
        idx = self.ids[i]

        img_files = glob.glob(os.path.join(self.img_dir, idx + '.*'))
        mask_files = glob.glob(os.path.join(self.mask_dir, idx + '.*'))

        assert len(img_files) == 1, f'{idx}: {img_files}'
        assert len(mask_files) == 1, f'{idx}: {mask_files}'

        # use Pillow's Image to read .gif mask
        # Referred from: https://answers.opencv.org/question/185929/how-to-read-gif-in-python/
        img = Image.open(img_files[0])
        mask = Image.open(mask_files[0])
        assert img.size == mask.size, f'{img.shape} # {mask.shape}'

        return self.transforms(img), \
               self.transforms(mask)
