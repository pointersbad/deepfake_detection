import torch
import random
from glob import glob
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset


class DeepFakeDataset(Dataset):
  def __vectorize(self, x, y):
    y = y - (y - 0.5) * 0.2
    y = torch.tensor(y)
    add_transforms = () if self.validation else (
        T.RandomHorizontalFlip(), T.RandomRotation(degrees=15))
    transform = T.Compose(
        (T.Resize(256), T.CenterCrop(224), *add_transforms, T.ToTensor()))
    with Image.open(x) as image:
      x = transform(image).to(self.device)
    add_transforms = () if self.validation else (
        T.RandomErasing(scale=(0.1, 0.2)),)
    transform = T.Compose((
        *add_transforms,
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ))
    return transform(x).detach().cpu(), y

  def __init__(self, validation=False, device=None, txt_path=None, N=None):
    self.validation = validation
    self.device = device
    if not txt_path:
      folder = f"{'val' if validation else 'train'}"
      fake = (f'FakeManipulation-{i+1}/**/*.jpg' for i in range(5))
      fake = (glob(f'../data/{folder}/{sub}', recursive=True) for sub in fake)
      real = (f'Real-{i+1}/**/*.jpg' for i in range(4))
      real = (glob(f'../data/{folder}/{sub}', recursive=True) for sub in real)
      d = [(x, y) for y, t in enumerate((fake, real)) for f in t for x in f]
    else:
      with open(txt_path, 'r') as f:
        d = []
        for line in f:
          x, y = line.rstrip().split()
          d.append((x, int(y)))
    random.seed(667)
    self.data = random.sample(d, k=len(d))[:N]

  def __getitem__(self, index):
    return self.__vectorize(*self.data[index])

  def __len__(self):
    return len(self.data)
