import torch
from network import Network
from dataset import DeepFakeDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet = Network('wide_resnet101_2', device)
resnet.load('warmup+cosine_annealing_warm_restarts_lr=0.002_e=96')
densenet1 = Network('densenet169', device)
densenet1.load('warmup+cosine_annealing_warm_restarts_lr=0.002_e=96')
densenet2 = Network('densenet169', device)
densenet2.load('warmup+cosine_annealing_warm_restarts_lr=0.002_e=72')

# the format of the text file should be:
# <filename: string> <target: 0 | 1>
txt_path = 'INSERT THE PATH TO THE TEXT FILE'

dataset = DeepFakeDataset(True, device, txt_path)
loader = DataLoader(dataset, 1, True)

conf = [0, 0], [0, 0]
targets, preds = [], []

resnet.eval()
densenet1.eval()
densenet2.eval()

with torch.no_grad():
  for x, y in loader:
    x, y = (i.to(device) for i in (x, y))
    pred1 = resnet(x)
    pred2 = densenet1(x)
    pred3 = densenet2(x)
    pred = 0.5 * pred1 + 0.3 * pred2 + 0.2 * pred3
    for pred, y in zip(pred, y):
      preds.append(pred.detach().cpu())
      pred, y = (1 if x > 0.5 else 0 for x in (pred, y))
      targets.append(y)
      conf[pred][y] += 1

print(f'AUC: {roc_auc_score(targets, preds)}')
Trainer.metrics(conf)
