import torch
from argparse import ArgumentParser
from network import Network
from dataset import DeepFakeDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--path')
# the format of the text file should be:
# <filename: string> <target: 0 | 1>
txt_path = parser.parse_args().path
dataset = DeepFakeDataset(True, device, txt_path)
loader = DataLoader(dataset)

resnet = Network('wide_resnet101_2', device)
resnet.load('warmup+cosine_annealing_warm_restarts_lr=0.002_e=96')
resnet.eval()
densenet = Network('densenet169', device)
densenet.load('warmup+cosine_annealing_warm_restarts_lr=0.002_e=96')
densenet.eval()
efficientnet = Network('efficientnet_v2_s', device)
efficientnet.load('warmup+multistep+swa_lr=0.008_e=27')
efficientnet.eval()


conf = [0, 0], [0, 0]
targets, preds = [], []
with torch.no_grad():
  for x, y in loader:
    x, y = (i.to(device) for i in (x, y))
    pred1 = resnet(x)
    pred2 = densenet(x)
    pred3 = efficientnet(x)
    pred = 0.4 * pred1 + 0.2 * pred2 + 0.4 * pred3
    for pred, y in zip(pred, y):
      preds.append(pred.detach().cpu())
      pred, y = (1 if x > 0.5 else 0 for x in (pred, y))
      targets.append(y)
      conf[pred][y] += 1
Trainer.metrics(conf)
print(f'AUC: {roc_auc_score(targets, preds)}')
