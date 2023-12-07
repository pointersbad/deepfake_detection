import torch
from torch.nn import Module, Sequential, Dropout, Linear, CELU, Sigmoid, Identity
from torchvision.models.densenet import *
from torchvision.models.resnet import *


class Network(Module):
  def __init__(self, name, device=None):
    super(Network, self).__init__()
    self.name = name
    self.backbone = globals()[name](True)
    head = 'fc' if hasattr(self.backbone, 'fc') else 'classifier'
    in_features = getattr(self.backbone, head).in_features
    setattr(self.backbone, head, Identity())
    self.classifier = Sequential(
        Linear(in_features, 512),
        CELU(0.1),
        Dropout(0.4),
        Linear(512, 1),
        Sigmoid()
    )
    self.to(device)

  def freeze(self):
    for param in self.backbone.parameters():
      param.requires_grad = False

  def thaw(self):
    for param in self.backbone.parameters():
      param.requires_grad = True

  def load(self, details):
    is_cuda = next(self.backbone.parameters()).is_cuda
    device = torch.device('cuda' if is_cuda else 'cpu')
    state_dict = torch.load(f'weights/{self.name}_{details}.pt', device)
    if 'scheduler' in state_dict.keys() \
            and 'SWA' in state_dict['scheduler'].keys():
      weights = state_dict['scheduler']['SWA']['model']
      for key in list(weights.keys()):
        weights[key.replace('module.', '')] = weights.pop(key)
      weights.pop('n_averaged')
    else:
      weights = state_dict['model']
    self.load_state_dict(weights)

  def forward(self, x):
    x = self.backbone(x)
    x = self.classifier(x)
    return x.squeeze(1)
