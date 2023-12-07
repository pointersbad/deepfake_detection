import os
import torch
from signal import *
from torch.optim import SGD
from torch.nn import BCELoss
from trainer import Trainer
from network import Network
from scheduler import Sequential, LambdaLR, SWA
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, LambdaLR


import matplotlib.pyplot as plt
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {str(device).upper()}.\n")


def lr_schedule1(optimizer, *_):
  return Sequential(
      schedulers=(LambdaLR(optimizer, lambda _: 1)),
      verbose=True), 'baseline'


def lr_schedule2(optimizer, *_):
  warmup_duration = 8
  warmup = LambdaLR(optimizer, lambda e: e / warmup_duration)
  step = MultiStepLR(optimizer, [8, 24, 32], 0.25)
  return Sequential(
      schedulers=(warmup, step),
      milestones=(warmup_duration),
      verbose=True), 'warmup+multistep'


def lr_schedule3(optimizer, model, lr):
  warmup_duration = 4
  annealing_duration = 56
  warmup = LambdaLR(optimizer, lambda e: e / warmup_duration)
  annealing = CosineAnnealingWarmRestarts(optimizer, 8, 2, lr * 1e-2)
  swa = SWA(optimizer, lr * 3e-2, model)
  return Sequential(
      schedulers=(warmup, annealing, swa),
      milestones=(warmup_duration, annealing_duration),
      verbose=True), 'warmup+cosine_annealing_warm_restarts'


for get_scheduler in (lr_schedule3,):
  for lr, _lr in ((2e-3, 6e-5),):
    epochs = 96
    model = Network('densenet169', device)
    optimizer = SGD([
        {'params': model.backbone.parameters(), 'lr': _lr},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=4e-4, nesterov=True, momentum=0.99)
    scheduler, details = get_scheduler(optimizer, model, lr)
    details = '_'.join((details, f'lr={lr}'))
    criterion = BCELoss()
    trainer = Trainer(model, criterion, optimizer, scheduler, details=details)

    for sig in (SIGABRT, SIGILL, SIGINT, SIGSEGV, SIGTERM):
      def clean(*_):
        trainer.save(sig.name)
        os._exit(0)
      signal(sig, clean)

    state_dict = trainer.train(epochs)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning rates')
    for running_lr in np.transpose(state_dict['scheduler']['lr']):
      ax2.plot(running_lr)
    ax1.plot(state_dict['loss']['train'], color='red')
    ax1.plot(state_dict['loss']['eval'], color='green')
    fig.tight_layout()
    os.makedirs('visuals', exist_ok=True)
    fig.savefig(f'visuals/{details}.png')
    fig.clear()
