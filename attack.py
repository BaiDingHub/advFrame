import adv_method
import data_loader
import model_loader
from model_loader import *
from user import *
from config import *
from utils import *
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


## Load Config
config = Config()
Log = config.log_output()

## Load Dataset
dataset_name = config.CONFIG['dataset_name']
train_set = getattr(data_loader, dataset_name + 'TrainSet')(**getattr(config, dataset_name))
test_set = getattr(data_loader, dataset_name + 'TestSet')(**getattr(config, dataset_name))

### Load Dataloader
train_loader = DataLoader(train_set, batch_size = 4, shuffle = True, num_workers = 4)
test_loader = DataLoader(test_set, batch_size = 32, shuffle = False, num_workers = 1)

## Load Model
model_name = config.CONFIG['model_name']
model = getattr(model_loader, 'load' + model_name)(**getattr(config, model_name))

## Load Loss
criterion_name = config.CONFIG['criterion_name']
criterion = getattr(nn, criterion_name)()

## Load Adv_Method
attack_name = config.CONFIG['attack_name']
attack_method = getattr(adv_method, attack_name)(model, criterion)


## Load Attacker
attacker = Attacker(model, criterion, config, attack_method)


##  Start Attack

## Config Log File
figure_dir = config.Checkpoint['figure_dir']
ensure_dir(figure_dir)
log_file = os.path.join(config.Checkpoint['log_dir'], config.Checkpoint['log_filename'])

##  Attack One Img
# x,y = train_set.__getitem__(7000)
# x, x_adv, pertubation, nowLabel = attacker.attack_one_img(x,y)

# img_file = os.path.join(figure_dir,'real_adv_img.png')
# plt.subplot(2,1,1)  #构建一个2行1列的子图，此处在第一个子图进行绘制
# plt.imshow(x[0])
# plt.subplot(2,1,2)  #此处在第二个子图进行绘制
# plt.imshow(x_adv[0])
# plt.savefig(img_file)

# print('real_label:{}, fake_label:{}'.format(y, nowLabel))

## Attack DataSet
acc,mean = attacker.attack_set(test_loader)
Log['acc'] = acc
Log['pertubmean'] = mean


## Save Log
f = open(log_file,'w')
log_write(f, Log)

print(Log)


# # plt.switch_backend('agg')
# plt.imshow(x[0])
# plt.imshow(x_adv[0])
# plt.show()
# plt.imshow(x_adv[0])