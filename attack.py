import advMethod
import dataLoader
import modelLoader
from modelLoader import *
from user import *
from config import *
from utils import *
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


######################################加载配置
config = Config()
Log = config.logOutput()

######################################加载数据集
dataset_name = config.CONFIG['dataset_name']
TrainSet = getattr(dataLoader,dataset_name+'TrainSet')(**getattr(config,dataset_name))
TestSet = getattr(dataLoader,dataset_name+'TestSet')(**getattr(config,dataset_name))

######################################加载数据加载器
TrainLoader = DataLoader(TrainSet,batch_size=4,shuffle=True,num_workers=4)
TestLoader = DataLoader(TestSet,batch_size=1,shuffle=False,num_workers=1)

######################################加载模型
model_name = config.CONFIG['model_name']
model = getattr(modelLoader,'load'+model_name)(**getattr(config,model_name))

######################################加载损失函数
criterion_name = config.CONFIG['criterion_name']
criterion = getattr(nn,criterion_name)()

######################################加载攻击方式
attack_name = config.CONFIG['attack_name']
attack_method = getattr(advMethod,attack_name)(model,criterion)


######################################加载攻击训练器
attacker = Attacker(model,criterion,config,attack_method)


#####################################开始攻击

###############################攻击一张图片
# x,y = TrainSet.__getitem__(7000)
# x_adv,pertubation,nowLabel = attacker.attackOneImage(x,y)
# print(y,nowLabel)

###############################攻击整个数据集
acc,mean = attacker.attackSet(TestLoader)
Log['acc'] = acc
Log['pertubmean'] = mean


####################################log保存
# filename = os.path.join(config.logDir,config.logFileName)
# f = open(filename,'w')
# logWrite(f,Log)

print(Log)


# # plt.switch_backend('agg')
# plt.imshow(x[0])
# plt.imshow(x_adv[0])
# plt.show()
# plt.imshow(x_adv[0])