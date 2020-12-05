import numpy as np
import torch
from config import *
from tqdm import tqdm


class Attacker(object):
    """[攻击者]

    Args:
        self.model ([]): [要攻击的模型]
        self.criterion ([]): 损失函数
        self.config ([]): 配置类
        self.attack_method ([]): 攻击方法
        self.use_gpu ([bool]): 是否使用GPU
        self.device_ids ([list]): 使用的GPU的id号
        self.attack_name ([str]): 攻击方法名称
        self.is_target ([bool]): 是否进行目标攻击
        self.target ([int]): 目标攻击的目标 
    """
    def __init__(self,model,criterion,config,attack_method):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.attack_method = attack_method


        #########################GPU配置
        self.use_gpu = False
        self.device_ids = [0]
        self.device = torch.device('cpu')
        if self.config.GPU['use_gpu']:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                self.device_ids = self.config.GPU['device_id']
                assert len(self.device_ids) > 0,message
                self.device = torch.device('cuda', self.device_ids[0])
                self.model = self.model.to(self.device)
                if len(self.device_ids) > 1:
                    self.model = nn.DataParallel(model, device_ids=self.device_ids)
                self.use_gpu = True


        #########################攻击信息
        self.attack_name = self.config.CONFIG['attack_name']
        #########################攻击方式---目标攻击设置
        self.is_target = False
        self.target = 0
        if 'is_target' in getattr(self.config,self.attack_name):
            self.is_target = getattr(self.config,self.attack_name)['is_target']
            self.target =  getattr(self.config,self.attack_name)['target']        



    def attack_batch(self, x, y):
        """[对一组数据进行攻击]]

        Args:
            x ([array]): [一组输入数据]
            y (list, 无目标攻击中设置): [输入数据对应的标签]]. Defaults to [0].

        Returns:
            x_advs ([array]): [得到的对抗样本，四维]
            pertubations ([array]): [得到的对抗扰动，四维]
            nowLabels ([array]): [攻击后的样本标签，一维]
        """ 
        
        #放到GPU设备中
        # if type(x) is np.ndarray:
        #     x = torch.from_numpy(x)
        # if type(y) is not torch.tensor:
        #     y = torch.Tensor(y.float())
        x = torch.Tensor(x)
        y = torch.Tensor(y)

        x = x.to(self.device).float()
        y = y.to(self.device).long()

        x_advs, pertubations, logits, nowLabels = self.attack_method.attack(
            x, y, **getattr(self.config, self.config.CONFIG['attack_name']))

        return x_advs,pertubations,nowLabels 


    def attack_one_img(self, x, y=[0]):   
        """[攻击一个图片]

        Args:
            x ([array]): [一个输入数据]
            y (list, 无目标攻击中设置): [输入数据对应的标签]]. Defaults to [0].

        Returns:
            x_adv ([array]): [得到的对抗样本，三维]
            pertubation ([array]): [得到的对抗扰动，三维]
            nowLabel ([int]): [攻击后的样本标签]
        """
        x = np.expand_dims(x, axis=0)                    #拓展成四维
        y = np.array(list([y]))                         #转成矩阵
        x_adv,pertubation,nowLabel = self.attack_batch(x, y)
        return x[0], x_adv[0], pertubation[0] ,nowLabel[0]

    def attack_set(self, data_loader):
        """[对一个数据集进行攻击]

        Args:
            data_loader ([DataLoader]): [数据加载器]

        Returns:
            acc [float]: [攻击后的准确率]
            mean [float]: 平均扰动大小
        """
        success_num = 0
        data_num = 0
        pertubmean = []
        for idx,(x,y) in enumerate(tqdm(data_loader)):
            x_advs ,pertubations, nowLabels = self.attack_batch(x, y)
            if self.is_target:
                success_num += ((self.target == nowLabels) == (self.target != y.numpy())).sum()
                data_num += (self.target != y.numpy()).sum()
            else:
                data_num +=  x.shape[0]
                success_num += (y.numpy() != nowLabels).sum()
            pertubmean.append(pertubations.mean())
        mean = np.mean(pertubmean)
        acc = 1 - success_num / data_num
        return acc, mean
  
        