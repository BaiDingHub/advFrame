import numpy as np
import torch
from advMethod import *
from config import *
from tqdm import tqdm


class Attacker(object):
    """[攻击者]

    Args:
        object ([type]): [description]
    """
    def __init__(self,model,criterion,config,attack_method):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.attack_method = attack_method
        #########################GPU配置
        self.use_gpu = False
        self.device_ids = self.config.GPU['device_id']
        if self.device_ids:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                assert len(self.device_ids) > 0,message
                self.model = self.model.cuda(self.device_ids[0])
                if len(self.device_ids) > 1:
                    self.model = nn.DataParallel(model, device_ids=self.device_ids)
                self.use_gpu = True
        #########################攻击信息
        self.attack_name = self.config.CONFIG['attack_name']
        #########################攻击方式---目标攻击设置
        self.isTarget = False
        self.target = 0
        if 'isTarget' in getattr(self.config,self.attack_name):
            self.isTarget = getattr(self.config,self.attack_name)['isTarget']
            self.target =  getattr(self.config,self.attack_name)['target']        



    def attackOnce(self,x,y):
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
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
        if type(y) is not torch.tensor:
            y = torch.Tensor(y.float())
        if self.use_gpu:
            x = x.cuda(self.device_ids[0]).float()
            y = y.cuda(self.device_ids[0]).long()
        x_advs,pertubations,nowLabels = self.attack_method.attack(x,y,**getattr(self.config,self.config.CONFIG['attack_name']))
        return x_advs,pertubations,nowLabels 


    def attackOneImage(self,x,y=[0]):   
        """[summary]

        Args:
            x ([array]): [一个输入数据]
            y (list, 无目标攻击中设置): [输入数据对应的标签]]. Defaults to [0].

        Returns:
            x_adv ([array]): [得到的对抗样本，三维]
            pertubation ([array]): [得到的对抗扰动，三维]
            nowLabel ([int]): [攻击后的样本标签]
        """
        x = np.expand_dims(x,axis=0)                    #拓展成四维
        y = np.array(list([y]))                         #转成矩阵
        x_adv,pertubation,nowLabel = self.attackOnce(x,y)
        return x_adv[0],pertubation[0],nowLabel[0]

    def attackSet(self,dataLoader):
        sucessNum = 0
        dataNum = 0
        pertubmean = []
        for idx,(x,y) in enumerate(tqdm(dataLoader)):
            x_advs,pertubations,nowLabels = self.attackOnce(x,y)
            if self.isTarget:
                sucessNum += ((self.target ==nowLabels) == (self.target !=y.numpy())).sum()
                dataNum += (self.target != y.numpy()).sum()
            else:
                dataNum +=  x.shape[0]
                sucessNum += (y.numpy()!=nowLabels).sum()
            pertubmean.append(pertubations.mean())
        mean = np.mean(pertubmean)
        acc = 1-sucessNum/dataNum
        return acc,mean
  
        