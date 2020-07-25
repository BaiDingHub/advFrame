import torch
import torch.nn as nn
import numpy as np

class FGSM(object):
    """
    FGSM
    """
    def __init__(self,model,criterion):
        """[summary]

        Args:
            model ([type]): [要攻击的模型]
            criterion ([type]): [损失函数]
        """
        super(FGSM,self).__init__()
        self.model = model
        self.criterion = criterion

    def attack(self,x,y=0,eps=0.03,isTarget=False,target=0):
        """[summary]

        Args:
            x ([array[float] or tensor]): [输入样本，四维]
            y (array[long], optional): [样本标签]. Defaults to 0.
            eps (float, optional): [控制FGSM精度]. Defaults to 0.03.
            isTarget (bool, optional): [是否为目标攻击]. Defaults to False.
            targets (int, optional): [攻击目标]. Defaults to 0.
            

        Returns:
            x_adv [array]: [对抗样本]
            pertubation [array]: [对抗扰动]
            pred [array]: [攻击后的标签]
        """
        if isTarget:
            x_adv,pertubation = self._attackWithTarget(x,target,eps)
            message = "At present, we haven't implemented the Target attack algorithm "
            assert x_adv is not None,message
        else:
            x_adv,pertubation = self._attackWithNoTarget(x,y,eps)
            message = "At present, we haven't implemented the No Target attack algorithm "
            assert x_adv is not None,message

        logits = self.model(x_adv)
        pred = logits.argmax(1)
        
        return x_adv.cpu().detach().numpy(),pertubation.cpu().detach().numpy(),pred.cpu().detach().numpy()


    def _attackWithNoTarget(self,x,y,eps):
        
        x_adv = x
        x_adv.requires_grad = True
        logits = self.model(x_adv)
            
        loss = self.criterion(logits, y)
        self.model.zero_grad()
        # if x_adv.grad is not None:
        #     x_adv.grad.data.fill_(0)
        loss.backward()

        data_grad = x_adv.grad.data
        #得到梯度的符号
        sign_data_grad = data_grad.sign()

        x_adv = x_adv + eps*sign_data_grad
        x_adv = torch.clamp(x_adv, 0, 1)
        pertubation = x_adv - x

        return x_adv,pertubation

    def _attackWithTarget(self,x,target,eps):
        target = torch.tensor([target]*x.shape[0]).cuda()
        x_adv = x
        x_adv.requires_grad = True
        logits = self.model(x_adv)
            
        loss = self.criterion(logits, target)
        self.model.zero_grad()
        # if x_adv.grad is not None:
        #     x_adv.grad.data.fill_(0)
        loss.backward()

        data_grad = x_adv.grad.data
        #得到梯度的符号
        sign_data_grad = data_grad.sign()

        x_adv = x_adv - eps*sign_data_grad
        x_adv = torch.clamp(x_adv, 0, 1)
        pertubation = x_adv - x

        return x_adv,pertubation