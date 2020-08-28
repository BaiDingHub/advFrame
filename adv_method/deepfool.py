import numpy as np
import torch
import torch.nn as nn


class DeepFool(object):
    """[DeepFool]

    Args:
        self.model ([]): 要攻击的模型
        self.criterion ([]): 损失函数
    """
    def __init__(self, model, criterion):
        """[summary]

        Args:
            model ([type]): [要攻击的模型]
            criterion ([type]): [损失函数]
        """
        super(DeepFool,self).__init__()
        self.model = model
        self.criterion = criterion

    def attack(self, x, y=0, max_iter=10, is_target=False, target=0):
        """[summary]

        Args:
            x ([array[float] or tensor]): [输入样本，四维]
            y (array[long], optional): [样本标签]. Defaults to 0.
            eps (float, optional): [控制FGSM精度]. Defaults to 0.03.
            is_target (bool, optional): [是否为目标攻击]. Defaults to False.
            targets (int, optional): [攻击目标]. Defaults to 0.
            
        Returns:
            x_adv [array]: [对抗样本]
            pertubation [array]: [对抗扰动]
            pred [array]: [攻击后的标签]
        """
        if is_target:
            x_adv,pertubation = self._attackWithTarget(x, target)
            message = "At present, we haven't implemented the Target attack algorithm "
            assert x_adv is not None,message
        else:
            x_adv,pertubation = self._attackWithNoTarget(x, y, max_iter)
            message = "At present, we haven't implemented the No Target attack algorithm "
            assert x_adv is not None,message

        logits = self.model(x_adv)
        pred = logits.argmax(1)
        
        return x_adv.cpu().detach().numpy(), pertubation.cpu().detach().numpy(), pred.cpu().detach().numpy()


    def _attackWithNoTarget(self, x, y, max_iter):
        
        x_advs = []
        pertubations = []
        for b in range(x.shape[0]):
            # 选择一张图片来进行查找（DeepFool目前只能一张一张的送入）
            image = x[b:b+1, :, :, :]
            x_adv = image
            
            # 得到原始的预测结果
            predict_origin = self.model(x_adv).cpu().detach().numpy()
            # 所有的类别数目
            classes_num = predict_origin.shape[1]
            # 得到原始的分类结果
            classes_origin = np.argmax(predict_origin, axis= 1)[0]
            
            

            for i in range(max_iter):
                x_adv.requires_grad = True
                pred = self.model(x_adv)[0]
                _,classes_now = torch.max(pred, 0)
                pred_origin = pred[classes_origin]
                grad_origin = torch.autograd.grad(pred_origin, x_adv, retain_graph= True, create_graph= True)[0]
                if classes_now != classes_origin:
                    break

                l = classes_origin
                l_value = np.inf
                l_w = None
                for k in range(classes_num):
                    if k == classes_origin:
                        continue
                    pred_k = pred[k]
                    grad_k = torch.autograd.grad(pred_k, x_adv, retain_graph= True, create_graph= True)[0]
                    w_k = grad_k - grad_origin
                    f_k = pred_k - pred_origin
                    value = torch.abs(f_k)/(torch.norm(w_k)**2)
                    if value < l_value :
                        l_value = value
                        l = k
                        l_w = w_k
                r = (1+0.02)*l_value * l_w
                x_adv = x_adv + r
                x_adv = x_adv.detach()
                x_adv = torch.clamp(x_adv, 0, 1)
                

            # x_adv = torch.clamp(x_adv, 0, 1)
            pertubation = x_adv - image

            # x_adv = x_adv.unsqueeze(0)
            # pertubation = pertubation.unsqueeze(0)

            x_advs.append(x_adv)
            pertubations.append(pertubation)
        
        x_advs = torch.cat(x_advs, dim = 0)
        pertubations = torch.cat(pertubations, dim = 0)

        return x_advs, pertubations

    def _attackWithTarget(self, x, target):
        
        raise NotImplementedError