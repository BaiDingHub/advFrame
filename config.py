class Config(object):
    def __init__(self):
        self.ENV = 'default'            #当前的环境参数
        self.Introduce = 'Not at the moment'    #对此次实验的描述


        ##################################################GPU配置
        self.GPU = dict(
            use_gpu = True,             #是否使用GPU，True表示使用
            device_id = [0],            #所使用的GPU设备号，type=list
        )


        self.CONFIG = dict(
            dataset_name = 'Mnist',     #所选择的数据集的名称
            model_name = 'LeNet',       #攻击模型的名称
            criterion_name = 'CrossEntropyLoss',       #损失函数的名称
            attack_name = 'BIM',       #设定攻击方法的名称
        )



        #################################################模型选择
        ##########################模型参数
        self.LeNet = dict(
            filepath = '/home/baiding/Desktop/Study/Deep/pretrained/lenet/lenet_1.pkl',     #预训练模型所在的位置
        )



        #################################################损失函数选择



        #################################################数据集
        ##########################数据集参数
        self.Mnist = dict(
            dirname = '/home/baiding/Desktop/Study/Deep/datasets/MNIST/raw',            #MNIST数据集存放的文件夹
            needVector = False,         #False表示得到784维向量数据，True表示得到28*28的图片数据
        )



        #################################################攻击方法
        ##########################FGSM方法
        self.FGSM = dict(
            eps = 0.2,                  #FGSM的控制大小的参数
            isTarget = False,           #控制攻击方式，目标攻击、无目标攻击
            target = 3,               #目标攻击的目标
        )
        ##########################BIM方法
        self.BIM = dict(
            eps = 0.03,                  #BIM的控制大小的参数
            epoch = 10,                 #BIM的迭代次数
            isTarget = True,           #控制攻击方式，目标攻击、无目标攻击
            target = 3,               #目标攻击的目标
        )


        #################################################log
        self.logDir = './log'           #log所在的文件夹
        self.expTime = 1                #第expTime次实验
        self.logFileName = self.CONFIG['dataset_name']+'_'+self.CONFIG['model_name']+"_" + \
            self.CONFIG['criterion_name']+'_'+self.CONFIG['attack_name']+'_'+ \
            str(self.expTime)+'.txt'
        
    

    def logOutput(self):
        log = {}
        log['ENV'] = self.ENV
        log['Introduce'] = self.Introduce
        log['CONFIG'] = self.CONFIG
        for name,value in self.CONFIG.items():
            if hasattr(self,value):
                log[value] = getattr(self,value)
        return log