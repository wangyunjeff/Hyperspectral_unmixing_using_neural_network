from __future__ import print_function

import configargparse

def model_opts(parser):
    """
    模型的超参数设置
    """
    group = parser.add_argument_group('Model AE')
    group.add('--encoder_type', '-encoder_type', type=str, default='deep',
    choices=['deep', 'shallow'],
    help="选择模型的两种结构，深层（deep）和前层（shallow）"
         "Options are: [deep|shallow]")
         
    # SLReLU unavailable, add assert in main    
    group.add('--soft_threshold', '-soft_threshold', type=str, default='SReLU',
    choices=['SReLU', 'SLReLU'],
    help="Options are: [SReLU|SLReLU]")
         
    group.add('--activation', '-activation', type=str, default='Leaky-ReLU',
    choices=['ReLU', 'Leaky-ReLU', 'Sigmoid'],
    help="激活函数"
         "[ReLU|Leaky-ReLU|Sigmoid]")
         
def extract_opts(parser):
    """
    训练参数设置
    """
    group = parser.add_argument_group('General')
    group.add('--src_dir', '-src_dir', type=str, default='../data/datasets/Samson/',
    help="数据集路径")
    
    group.add('--ckpt', '-ckpt', type=str, default="../tools/logs/hyperspecae_final.pt",
    help="模型保存路径")
          
    group.add('--save_dir', '-save_dir', type=str, default="../imgs/",
    help="模型提取的丰度和端元保存路径")
    
    group.add('--num_bands', '-num_bands', type=int, default=156,
    help="输入高光谱图像的波段数")
    
    group.add('--end_members', '-end_members', type=int, default=3,
    help="需要提取的高光谱图像的端元数")
    
    group = parser.add_argument_group('Hyperparameters')
    group.add('--batch_size', '-batch_size', type=int, default=20,
    help="batch_size")
    
    group.add('--learning_rate','-learning_rate', type=float, default=1e-3,
    help="学习率")
    
    group.add('--epochs','-epochs', type=int, default=100,
    help="epochs")
    
    group.add('--gaussian_dropout', '-gaussian_dropout', type=float, default=0.2,
    help="用于正则化的乘性高斯噪声的均值")
    
    group.add('--threshold', '-threshold', type=float, default=1.0,
    help="soft-thresholding的大小")
    
    
         
         