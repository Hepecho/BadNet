import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import join as ospj


class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'CCNN'
        self.log_dir = ospj('log', self.model_name)
        self.ckpt_dir = ospj('checkpoint', self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        # self.dropout = 0.5                                            # 随机失活
        self.epochs = 20                                                # epoch数
        self.batch_size = 32                                            # mini-batch大小

        # skip-gram模型参数
        self.embed_size = 128                                           # 向量的维度
        self.window_size = 5                                            # 上下文窗口大小
        self.learning_rate = 0.01                                       # 学习率 alpha
        self.iter = 5                                                   # 迭代次数 epochs
        self.workers = 3                                                # 线程数
        self.sg = 1                                                     # 设定为word2vec的skip-gram模型
        self.hs = 1                                                     # 使用Hierarchical Softmax
        self.min_count = 0                                              # 忽略词频小于此值的单词


class Model(torch.nn.Module):
    """CCNN"""

    def __init__(self, config):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # (B, 1, 28, 28)
                out_channels=16,
                kernel_size=5,
                stride=1
            ),  # (B, 16, 24, 24)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)  # 默认步长跟池化窗口大小一致 -> (B, 16, 12, 12)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # (B, 16, 12, 12)
                out_channels=32,
                kernel_size=5,
                stride=1
            ),  # (B, 8, 8, 32)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)  # (B, 32, 4, 4)
        )

        self.flatten = nn.Flatten()  # torch.nn.Flatten(start_dim=1,end_dim=-1) 默认从第1维到-1维展平，batch为第0维
        # (B, 32*4*4)
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()
        )  # (B, 512)

        self.fc2 = nn.Linear(512, 10)
        # nn.Softmax(dim=1)
        # (B, 10)

    def forward(self, x):
        # print(x.shape)
        # x = [B, 1, 28, 28]

        x = self.conv1(x)
        # x = [B, 16, 12, 12]

        x = self.conv2(x)
        # x = [B, 32, 4, 4]

        x = self.flatten(x)
        # x = [B, 32*4*4]

        x = self.fc1(x)
        # x = [B, 512]

        x = self.fc2(x)
        # x = [B, 10]
        return x
