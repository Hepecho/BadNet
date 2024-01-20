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
        self.mode = 'ij'                                                # 攻击模式 ['raw', 'ij', 'a2a']

        # self.dropout = 0.5                                            # 随机失活
        self.epochs = 25                                                # epoch数  20
        self.batch_size = 32                                            # mini-batch大小  32
        self.learning_rate = 0.0085                                       # 学习率 alpha  0.01
        if self.mode == 'raw':
            self.need_backdoor = False
        else:
            self.need_backdoor = True                                   # 是否植入后门

        # backdoor参数
        self.ij_class = [0, 8]                                          # 单目标攻击：向i中添加trigger，标签标记为j
        if self.mode == 'a2a':
            self.a2a_attack = True                                      # all-to-all attack：i中添加trigger，标签标记为i+1
        else:
            self.a2a_attack = False
        self.p = 0.05                                                   # 投毒比例，中毒图像占所有图像的比例
        self.trigger_size = 1                                           # trigger大小


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
