import torch
import os
from os.path import join as ospj
import torch.optim as optim
import torch.nn as nn
from importlib import import_module
from runx.logx import logx
import numpy as np
import argparse

from load_dataset import get_dataset
from trainer import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BadNet')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='CCNN')
    args = parser.parse_args()

    import sys
    sys.path.insert(0, sys.path[0] + "/../")
    x = import_module('model.' + args.model)
    config = x.Config()
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.ckpt_dir, exist_ok=True)
    logx.initialize(logdir=config.log_dir, coolname=False, tensorboard=False)
    logx.msg(str(args))
    logx.msg(str(config.__dict__))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # 获取数据集
    train_loader, valid_loader, test_loader = get_dataset(config)

    # 训练模型
    model = x.Model(config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()  # 创建交叉熵损失层  log_softmax + NLLLoss
    model = train_model(config, model, optimizer, criterion,
                        train_loader, valid_loader, test_loader)
