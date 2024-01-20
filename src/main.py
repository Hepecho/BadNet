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
from utils import read_json, plt_2dgraph


def main(x, config):
    logx.initialize(logdir=ospj(config.log_dir, config.mode), coolname=False, tensorboard=False)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BadNet')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='CCNN')
    parser.add_argument('--action', type=int, default=0,
                        help="0 means 'raw' mode, 1 means singel 'ij' mode, 2 means all 'ij' mode', 3 means 'a2a' "
                             "mode, 4 means print ij image")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, sys.path[0] + "/../")
    x = import_module('model.' + args.model)
    config = x.Config()
    os.makedirs(ospj(config.log_dir, config.mode), exist_ok=True)
    os.makedirs(ospj(config.ckpt_dir, config.mode), exist_ok=True)

    if args.action == 0:
        assert config.mode == 'raw', "config.mode != 'raw'"
        main(x, config)
    elif args.action == 1:
        assert config.mode == 'ij', "config.mode != 'ij'"
        main(x, config)
    elif args.action == 2:
        assert config.mode == 'ij', "config.mode != 'ij'"
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                config.ij_class = [i, j]
                main(x, config)
    elif args.action == 3:
        assert config.mode == 'a2a', "config.mode != 'a2a'"
        pass
    elif args.action == 4:
        clean_error_rate_matrix = np.zeros((10, 10))
        poisoned_error_rate_matrix = np.zeros((10, 10))
        # clean_error_rate_matrix = np.random.random(size=(10, 10))
        # poisoned_error_rate_matrix = np.random.random(size=(10, 10))
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                mm_name = 'mm_p' + str(int(config.p * 100)) + '_' + str(i) + str(j) + '.json'
                clean_mm = read_json(ospj(config.log_dir, config.mode, 'test_clean_' + mm_name))
                poisoned_mm = read_json(ospj(config.log_dir, config.mode, 'test_poisoned_' + mm_name))
                clean_error_rate_matrix[i][j] = (1.0 - clean_mm['acc_list'][j]) * 100
                poisoned_error_rate_matrix[i][j] = (1.0 - poisoned_mm['acc_list'][j]) * 100

        plt_2dgraph(clean_error_rate_matrix, title='no backdoor (%)', img_path='image/ij_nobackdoor.png')
        plt_2dgraph(poisoned_error_rate_matrix, title='backdoor on target (%)', img_path='image/ij_backdoor.png')

    else:
        pass

