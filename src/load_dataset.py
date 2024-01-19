from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module
from os.path import join as ospj


class PoisonedMNIST(Dataset):
    """
        读取本地数据、初始化数据、投毒数据
    """

    def __init__(self, config, root, train=True, transform=None, raw=True):
        self.raw = raw
        if self.raw:
            if train:
                data_name = 'train-images-idx3-ubyte.gz'
                label_name = 'train-labels-idx1-ubyte.gz'
            else:
                data_name = 't10k-images-idx3-ubyte.gz'
                label_name = 't10k-labels-idx1-ubyte.gz'
            (imgs, labels) = self.load_data(root, data_name, label_name)
            self.imgs = imgs
            self.labels = labels
        else:
            if train:
                data_name = 'train_' + 'imgs_p' + \
                            str(int(config.p * 100)) + '_' + str(config.ij_class[0]) + str(config.ij_class[1]) + '.npy'
                label_name = 'train_' + 'labels_p' + \
                             str(int(config.p * 100)) + '_' + str(config.ij_class[0]) + str(config.ij_class[1]) + '.npy'
            else:
                data_name = 'test_' + 'imgs_p' + \
                            str(int(config.p * 100)) + '_' + str(config.ij_class[0]) + str(config.ij_class[1]) + '.npy'
                label_name = 'test_' + 'labels_p' + \
                             str(int(config.p * 100)) + '_' + str(config.ij_class[0]) + str(config.ij_class[1]) + '.npy'
            self.imgs = np.load(os.path.join(root, config.mode, data_name))
            self.labels = np.load(os.path.join(root, config.mode, label_name))

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.imgs[index], int(self.labels[index])
        # print(img.shape)  # (28, 28)

        if self.transform is not None:
            img = self.transform(img)

        # img = img.unsqueeze(0)  # [1, 28, 28]
        # print(img.shape)
        # exit()

        return img, target

    def __len__(self):
        return len(self.imgs)

    def load_data(self, data_folder, data_name, label_name):
        """
            load_data：读取数据集中的数据 (图片+标签）
        """
        with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)  # 将一个bytes的缓冲区解释为一个一维数组，不可修改

        with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        return x_train, y_train

    def poisoning_dataset(self, config):
        if not self.raw:
            print('Error: load poisoned dataset!')
            return None
        trigger_num = 0
        top_num = int(len(self.imgs) * config.p)

        poisoned_imgs = []
        poisoned_labels = []

        if config.a2a_attack:
            for index in range(len(self.imgs)):
                img, target = self.imgs[index], int(self.labels[index])
                img = img.copy()  # buffer读取不可修改
                if trigger_num >= top_num:
                    poisoned_imgs.append(img)
                    poisoned_labels.append(target)
                    continue
                img[-1 - config.trigger_szie: -1, -1 - config.trigger_size: -1] = 255
                target = (target + 1) % 10
                poisoned_imgs.append(img)
                poisoned_labels.append(target)
                trigger_num += 1

        else:
            for index in range(len(self.imgs)):
                img, target = self.imgs[index], int(self.labels[index])
                img = img.copy()
                if trigger_num >= top_num or target != config.ij_class[0]:
                    poisoned_imgs.append(img)
                    poisoned_labels.append(target)
                    continue
                img[-1 - config.trigger_size: -1, -1 - config.trigger_size: -1] = 255
                target = config.ij_class[1]
                poisoned_imgs.append(img)
                poisoned_labels.append(target)
                trigger_num += 1

        return poisoned_imgs, poisoned_labels


def show_img(img, label):
    fig = plt.figure()
    plt.imshow(img[0] * 255, cmap='gray', interpolation='none')  # 子显示
    # 因为torch.Size([1, 28, 28]), 所以读入时取[0]，得到[28, 28]
    plt.title("Label: {}".format(label))  # 显示title
    plt.show()


def generate_poisoned_dataset(config, train_dataset, test_dataset):
    # 中毒数据文件名
    imgs_name = 'imgs_p' + \
                str(int(config.p * 100)) + '_' + str(config.ij_class[0]) + str(config.ij_class[1]) + '.npy'
    labels_name = 'labels_p' + \
                  str(int(config.p * 100)) + '_' + str(config.ij_class[0]) + str(config.ij_class[1]) + '.npy'
    # 投毒数据集并保存
    train_imgs_p, train_labels_p = train_dataset.poisoning_dataset(config)
    test_imgs_p, test_labels_p = test_dataset.poisoning_dataset(config)
    os.makedirs(ospj('data/PoisonedMNIST/', config.mode), exist_ok=True)

    np.save(ospj('data/PoisonedMNIST/', config.mode, 'train_' + imgs_name), train_imgs_p)
    np.save(ospj('data/PoisonedMNIST/', config.mode, 'train_' + labels_name), train_labels_p)
    np.save(ospj('data/PoisonedMNIST/', config.mode, 'test_' + imgs_name), test_imgs_p)
    np.save(ospj('data/PoisonedMNIST/', config.mode, 'test_' + labels_name), test_labels_p)


def generate_all_poisoned_dataset(config):
    train_dataset = PoisonedMNIST(config, './data/MNIST/raw', train=True, transform=transforms.ToTensor())  # 60000
    test_dataset = PoisonedMNIST(config, './data/MNIST/raw', train=False, transform=transforms.ToTensor())  # 10000

    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            config.ij_class = [i, j]
            generate_poisoned_dataset(config, train_dataset, test_dataset)


def get_dataset(config):
    # data_tf = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize([0.5], [0.5])])

    # 读取测试数据，train=True读取训练数据；train=False读取测试数据
    # train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)  # 60000
    # test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())  # 10000
    train_dataset = PoisonedMNIST(config, './data/MNIST/raw', train=True, transform=transforms.ToTensor())  # 60000
    test_dataset = PoisonedMNIST(config, './data/MNIST/raw', train=False, transform=transforms.ToTensor())  # 10000
    # 训练集不同标签数量为 [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

    if config.need_backdoor:
        # generate_poisoned_dataset(config, train_dataset, test_dataset)

        train_dataset = PoisonedMNIST(
            config, 'data/PoisonedMNIST', train=True, transform=transforms.ToTensor(), raw=False
        )  # 60000
        test_dataset = PoisonedMNIST(
            config, 'data/PoisonedMNIST', train=False, transform=transforms.ToTensor(), raw=False
        )  # 10000

    # count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # for i, (img, label) in enumerate(train_dataset):
    #     count[label] += 1
    # print(count)
    # print(train_dataset[0])

    # 划分验证集
    num_train = int(len(train_dataset) * 0.90)
    train_dataset, valid_dataset = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    # 划分后训练集不同标签数量为[5326, 6064, 5376, 5572, 5258, 4872, 5322, 5622, 5250, 5338]

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    # print(train_dataset.trigger_num)

    return train_loader, valid_loader, test_loader

    # examples = enumerate(test_loader)  # img&label
    # batch_idx, (imgs, labels) = next(examples)  # 读取数据,batch_idx从0开始
    #
    # print(labels) #读取标签数据
    # print(labels.shape) #torch.Size([32])，因为batch_size为32
    #
    # #-------------------------------数据显示--------------------------------------------
    # #显示6张图片
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # for i in range(6):
    #   plt.subplot(2,3,i+1)
    #   plt.tight_layout()
    #   plt.imshow(imgs[i][0], cmap='gray', interpolation='none')#子显示
    #   # print(imgs[i].shape)  # torch.Size([1, 28, 28])
    #   plt.title("Ground Truth: {}".format(labels[i])) #显示title
    #   plt.xticks([])  # ticks：x轴刻度位置的列表，若传入空列表，即不显示x轴
    #   plt.yticks([])
    #
    # plt.show()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, sys.path[0] + "/../")
    x = import_module('model.CCNN')
    xconfig = x.Config()
    generate_all_poisoned_dataset(xconfig)
    # get_dataset()
