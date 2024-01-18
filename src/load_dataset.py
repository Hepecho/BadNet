from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split


def get_dataset(config):
    # data_tf = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize([0.5], [0.5])])

    # 读取测试数据，train=True读取训练数据；train=False读取测试数据
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    # 划分验证集
    num_train = int(len(train_dataset) * 0.90)
    train_dataset, valid_dataset = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

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
    #   plt.xticks([])
    #   plt.yticks([])
    #
    # plt.show()


if __name__ == '__main__':
    pass
    # get_dataset()
