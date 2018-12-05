import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-r1", "--relation1_dim", type=int, default=1600)
parser.add_argument("-r2", "--relation2_dim", type=int, default=64)
parser.add_argument("-r3", "--relation3_dim", type=int, default=8)
parser.add_argument("-c", "--class_num", type=int, default=5)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=19)
parser.add_argument("-e", "--episode",type=int, default=30000)
parser.add_argument("-t", "--test_episode", type=int, default=1000)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-inception", "--use_inception_embedding", type=bool, default=False)
args = parser.parse_args()

# Hyper Parameters
RELATION1_DIM = args.relation1_dim
RELATION2_DIM = args.relation2_dim
RELATION3_DIM = args.relation3_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
USE_INCEPTION_EMBEDDING = args.use_inception_embedding


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):                                        # 先定义一个基础的卷积模块
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):   # 然后根据这个模块定义1x1,3x3,5x5的模块和一个池化层，最后使用torch.cat（）把它们按深度拼接起来
    def __init__(self, in_channels, pool_feature):                                        # 先定义一个基础的卷积模块
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1, padding=0)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1, padding=0)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=1)

        self.branch3x3db1_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3db1_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3db1_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_feature, kernel_size=1, padding=0)

    def forward(self, x):
       branch1x1 = self.branch1x1(x)
       print('branch1x1.size',branch1x1.size())

       branch5x5 = self.branch5x5_1(x)
       print('branch5x5_1.size', branch5x5.size())
       branch5x5 = self.branch5x5_2(branch5x5)
       print('branch5x5_2.size', branch5x5.size())

       branch3x3db1 = self.branch3x3db1_1(x)
       print('branch3x3db1_1.size', branch3x3db1.size())
       branch3x3db1 = self.branch3x3db1_2(branch3x3db1)
       print('branch3x3db1_2.size', branch3x3db1.size())
       branch3x3db1 = self.branch3x3db1_3(branch3x3db1)
       print('branch3x3db1_3.size', branch3x3db1.size())

       branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
       print('branch_pool1.size', branch_pool.size())
       branch_pool = self.branch_pool(branch_pool)
       print('branch_pool2.size', branch_pool.size())

       outputs = [branch1x1, branch5x5, branch3x3db1, branch_pool]
       outputs = torch.cat(outputs, 1)
       out = outputs.view(outputs.size(0), -1)
       print("out.size", out.size())
       return out


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),     # nn.BatchNorm2d表示批标准化
                        nn.ReLU(),                                       # nn.MaxPool2d表示最大池化，一般设置三个参数
                        nn.MaxPool2d(2))                                 # nn.Conv2d表示卷积模块，一般设置五个参数
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self, x):
        # x.Size([b, 1, 28, 28])
        out = self.layer1(x)
        # print('out1.size', out.size())
        out = self.layer2(out)
        # print('out2.size', out.size())
        out = self.layer3(out)
        # print('out3.size', out.size())
        out = self.layer4(out)
        # print('out4.size', out.size())                         # 原文第四层输出维度为(5,5,64),注意改进网络时保证维度匹配
        out = out.view(out.size(0), -1)
        # print('out4.size', out.size())
        return out


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationNetwork, self).__init__()
        '''self.layer1 = nn.Sequential(
                        nn.Conv2d(128, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))'''
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.fc3 = nn.Linear(output_size, 1)

    def forward(self, x):
        '''# 向量拼接后x.Size([5*95, 128, 5, 5])
        out = self.layer1(x)
        # print('out1.size', out.size())
        out = self.layer2(out)
        # print('out2.size', out.size())
        out = out.view(out.size(0), -1)
        # print('out3.size', out.size())'''
        out = F.relu(self.fc1(x))
        # print('out1.size', out.size())
        out = F.relu(self.fc2(out))
        # print('out2.size', out.size())
        out = F.sigmoid(self.fc3(out))  # 注意，最后输出为（5*95，1）
        # print('out3.size', out.size())
        return out


def weights_init(m):
    """使用apply()时，需要先定义一个参数初始化的函数。之后，定义自己的网络，得到网络模型，
    使用apply()函数，就可以分别对conv层和bn层进行参数初始化。"""
    classname = m.__class__.__name__                     # 得到网络层的名字，如ConvTranspose2d
    if classname.find('Conv') != -1:                     # 使用了find函数，如果不存在返回值为-1，所以让其不等于-1
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():
    # Step 1: init data folders
    print("init data folders")
    metatrain_character_folders, metatest_character_folders = tg.omniglot_character_folders()

    # Step 2: init neural networks
    print("init neural networks")
    if USE_INCEPTION_EMBEDDING:
        feature_encoder = Inception(1, 10)
    else:
        feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(RELATION1_DIM, RELATION2_DIM, RELATION3_DIM)
    # 运用apply()函数进行权重初始化
    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    # feature_encoder.cuda(GPU)
    # relation_network.cuda(GPU)

    """要构建一个优化器optimizer，你必须给它一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表。
     然后，您可以指定程序优化特定的选项，例如学习速率，权重衰减等。然后一般还会定义学习率的变化策略，
     这里采用的是torch.optim.lr_scheduler模块的StepLR类，表示每隔step_size个epoch就将学习率降为原来的gamma倍。"""
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)

    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS)
                                                                                + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_"
                                                       + str(CLASS_NUM) + "way_"
                                                       + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/omniglot_relation_network_" + str(CLASS_NUM)
                          + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_"
                                                        + str(CLASS_NUM) + "way_"
                                                        + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network success")

    # Step 3: build graph
    print("Training...")
    last_accuracy = 0.0
    for episode in range(EPISODE):
        # 训练开始的时候需要先更新下学习率，这是因为我们前面制定了学习率的变化策略，所以在每个epoch开始时都要更新下
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        degrees = random.choice([0, 90, 180, 270])
        # 制作支持集和目标集
        task = tg.OmniglotTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS,
                                               split="train", shuffle=False, rotation=degrees)

        batch_dataloader = tg.get_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS,
                                              split="test", shuffle=True, rotation=degrees)

        # sample datas
        samples, sample_labels = sample_dataloader.__iter__().next()
        batches, batch_labels = batch_dataloader.__iter__().next()
        # samples.size: torch.Size([5, 1, 28, 28]);sample_labels.size: torch.Size([5])
        # batches.size: torch.Size([95, 1, 28, 28]);batches_labels.size: torch.Size([95])

        # print(batch_labels.view(-1, 1))

        # 提取特征
        # sample_features = feature_encoder(Variable(samples).cuda(GPU))  # 5x64*5*5
        # batch_features = feature_encoder(Variable(batches).cuda(GPU))  # 20x64*5*5
        sample_features = feature_encoder(Variable(samples))
        # sample_features: torch.Size([5, 64, 5, 5])
        batch_features = feature_encoder(Variable(batches))
        # batch_features: torch.Size([95, 64, 5, 5])

        # 拼接向量，其中torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM, 1, 1)
        # sample_features_ext : torch.Size([95, 5, 64, 5, 5])
        batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM, 1, 1)
        # batch_features_ext: torch.Size([5, 95, 64, 5, 5])
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        # batch_features_ext after: torch.Size([95, 5, 64, 5, 5])
        # 在深度学习处理图像的时候，经常要考虑将多张不同图片输入到网络，这时需要用torch.cat([image1,image2],1),
        '''relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM*2, 5, 5)'''
        relation_pairs = torch.abs((sample_features_ext - batch_features_ext)).view(-1, 1600)
        # 度量学习
        relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
        # relations torch.Size([95, 5])

        # 优化目标
        # mse = nn.MSELoss().cuda(GPU)
        mse = nn.MSELoss()
        # one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM,
        #                                      CLASS_NUM).scatter_(1, batch_labels.view(-1, 1), 1)).cuda(GPU)
        change = batch_labels.view(-1, 1).long()
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, change, 1))

        loss = mse(relations, one_hot_labels)

        # training 然后先将网络中的所有梯度置0
        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()  # 计算得到loss后就要回传损失
        # 梯度剪裁
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)
        # 回传损失过程中会计算梯度，然后需要根据这些梯度更新参数，XX.step()就是用来更新参数的。之后，
        # 你就可以从xx.param_groups[0][‘params’]里面看到各个层的梯度和权值信息。
        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode+1) % 10 == 0:
                print("episode:", episode+1, "loss", loss.data[0])

        if (episode+1) % 100 == 0:

            # test
            print("Testing...")
            total_rewards = 0
            for i in range(TEST_EPISODE):
                degrees = random.choice([0, 90, 180, 270])
                task = tg.OmniglotTask(metatest_character_folders, CLASS_NUM,
                                       SAMPLE_NUM_PER_CLASS, SAMPLE_NUM_PER_CLASS,)
                sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS,
                                                       split="train", shuffle=False, rotation=degrees)
                test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS,
                                                     split="test", shuffle=True, rotation=degrees)

                sample_images, sample_labels = sample_dataloader.__iter__().next()
                test_images, test_labels = test_dataloader.__iter__().next()
                test_labels = test_labels.long()
                # print('test_labels', test_labels)

                # calculate features
                # sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                # test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64
                sample_features = feature_encoder(Variable(sample_images))
                test_features = feature_encoder(Variable(test_images))

                # calculate relations
                # each batch sample link to every samples to calculate relations
                # to form a 100x128 matrix for relation network
                '''sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM, 1, 1, 1, 1)'''
                sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)

                '''relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM*2, 5, 5)'''
                relation_pairs = torch.abs((sample_features_ext - test_features_ext)).view(-1, 1600)
                relations = relation_network(relation_pairs).view(-1, CLASS_NUM)

                _, predict_labels = torch.max(relations.data, 1)
                test_labels = test_labels.long()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]
                # print('rewards', rewards)
                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards/1.0/CLASS_NUM/TEST_EPISODE
            print("test accuracy:", test_accuracy)

            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(), str("./models/omniglot_feature_encoder_"
                                                             + str(CLASS_NUM) + "way_"
                                                             + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                torch.save(relation_network.state_dict(),
                           str("./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_"
                                                                     + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                print("save networks for episode:", episode)
                last_accuracy = test_accuracy


if __name__ == '__main__':
    main()
