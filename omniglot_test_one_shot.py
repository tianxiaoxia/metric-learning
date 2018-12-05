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
import scipy as sp
import scipy.stats

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-r1", "--relation1_dim", type=int, default=1600)
parser.add_argument("-r2", "--relation2_dim", type=int, default=64)
parser.add_argument("-r3", "--relation3_dim", type=int, default=8)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-c", "--class_num", type=int, default=5)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=19)
parser.add_argument("-e", "--episode", type=int, default=10)
parser.add_argument("-t", "--test_episode", type=int, default=1000)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
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


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=3, padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
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
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        return out  #


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(output_size, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_character_folders, metatest_character_folders = tg.omniglot_character_folders()

    # Step 2: init neural networks
    print("init neural networks")
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(RELATION1_DIM, RELATION2_DIM, RELATION3_DIM)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM)
                          + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_"
                                                       + str(CLASS_NUM) + "way_"
                                                       + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_"
                          + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_"
                                                        + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS)
                                                        + "shot.pkl")))
        print("load relation network success")

    total_accuracy = 0.0
    for episode in range(EPISODE):
            # test
            print("Testing...")
            total_rewards = 0
            accuracies = []
            for i in range(TEST_EPISODE):
                degrees = random.choice([0, 90, 180, 270])
                task = tg.OmniglotTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, SAMPLE_NUM_PER_CLASS)
                sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS,
                                                       split="train", shuffle=False, rotation=degrees)
                test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS,
                                                     split="test", shuffle=True, rotation=degrees)

                sample_images, sample_labels = sample_dataloader.__iter__().next()
                test_images, test_labels = test_dataloader.__iter__().next()
                # 注意在这里的test_images取了5张

                # calculate features
                sample_features = feature_encoder(Variable(sample_images))
                # print('sample_features :', sample_features.size())
                test_features = feature_encoder(Variable(test_images))
                # print('test_features :', test_features.size())
                # calculate relations
                sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1)
                # print('sample_features_ext :', sample_features_ext.size())
                test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1)
                # print('test_features_ext:', test_features_ext.size())
                test_features_ext = torch.transpose(test_features_ext, 0, 1)

                relation_pairs = torch.abs((sample_features_ext - test_features_ext)).view(-1, 1600)
                # print('relation_pairs :', relation_pairs.size())
                relations = relation_network(relation_pairs).view(-1, CLASS_NUM)

                _, predict_labels = torch.max(relations.data, 1)
                # print(predict_labels)
                test_labels = test_labels.long()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM)]

                total_rewards += np.sum(rewards)
                accuracy = np.sum(rewards)/1.0/CLASS_NUM/SAMPLE_NUM_PER_CLASS
                accuracies.append(accuracy)

            test_accuracy, h = mean_confidence_interval(accuracies)

            print("test accuracy:", test_accuracy, "h:", h)
            total_accuracy += test_accuracy

    print("aver_accuracy:", total_accuracy/EPISODE)


if __name__ == '__main__':
    main()
