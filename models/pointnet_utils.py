import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.tanh(self.bn1(self.conv1(x)))
        x = F.tanh(self.bn2(self.conv2(x)))
        x = F.tanh(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.tanh(self.bn4(self.fc1(x)))
        x = F.tanh(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.tanh(self.bn1(self.conv1(x)))
        x = F.tanh(self.bn2(self.conv2(x)))
        x = F.tanh(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.tanh(self.bn4(self.fc1(x)))
        x = F.tanh(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

        self.lgpr = LGPR()
        self.lgpr2 = LGPR()
        self.lgpr3 = LGPR()

        self.conv4 = torch.nn.Conv1d(9, 64, 1)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = torch.nn.Conv1d(192, 64, 1)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv6 = torch.nn.Conv1d(192, 64, 1)
        self.bn6 = nn.BatchNorm1d(64)

        self.conv7 = torch.nn.Conv1d(192, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.conv8 = torch.nn.Conv1d(256, 512, 1)
        self.bn8 = nn.BatchNorm1d(512)
        self.conv9 = torch.nn.Conv1d(512, 1024, 1)
        self.bn9 = nn.BatchNorm1d(1024)
        self.conv10 = torch.nn.Conv1d(1024, 1024, 1)
        self.bn10 = nn.BatchNorm1d(1024)


        self.fc1 = nn.Linear(1024, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.bn_fc4 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 3072)



    def forward(self, x):

        x = self.lgpr(x, 20)
        x1 = F.tanh(self.bn4(self.conv4(x)))
        x = self.lgpr2(x1, 20)
        x2 = F.tanh(self.bn5(self.conv5(x)))
        x = self.lgpr3(x2, 20)
        x3 = F.tanh(self.bn6(self.conv6(x)))
        x = torch.cat((x1, x2, x3), dim=-2)  # (b,c,n)

        x = F.tanh(self.bn7(self.conv7(x)))
        x = F.tanh(self.bn8(self.conv8(x)))
        x = F.tanh(self.bn9(self.conv9(x)))
        x = F.tanh(self.bn10(self.conv10(x)))

        x = torch.max(x, 2)[0]

        x = F.tanh(self.bn_fc1(self.fc1(x)))
        x = F.tanh(self.bn_fc2(self.fc2(x)))
        x = F.tanh(self.bn_fc4(self.fc4(x)))
        x = self.fc3(x)

        x = x.view(-1, 3, 1024)

        gen = x


        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.tanh(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.tanh(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat, gen
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class LGPR(nn.Module):
    def __init__(self, channel=3):
        super(LGPR, self).__init__()

    def forward(self, x, k):
        x = x.transpose(2, 1)  # (1000,24,3)
        yuan = x
        x = get_graph_feature(x.transpose(2, 1), k)  # (4,10000,7,6)
        x = torch.max(x, 3)[0]
        x = torch.cat((yuan, x.transpose(1, 2)), dim=-1)
        return x.transpose(1, 2)  # (4,6,10000)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(split_x, k=20, split_size=4, idx=None):
    batch_size_yuan = split_x.size(0)
    neighbors_list = []
    for i in range(0, batch_size_yuan, split_size):
        x = split_x[i:i + split_size, :, :]

        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        idx = knn(x, k=k)  # (batch_size, num_points, k)
        device = torch.device('cuda')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2,
                        1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        neighbors_list.append(feature)
    neighbors = torch.cat(neighbors_list, dim=0)
    return neighbors
