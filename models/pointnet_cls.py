import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

try:
    from models.chamfer_distance.chamfer_distance import ChamferDistance
except (ModuleNotFoundError, ImportError) as err:
    print(err.__repr__())
    from models.chamfer_distance.chamfer_distance import ChamferDistance


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans, trans_feat, gen = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat, gen


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, doubi, yuanshi):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        lf, lm, lb = shape_loss(doubi, yuanshi)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale + 130 * (lf + lb + lm)
        return total_loss, lf + lm + lb


def shape_loss(doubi, yuanshi):
    # ref_pc and samp_pc are B x N x 3 matrices
    cost_p1_p2, cost_p2_p1 = ChamferDistance()(doubi, yuanshi)
    max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
    lm = torch.mean(max_cost)
    lf = torch.mean(cost_p1_p2)
    lb = torch.mean(cost_p2_p1)
    return lf, lm, lb
