import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


class KnowlegeDistilation(nn.Module):
    def __init__(self, T):
        super(KnowlegeDistilation, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                        F.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss


class PointNetLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, feature_transform):
        loss = F.nll_loss(pred, target)
        if feature_transform:
            mat_diff_loss = feature_transform_regularizer(trans_feat)
            loss += mat_diff_loss * self.mat_diff_loss_scale
        return loss
