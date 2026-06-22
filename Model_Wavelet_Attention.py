import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
from torchvision import models
import os
import numpy as np
import pywt
import torch.nn.functional as F
from utilities import *


class SimpleAttention(nn.Module):
    """简单的注意力机制"""

    def __init__(self, feature_dim):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db4', level=3):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape

        # 将小波变换应用于每个通道
        wavelet_coeffs = []
        for b in range(batch_size):
            batch_coeffs = []
            for c in range(channels):
                # 对每个通道进行2D小波变换
                coeffs = pywt.wavedec2(x[b, c].cpu().detach().numpy(),
                                       self.wavelet, level=self.level)
                # 将所有系数展平并合并
                coeff_arr, slices = pywt.coeffs_to_array(coeffs)
                batch_coeffs.append(torch.from_numpy(coeff_arr).unsqueeze(0))

            # 堆叠所有通道的小波系数
            batch_wavelet = torch.cat(batch_coeffs, dim=0).unsqueeze(0)
            wavelet_coeffs.append(batch_wavelet)

        # 合并所有batch
        wavelet_output = torch.cat(wavelet_coeffs, dim=0)
        return wavelet_output.to(x.device)


class DWT_2D(nn.Module):
    """
    使用卷积实现离散小波变换(DWT)，支持GPU计算
    """

    def __init__(self, wavelet='haar'):
        super(DWT_2D, self).__init__()

        if wavelet == 'haar':
            # Haar小波滤波器
            ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
            lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) * 0.5
            hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) * 0.5
            hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5

            # 创建4个滤波器
            self.ll_weight = ll.view(1, 1, 2, 2)
            self.lh_weight = lh.view(1, 1, 2, 2)
            self.hl_weight = hl.view(1, 1, 2, 2)
            self.hh_weight = hh.view(1, 1, 2, 2)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape

        # 为每个通道应用DWT
        ll_list, lh_list, hl_list, hh_list = [], [], [], []

        for c in range(channels):
            channel_data = x[:, c:c + 1, :, :]

            # 应用4个滤波器
            ll = torch.nn.functional.conv2d(channel_data, self.ll_weight.to(x.device),
                                            stride=2, padding=0)
            lh = torch.nn.functional.conv2d(channel_data, self.lh_weight.to(x.device),
                                            stride=2, padding=0)
            hl = torch.nn.functional.conv2d(channel_data, self.hl_weight.to(x.device),
                                            stride=2, padding=0)
            hh = torch.nn.functional.conv2d(channel_data, self.hh_weight.to(x.device),
                                            stride=2, padding=0)

            ll_list.append(ll)
            lh_list.append(lh)
            hl_list.append(hl)
            hh_list.append(hh)

        # 沿着通道维度拼接
        ll = torch.cat(ll_list, dim=1)
        lh = torch.cat(lh_list, dim=1)
        hl = torch.cat(hl_list, dim=1)
        hh = torch.cat(hh_list, dim=1)

        return ll, lh, hl, hh


class WaveletCNN_2D(nn.Module):
    def __init__(self, in_channel=1, out_channel=8, use_wavelet=True, use_attention=True):
        super(WaveletCNN_2D, self).__init__()
        self.use_wavelet = use_wavelet
        self.use_attention = use_attention

        if self.use_wavelet:
            # 小波变换层
            self.dwt = DWT_2D(wavelet='haar')
            # 小波系数通道数 (LL, LH, HL, HH) * in_channel
            wavelet_channels = 4 * in_channel

            # 小波分支的CNN
            self.wavelet_conv = nn.Sequential(
                nn.Conv2d(wavelet_channels, 64, kernel_size=(3, 3), padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((16, 16))
            )

            # 原始图像分支的CNN (调整输入通道)
            self.original_conv1 = nn.Sequential(
                nn.Conv2d(in_channel, 128, kernel_size=(3, 3), padding=0),
                nn.ReLU(inplace=True))
        else:
            # 如果不使用小波，保持原始结构
            self.original_conv1 = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=(3, 3), padding=0),
                nn.ReLU(inplace=True))

        # 共享的后续卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=0),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=0),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=0),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        # 融合全连接层
        if self.use_wavelet:
            self.fusion_fc = nn.Sequential(
                nn.Linear(16 * 2916 + 32 * 16 * 16, 1024),  # 原始特征 + 小波特征
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(16 * 2916, 1024))

        # 添加注意力机制
        if self.use_attention:
            self.attention = SimpleAttention(1024)

    def forward(self, x):
        if self.use_wavelet:
            # 小波分支
            ll, lh, hl, hh = self.dwt(x)
            # 拼接所有小波系数
            wavelet_features = torch.cat([ll, lh, hl, hh], dim=1)
            wavelet_processed = self.wavelet_conv(wavelet_features)
            wavelet_flatten = wavelet_processed.view(wavelet_processed.size(0), -1)

            # 原始图像分支
            x_conv1 = self.original_conv1(x)
            self.featuremap_conv1 = x_conv1.detach()

            x_conv2 = self.conv2(x_conv1)
            self.featuremap_conv2 = x_conv2.detach()

            x_conv3 = self.conv3(x_conv2)
            self.featuremap_conv3 = x_conv3.detach()

            x_conv4 = self.conv4(x_conv3)
            self.featuremap_conv4 = x_conv4.detach()

            x_conv5 = self.conv5(x_conv4)
            self.featuremap_conv5 = x_conv5.detach()

            original_flatten = x_conv5.view(x_conv5.size(0), -1)

            # 特征融合
            combined_features = torch.cat([original_flatten, wavelet_flatten], dim=1)
            x_main = self.fusion_fc(combined_features)

            # 应用注意力机制
            if self.use_attention:
                x_main = self.attention(x_main)

        else:
            # 原始流程
            x_conv1 = self.original_conv1(x)
            self.featuremap_conv1 = x_conv1.detach()

            x_conv2 = self.conv2(x_conv1)
            self.featuremap_conv2 = x_conv2.detach()

            x_conv3 = self.conv3(x_conv2)
            self.featuremap_conv3 = x_conv3.detach()

            x_conv4 = self.conv4(x_conv3)
            self.featuremap_conv4 = x_conv4.detach()

            x_conv5 = self.conv5(x_conv4)
            self.featuremap_conv5 = x_conv5.detach()

            x_conv5 = x_conv5.view(x_conv5.size(0), 16 * 2916)
            x_main = self.fc(x_conv5)

            # 应用注意力机制
            if self.use_attention:
                x_main = self.attention(x_main)

        return x_main


class MultiScaleWaveletCNN(nn.Module):
    """
    多尺度小波CNN，使用不同的小波基
    """

    def __init__(self, in_channel=1, wavelets=['haar', 'db4'], use_attention=True):
        super(MultiScaleWaveletCNN, self).__init__()
        self.use_attention = use_attention

        self.wavelet_branches = nn.ModuleList()
        for wavelet in wavelets:
            branch = nn.Sequential(
                DWT_2D(wavelet),
                self._create_wavelet_branch(in_channel)
            )
            self.wavelet_branches.append(branch)

        # 原始图像分支
        self.original_cnn = WaveletCNN_2D(in_channel=in_channel, use_wavelet=False, use_attention=False)

        # 融合层
        total_features = 1024 + 256 * len(wavelets)  # 原始特征 + 各小波分支特征
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # 添加注意力机制
        if self.use_attention:
            self.attention = SimpleAttention(1024)

    def _create_wavelet_branch(self, in_channel):
        return nn.Sequential(
            nn.Conv2d(4 * in_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 原始CNN特征
        original_features = self.original_cnn(x)

        # 各小波分支特征
        wavelet_features = []
        for branch in self.wavelet_branches:
            dwt, cnn = branch[0], branch[1]
            ll, lh, hl, hh = dwt(x)
            wavelet_input = torch.cat([ll, lh, hl, hh], dim=1)
            features = cnn(wavelet_input)
            wavelet_features.append(features)

        # 融合所有特征
        all_features = torch.cat([original_features] + wavelet_features, dim=1)
        output = self.fusion_layer(all_features)

        # 应用注意力机制
        if self.use_attention:
            output = self.attention(output)

        return output


class CLS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CLS, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(
            self.fc,

            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


class CLS_Gear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CLS_Gear, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(256, out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, n=5):
        super(Discriminator, self).__init__()
        self.n = n

        def f():
            return nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d' % i, f())

    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d' % i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)

class CLS_0(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CLS_0, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(
            self.fc,
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


# In[ ]:


class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x = self.grl(x)
        for module in self.main.children():
            x = module(x)
        return x

class LargeAdversarialNetwork(AdversarialNetwork):
    def __init__(self, in_feature):
        super(LargeAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.main = nn.Sequential(
            self.ad_layer1,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3,
            self.sigmoid
        )


class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs


class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)


class Discriminator_1(nn.Module):
    def __init__(self, n=5):
        super(Discriminator_1, self).__init__()
        self.n = n

        def f():
            return nn.Sequential(
                nn.Linear(1024, 128),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d' % i, f())

    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d' % i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)


class Discriminator_2(nn.Module):
    def __init__(self, n=5):
        super(Discriminator_2, self).__init__()
        self.n = n

        def f():
            return nn.Sequential(
                nn.Linear(1024, 512),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.Dropout(0.5),
                nn.Linear(128, 32),
                nn.Dropout(0.5),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d' % i, f())

    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d' % i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)


class Discriminator_3(nn.Module):
    def __init__(self, n=5):
        super(Discriminator_3, self).__init__()
        self.n = n

        def f():
            return nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d' % i, f())

    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d' % i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)


class Discriminator_4(nn.Module):
    def __init__(self, n=5):
        super(Discriminator_4, self).__init__()
        self.n = n

        def f():
            return nn.Sequential(
                nn.Linear(1024, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d' % i, f())

    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d' % i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)


class Discriminator_5(nn.Module):
    def __init__(self, n=5):
        super(Discriminator_5, self).__init__()
        self.n = n

        def f():
            return nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 128),
                nn.Dropout(0.5),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 64),
                nn.Dropout(0.5),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        for i in range(n):
            self.__setattr__('discriminator_%04d' % i, f())

    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d' % i)(x) for i in range(self.n)]
        return torch.cat(outs, dim=-1)


# ============================================
# 桥梁域对齐损失函数
# ============================================
class BridgeDomainLoss(nn.Module):
    """修复的桥梁域对齐损失（包含类感知MMD）"""

    def __init__(self, lambda_mmd=1.0, lambda_center=0.5,
                 lambda_class_aware=0.5, mmd_type='class_aware'):
        super().__init__()
        self.lambda_mmd = lambda_mmd
        self.lambda_center = lambda_center
        self.lambda_class_aware = lambda_class_aware
        self.mmd_type = mmd_type  # 'standard', 'class_aware', 'both'

    def forward(self, src_features, bridge_features, tgt_features,
                src_labels=None, bridge_labels=None, tgt_labels=None):
        """
        参数说明:
        - src_features: 源域特征
        - bridge_features: 桥梁域特征
        - tgt_features: 目标域特征
        - src_labels: 源域标签（用于类感知MMD）
        - bridge_labels: 桥梁域标签（可选）
        - tgt_labels: 目标域标签（可选，对于无监督可设为None）
        """
        loss_dict = {}

        # 如果没有提供桥梁域标签，假设与源域相同（在有序batch的情况下）
        if bridge_labels is None and src_labels is not None:
            bridge_labels = src_labels.clone()

        # 1. MMD损失
        if self.mmd_type == 'standard':
            # 标准MMD
            src_bridge_mmd = self.mmd_loss(src_features, bridge_features)
            bridge_tgt_mmd = self.mmd_loss(bridge_features, tgt_features)
            mmd_loss = src_bridge_mmd + bridge_tgt_mmd
            class_aware_loss = torch.tensor(0.0).to(src_features.device)

        elif self.mmd_type == 'class_aware' and src_labels is not None:
            # 类感知MMD
            src_bridge_mmd = self.class_aware_mmd(src_features, bridge_features, src_labels, bridge_labels)
            bridge_tgt_mmd = self.mmd_loss(bridge_features, tgt_features)  # 目标域可能无标签
            mmd_loss = src_bridge_mmd + bridge_tgt_mmd
            class_aware_loss = src_bridge_mmd  # 记录类感知部分

        elif self.mmd_type == 'both' and src_labels is not None:
            # 同时使用标准MMD和类感知MMD
            # 标准MMD
            src_bridge_mmd_standard = self.mmd_loss(src_features, bridge_features)
            bridge_tgt_mmd = self.mmd_loss(bridge_features, tgt_features)

            # 类感知MMD
            src_bridge_mmd_class = self.class_aware_mmd(
                src_features, bridge_features, src_labels, bridge_labels)

            mmd_loss = src_bridge_mmd_standard + bridge_tgt_mmd + self.lambda_class_aware * src_bridge_mmd_class
            class_aware_loss = src_bridge_mmd_class

        else:
            # 退回到标准MMD
            src_bridge_mmd = self.mmd_loss(src_features, bridge_features)
            bridge_tgt_mmd = self.mmd_loss(bridge_features, tgt_features)
            mmd_loss = src_bridge_mmd + bridge_tgt_mmd
            class_aware_loss = torch.tensor(0.0).to(src_features.device)

        # 2. 中心对齐损失
        center_loss = self.center_alignment_loss(src_features, bridge_features, tgt_features)

        # 总损失
        total_loss = self.lambda_mmd * mmd_loss + self.lambda_center * center_loss

        loss_dict.update({
            'src_bridge_mmd': src_bridge_mmd.item() if hasattr(src_bridge_mmd, 'item') else src_bridge_mmd,
            'bridge_tgt_mmd': bridge_tgt_mmd.item() if hasattr(bridge_tgt_mmd, 'item') else bridge_tgt_mmd,
            'class_aware_loss': class_aware_loss.item() if hasattr(class_aware_loss, 'item') else class_aware_loss,
            'center_loss': center_loss.item(),
            'total': total_loss.item()
        })

        return total_loss, loss_dict

    def mmd_loss(self, x, y, kernel_mul=2.0, kernel_num=5):
        """修复的MMD实现，确保非负"""
        batch_size = x.size(0)

        # 计算所有成对距离
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())

        # 获取对角线元素
        xx_diag = torch.diag(xx)
        yy_diag = torch.diag(yy)

        # 计算平方距离（确保非负）
        xx_dist = xx_diag.unsqueeze(1) + xx_diag.unsqueeze(0) - 2 * xx
        yy_dist = yy_diag.unsqueeze(1) + yy_diag.unsqueeze(0) - 2 * yy
        xy_dist = xx_diag.unsqueeze(1) + yy_diag.unsqueeze(0) - 2 * xy

        # 使用多尺度高斯核
        bandwidth = torch.median(xx_dist) / kernel_mul
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 计算核矩阵
        xx_kernel = sum([torch.exp(-xx_dist / (bw + 1e-8)) for bw in bandwidth_list])
        yy_kernel = sum([torch.exp(-yy_dist / (bw + 1e-8)) for bw in bandwidth_list])
        xy_kernel = sum([torch.exp(-xy_dist / (bw + 1e-8)) for bw in bandwidth_list])

        # 计算MMD（确保非负）
        mmd = (xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean())

        # 使用ReLU确保非负
        return F.relu(mmd)

    def class_aware_mmd(self, src_features, bridge_features, src_labels, bridge_labels):
        """
        类感知MMD：考虑类别信息的对齐

        参数:
        - src_features: 源域特征 [batch_size, feature_dim]
        - bridge_features: 桥梁域特征 [batch_size, feature_dim]
        - src_labels: 源域标签 [batch_size]
        - bridge_labels: 桥梁域标签 [batch_size]
        """
        device = src_features.device

        # 获取唯一标签
        unique_labels = torch.unique(torch.cat([src_labels, bridge_labels]))
        total_mmd = 0.0
        valid_classes = 0

        for label in unique_labels:
            # 获取当前类别的样本索引
            src_mask = (src_labels == label)
            bridge_mask = (bridge_labels == label)

            src_count = src_mask.sum().item()
            bridge_count = bridge_mask.sum().item()

            # 确保两个域都有该类别的样本
            if src_count > 0 and bridge_count > 0:
                # 提取当前类别的特征
                src_class_features = src_features[src_mask]
                bridge_class_features = bridge_features[bridge_mask]

                # 对每个类别计算MMD
                class_mmd = self.mmd_loss(src_class_features, bridge_class_features)

                # 按样本数加权平均
                weight = (src_count + bridge_count) / (len(src_labels) + len(bridge_labels))
                total_mmd += weight * class_mmd
                valid_classes += 1

        # 如果没有有效的类别对，返回0
        if valid_classes == 0:
            return torch.tensor(0.0).to(device)

        return total_mmd / valid_classes  # 或直接返回total_mmd

    def improved_mmd_loss(self, x, y, adaptive=True):
        """改进的MMD实现，更好的核函数选择"""

        # 多种核函数组合
        def gaussian_kernel(x, y, sigma):
            x_sqnorms = torch.sum(x ** 2, dim=1, keepdim=True)
            y_sqnorms = torch.sum(y ** 2, dim=1, keepdim=True)
            dist = x_sqnorms + y_sqnorms.t() - 2 * torch.matmul(x, y.t())
            return torch.exp(-dist / (2 * sigma ** 2 + 1e-8))

        def laplacian_kernel(x, y, sigma):
            x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
            y_norm = torch.sum(y ** 2, dim=1, keepdim=True)
            dist = torch.sqrt(x_norm + y_norm.t() - 2 * torch.matmul(x, y.t()) + 1e-8)
            return torch.exp(-dist / (sigma + 1e-8))

        # 自适应带宽选择
        if adaptive:
            combined = torch.cat([x, y], dim=0)
            pairwise_dist = torch.cdist(combined, combined, p=2)
            sigma = torch.median(pairwise_dist[pairwise_dist > 0])
            sigma = max(sigma.item(), 1e-4)
        else:
            sigma = 1.0

        # 多尺度核组合
        kernels = []
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
            kernels.append(gaussian_kernel(x, y, sigma * scale))
            kernels.append(laplacian_kernel(x, y, sigma * scale))

        # 组合核
        combined_kernel = sum(kernels) / len(kernels)

        # 计算MMD
        K_xx = combined_kernel[:x.size(0), :x.size(0)]
        K_yy = combined_kernel[x.size(0):, x.size(0):]
        K_xy = combined_kernel[:x.size(0), x.size(0):]

        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

        return F.relu(mmd)

    def center_alignment_loss(self, src, bridge, tgt):
        """简单的中心对齐损失"""
        src_center = torch.mean(src, dim=0)
        bridge_center = torch.mean(bridge, dim=0)
        tgt_center = torch.mean(tgt, dim=0)

        # 欧氏距离
        src_bridge_dist = torch.norm(src_center - bridge_center, p=2)
        bridge_tgt_dist = torch.norm(bridge_center - tgt_center, p=2)

        return src_bridge_dist + bridge_tgt_dist

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        """
        三元组损失：使桥梁域（positive）更接近源域（anchor）而不是目标域（negative）
        同时使桥梁域更接近目标域（anchor）而不是源域（negative）
        """
        # 计算距离
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # 三元组损失
        loss = F.relu(pos_dist - neg_dist + margin)

        return torch.mean(loss)