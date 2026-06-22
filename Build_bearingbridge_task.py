#!/usr/bin/env python
# coding: utf-8

# # Load the dataset

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import torch
from torch.utils import data as Data

X_0 = np.load("./bearingset/raw/PDB_0_FFT.npy")
X_1 = np.load("./bearingset/raw/PDB_1_FFT.npy")
X_2 = np.load("./bearingset/raw/PDB_2_FFT.npy")
X_3 = np.load("./bearingset/raw/PDB_3_FFT.npy")

F_0 = np.load("./bearingset/raw/FFT_0.npy")
F_1 = np.load("./bearingset/raw/FFT_1.npy")
F_2 = np.load("./bearingset/raw/FFT_2.npy")
F_3 = np.load("./bearingset/raw/FFT_3.npy")

Y = np.zeros((8000, 8))
for i in range(8):
    Y[1000 * i:1000 * i + 1000, i] = 1
np.save("./bearingset/data_thirdspace/Label.npy", Y)
Y_image = np.zeros((800, 8))
for i in range(8):
    Y_image[100 * i:100 * i + 100, i] = 1
np.save("./bearingset/data_thirdspace/Label_image.npy", Y_image)


# In[2]:

def create_bridge_domain(source_data, target_data, bridge_ratio):
    """
    创建桥梁域 - 作为源域和目标域的中间过渡域
    bridge_ratio: 桥梁域中源域和目标域数据的混合比例
    """
    # 确保两个域的数据量相同
    min_samples = min(len(source_data), len(target_data))
    source_samples = int(min_samples * bridge_ratio)
    target_samples = min_samples - source_samples

    # 从源域和目标域各取一部分数据构建桥梁域
    bridge_source = source_data[:source_samples]
    bridge_target = target_data[:target_samples]
    bridge_domain = np.vstack([bridge_source, bridge_target])

    return bridge_domain


def create_bridge_labels(source_label, target_label, bridge_samples):
    """
    创建桥梁域的标签
    桥梁域包含源域和目标域的部分数据，标签需要相应处理
    """
    source_samples = bridge_samples // 2
    target_samples = bridge_samples - source_samples

    bridge_label = np.vstack([
        source_label[:source_samples],
        target_label[:target_samples]
    ])

    return bridge_label


# # Build the OSDT task-1 with Bridge Domains

# source Domain:
# 1,2,3,4,6,7,8
#
# target Domain:
# 1,2,3,4,**5**,6,7,8

# In[3]:


def T1(X_0, X_1, X_2, X_3, transfer_num):
    if transfer_num == 0:
        # Transfer 0 - 1
        source = np.vstack((X_0[0:100], X_0[100:200], X_0[200:300], X_0[300:400], X_0[500:600], X_0[600:700],
                            X_0[700:800], X_0[700:800]))
        target = np.vstack((X_1[0:100], X_1[100:200], X_1[200:300], X_1[300:400], X_1[500:600], X_1[600:700],
                            X_1[700:800], X_1[400:500]))
    if transfer_num == 1:
        # Transfer 0 - 2
        source = np.vstack((X_0[0:100], X_0[100:200], X_0[200:300], X_0[300:400], X_0[500:600], X_0[600:700],
                            X_0[700:800], X_0[700:800]))
        target = np.vstack((X_2[0:100], X_2[100:200], X_2[200:300], X_2[300:400], X_2[500:600], X_2[600:700],
                            X_2[700:800], X_2[400:500]))
    if transfer_num == 2:
        # Transfer 0 - 3
        source = np.vstack((X_0[0:100], X_0[100:200], X_0[200:300], X_0[300:400], X_0[500:600], X_0[600:700],
                            X_0[700:800], X_0[700:800]))
        target = np.vstack((X_3[0:100], X_3[100:200], X_3[200:300], X_3[300:400], X_3[500:600], X_3[600:700],
                            X_3[700:800], X_3[400:500]))
    if transfer_num == 3:
        # Transfer 1 - 2
        source = np.vstack((X_1[0:100], X_1[100:200], X_1[200:300], X_1[300:400], X_1[500:600], X_1[600:700],
                            X_1[700:800], X_1[700:800]))
        target = np.vstack((X_2[0:100], X_2[100:200], X_2[200:300], X_2[300:400], X_2[500:600], X_2[600:700],
                            X_2[700:800], X_2[400:500]))
    if transfer_num == 4:
        # Transfer 1 - 3
        source = np.vstack((X_1[0:100], X_1[0:50], X_1[100:200], X_1[250:300], X_1[300:400], X_1[500:600], X_1[600:700],
                            X_1[700:800], X_1[700:800]))
        target = np.vstack((X_3[0:100], X_3[100:200], X_3[200:300], X_3[300:400], X_3[500:600], X_3[600:700],
                            X_3[700:800], X_3[400:500]))
    if transfer_num == 5:
        # Transfer 2 - 3
        source = np.vstack((X_2[0:100], X_2[100:200], X_2[200:300], X_2[300:400], X_2[500:600], X_2[600:700],
                            X_2[700:800], X_2[700:800]))
        target = np.vstack((X_3[0:100], X_3[100:200], X_3[200:300], X_3[300:400], X_3[500:600], X_3[600:700],
                            X_3[700:800], X_3[400:500]))

    # 创建桥梁域
    bridge = create_bridge_domain(source, target, bridge_ratio=0.5)

    label_source = np.zeros((800, 8))
    label_source[0:150, 0] = 1
    label_source[150:250, 1] = 1
    label_source[250:300, 2] = 1
    label_source[300:400, 3] = 1
    label_source[400:500, 4] = 1
    label_source[500:600, 5] = 1
    label_source[600:700, 6] = 1
    label_source[700:800, 6] = 1

    label_target = np.zeros((800, 8))
    label_target[0:100, 0] = 1
    label_target[100:200, 1] = 1
    label_target[200:300, 2] = 1
    label_target[300:400, 3] = 1
    label_target[400:500, 4] = 1
    label_target[500:600, 5] = 1
    label_target[600:700, 6] = 1
    label_target[700:800, 7] = 1

    # 创建桥梁域的标签
    label_bridge = create_bridge_labels(label_source, label_target, len(bridge))

    # 保存所有域的数据和标签
    np.save("./bearingset/data_thirdspace/T1_source_image.npy", source)
    np.save("./bearingset/data_thirdspace/T1_target_image.npy", target)
    np.save("./bearingset/data_thirdspace/T1_bridge_image.npy", bridge)
    np.save("./bearingset/data_thirdspace/T1_source_label.npy", label_source)
    np.save("./bearingset/data_thirdspace/T1_target_label.npy", label_target)
    np.save("./bearingset/data_thirdspace/T1_bridge_label.npy", label_bridge)


# # Build the OSDT task-2 with Bridge Domains

# source Domain:
# 1,3,4,6,7,8
#
# target Domain:
# 1,3,4,6,7,8,**5**,**2**

# In[4]:


def T2(X_0, X_1, X_2, X_3, transfer_num):
    if transfer_num == 0:
        # Transfer 0 - 1
        source = np.vstack((X_0[0:100], X_0[200:300], X_0[300:400], X_0[500:600], X_0[600:700], X_0[600:700],
                            X_0[700:800], X_0[700:800]))
        target = np.vstack((X_1[0:100], X_1[200:300], X_1[300:400], X_1[500:600], X_1[600:700], X_1[700:800],
                            X_1[400:500], X_1[100:200]))
    if transfer_num == 1:
        # Transfer 0 - 2
        source = np.vstack((X_0[0:100], X_0[200:300], X_0[300:400], X_0[500:600], X_0[600:700], X_0[600:700],
                            X_0[700:800], X_0[700:800]))
        target = np.vstack((X_2[0:100], X_2[200:300], X_2[300:400], X_2[500:600], X_2[600:700], X_2[700:800],
                            X_2[400:500], X_2[100:200]))
    if transfer_num == 2:
        # Transfer 0 - 3
        source = np.vstack((X_0[0:100], X_0[200:300], X_0[300:400], X_0[500:600], X_0[600:700], X_0[600:700],
                            X_0[700:800], X_0[700:800]))
        target = np.vstack((X_3[0:100], X_3[200:300], X_3[300:400], X_3[500:600], X_3[600:700], X_3[700:800],
                            X_3[400:500], X_3[100:200]))
    if transfer_num == 3:
        # Transfer 1 - 2
        source = np.vstack((X_1[0:100], X_1[200:300], X_1[300:400], X_1[500:600], X_1[600:700], X_1[600:700],
                            X_1[700:800], X_1[700:800]))
        target = np.vstack((X_2[0:100], X_2[200:300], X_2[300:400], X_2[500:600], X_2[600:700], X_2[700:800],
                            X_2[400:500], X_2[100:200]))
    if transfer_num == 4:
        # Transfer 1 - 3
        source = np.vstack((X_1[0:100], X_1[200:300], X_1[300:400], X_1[500:600], X_1[600:700], X_1[600:700],
                            X_1[700:800], X_1[700:800]))
        target = np.vstack((X_3[0:100], X_3[200:300], X_3[300:400], X_3[500:600], X_3[600:700], X_3[700:800],
                            X_3[400:500], X_3[100:200]))
    if transfer_num == 5:
        # Transfer 2 - 3
        source = np.vstack((X_2[0:100], X_2[200:300], X_2[300:400], X_2[500:600], X_2[600:700], X_2[600:700],
                            X_2[700:800], X_2[700:800]))
        target = np.vstack((X_3[0:100], X_3[200:300], X_3[300:400], X_3[500:600], X_3[600:700], X_3[700:800],
                            X_3[400:500], X_3[100:200]))

    # 创建桥梁域
    bridge = create_bridge_domain(source, target, bridge_ratio=0.5)

    label_source = np.zeros((800, 7))
    label_source[0:100, 0] = 1
    label_source[100:200, 1] = 1
    label_source[200:300, 2] = 1
    label_source[300:400, 3] = 1
    label_source[400:500, 4] = 1
    label_source[500:600, 4] = 1
    label_source[600:700, 5] = 1
    label_source[700:800, 5] = 1

    label_target = np.zeros((800, 7))
    label_target[0:100, 0] = 1
    label_target[100:200, 1] = 1
    label_target[200:300, 2] = 1
    label_target[300:400, 3] = 1
    label_target[400:500, 4] = 1
    label_target[500:600, 5] = 1
    label_target[600:700, 6] = 1
    label_target[700:800, 6] = 1

    # 创建桥梁域的标签
    label_bridge = create_bridge_labels(label_source, label_target, len(bridge))

    # 保存所有域的数据和标签
    np.save("./bearingset/data_thirdspace/T2_source_image.npy", source)
    np.save("./bearingset/data_thirdspace/T2_target_image.npy", target)
    np.save("./bearingset/data_thirdspace/T2_bridge_image.npy", bridge)
    np.save("./bearingset/data_thirdspace/T2_source_label.npy", label_source)
    np.save("./bearingset/data_thirdspace/T2_target_label.npy", label_target)
    np.save("./bearingset/data_thirdspace/T2_bridge_label.npy", label_bridge)


# # Build the OSDT task-3 with Bridge Domains

# source Domain:
# 1,2,3,4,5
#
# target Domain:
# 1,2,3,4,5,**6**,**7**,**8**

# In[5]:


def T3(X_0, X_1, X_2, X_3, transfer_num):
    if transfer_num == 0:
        # Transfer 0 - 1
        source = np.vstack((
                           X_0[0:100], X_0[100:200], X_0[200:300], X_0[300:400], X_0[400:500], X_0[0:100], X_0[100:200],
                           X_0[200:300]))
        target = np.vstack((X_1[0:100], X_1[100:200], X_1[200:300], X_1[300:400], X_1[400:500], X_1[500:600],
                            X_1[600:700], X_1[700:800]))
    if transfer_num == 1:
        # Transfer 0 - 2
        source = np.vstack((
                           X_0[0:100], X_0[100:200], X_0[200:300], X_0[300:400], X_0[400:500], X_0[0:100], X_0[100:200],
                           X_0[200:300]))
        target = np.vstack((X_2[0:100], X_2[100:200], X_2[200:300], X_2[300:400], X_2[400:500], X_2[500:600],
                            X_2[600:700], X_2[700:800]))
    if transfer_num == 2:
        # Transfer 0 - 3
        source = np.vstack((
                           X_0[0:100], X_0[100:200], X_0[200:300], X_0[300:400], X_0[400:500], X_0[0:100], X_0[100:200],
                           X_0[200:300]))
        target = np.vstack((X_3[0:100], X_3[100:200], X_3[200:300], X_3[300:400], X_3[400:500], X_3[500:600],
                            X_3[600:700], X_3[700:800]))
    if transfer_num == 3:
        # Transfer 1 - 2
        source = np.vstack((
                           X_1[0:100], X_1[100:200], X_1[200:300], X_1[300:400], X_1[400:500], X_1[0:100], X_1[100:200],
                           X_1[200:300]))
        target = np.vstack((X_2[0:100], X_2[100:200], X_2[200:300], X_2[300:400], X_2[400:500], X_2[500:600],
                            X_2[600:700], X_2[700:800]))
    if transfer_num == 4:
        # Transfer 1 - 3
        source = np.vstack((
                           X_1[0:100], X_1[100:200], X_1[200:300], X_1[300:400], X_1[400:500], X_1[0:100], X_1[100:200],
                           X_1[200:300]))
        target = np.vstack((X_3[0:100], X_3[100:200], X_3[200:300], X_3[300:400], X_3[400:500], X_3[500:600],
                            X_3[600:700], X_3[700:800]))
    if transfer_num == 5:
        # Transfer 2 - 3
        source = np.vstack((
                           X_2[0:100], X_2[100:200], X_2[200:300], X_2[300:400], X_2[400:500], X_2[0:100], X_2[100:200],
                           X_2[200:300]))
        target = np.vstack((X_3[0:100], X_3[100:200], X_3[200:300], X_3[300:400], X_3[400:500], X_3[500:600],
                            X_3[600:700], X_3[700:800]))

    # 创建桥梁域
    bridge = create_bridge_domain(source, target, bridge_ratio=0.5)

    label_source = np.zeros((800, 6))
    label_source[0:100, 0] = 1
    label_source[100:200, 1] = 1
    label_source[200:300, 2] = 1
    label_source[300:400, 3] = 1
    label_source[400:500, 4] = 1
    label_source[500:600, 0] = 1
    label_source[600:700, 1] = 1
    label_source[700:800, 2] = 1

    label_target = np.zeros((800, 6))
    label_target[0:100, 0] = 1
    label_target[100:200, 1] = 1
    label_target[200:300, 2] = 1
    label_target[300:400, 3] = 1
    label_target[400:500, 4] = 1
    label_target[500:600, 5] = 1
    label_target[600:700, 5] = 1
    label_target[700:800, 5] = 1

    # 创建桥梁域的标签
    label_bridge = create_bridge_labels(label_source, label_target, len(bridge))

    # 保存所有域的数据和标签
    np.save("./bearingset/data_thirdspace/T3_source_image.npy", source)
    np.save("./bearingset/data_thirdspace/T3_target_image.npy", target)
    np.save("./bearingset/data_thirdspace/T3_bridge_image.npy", bridge)
    np.save("./bearingset/data_thirdspace/T3_source_label.npy", label_source)
    np.save("./bearingset/data_thirdspace/T3_target_label.npy", label_target)
    np.save("./bearingset/data_thirdspace/T3_bridge_label.npy", label_bridge)

# # Build the OSDT task-4 with Bridge Domains

# **Source Domain:
# 1,4,6,8**
#
# **Target Domain:
# 1,5*,4,6,8**

# In[6]:


def T4(X_0, X_1, X_2, X_3, transfer_num):
    if transfer_num == 0:
        # Transfer 0 - 1
        source = np.vstack((
                           X_0[0:100], X_0[300:400], X_0[500:600], X_0[700:800], X_0[0:100], X_0[300:400], X_0[500:600],
                           X_0[700:800]))
        target = np.vstack((X_1[0:100], X_1[300:400], X_1[500:600], X_1[700:800], X_1[400:500], X_1[400:500],
                            X_1[400:500], X_1[400:500]))
    if transfer_num == 1:
        # Transfer 0 - 2
        source = np.vstack((
                           X_0[0:100], X_0[300:400], X_0[500:600], X_0[700:800], X_0[0:100], X_0[300:400], X_0[500:600],
                           X_0[700:800]))
        target = np.vstack((X_2[0:100], X_2[300:400], X_2[500:600], X_2[700:800], X_2[400:500], X_2[400:500],
                            X_2[400:500], X_2[400:500]))
    if transfer_num == 2:
        # Transfer 0 - 3
        source = np.vstack((
                           X_0[0:100], X_0[300:400], X_0[500:600], X_0[700:800], X_0[0:100], X_0[300:400], X_0[500:600],
                           X_0[700:800]))
        target = np.vstack((X_3[0:100], X_3[300:400], X_3[500:600], X_3[700:800], X_3[400:500], X_3[400:500],
                            X_3[400:500], X_3[400:500]))
    if transfer_num == 3:
        # Transfer 1 - 2
        source = np.vstack((
                           X_1[0:100], X_1[300:400], X_1[500:600], X_1[700:800], X_1[0:100], X_1[300:400], X_1[500:600],
                           X_1[700:800]))
        target = np.vstack((X_2[0:100], X_2[300:400], X_2[500:600], X_2[700:800], X_2[400:500], X_2[400:500],
                            X_2[400:500], X_2[400:500]))
    if transfer_num == 4:
        # Transfer 1 - 3
        source = np.vstack((
                           X_1[0:100], X_1[300:400], X_1[500:600], X_1[700:800], X_1[0:100], X_1[300:400], X_1[500:600],
                           X_1[700:800]))
        target = np.vstack((X_3[0:100], X_3[300:400], X_3[500:600], X_3[700:800], X_3[400:500], X_3[400:500],
                            X_3[400:500], X_3[400:500]))
    if transfer_num == 5:
        # Transfer 2 - 3
        source = np.vstack((
                           X_2[0:100], X_2[300:400], X_2[500:600], X_2[700:800], X_2[0:100], X_2[300:400], X_2[500:600],
                           X_2[700:800]))
        target = np.vstack((X_3[0:100], X_3[300:400], X_3[500:600], X_3[700:800], X_3[400:500], X_3[400:500],
                            X_3[400:500], X_3[400:500]))

    # 创建桥梁域
    bridge = create_bridge_domain(source, target, bridge_ratio=0.5)

    label_source = np.zeros((800, 5))
    label_source[0:100, 0] = 1
    label_source[100:200, 1] = 1
    label_source[200:300, 2] = 1
    label_source[300:400, 3] = 1
    label_source[400:500, 0] = 1
    label_source[500:600, 1] = 1
    label_source[600:700, 2] = 1
    label_source[700:800, 3] = 1

    label_target = np.zeros((800, 5))
    label_target[0:100, 0] = 1
    label_target[100:200, 1] = 1
    label_target[200:300, 2] = 1
    label_target[300:400, 3] = 1
    label_target[400:800, 4] = 1

    # 创建桥梁域的标签
    label_bridge = create_bridge_labels(label_source, label_target, len(bridge))

    # 保存所有域的数据和标签
    np.save("./bearingset/data_thirdspace/T4_source_image.npy", source)
    np.save("./bearingset/data_thirdspace/T4_target_image.npy", target)
    np.save("./bearingset/data_thirdspace/T4_bridge_image.npy", bridge)
    np.save("./bearingset/data_thirdspace/T4_source_label.npy", label_source)
    np.save("./bearingset/data_thirdspace/T4_target_label.npy", label_target)
    np.save("./bearingset/data_thirdspace/T4_bridge_label.npy", label_bridge)


# # Generate all OSDT tasks with Enhanced Domains

# In[7]:


transfer_num = 4

print("Generating T1 with bridge domains...")
T1(X_0, X_1, X_2, X_3, transfer_num)
print("T1 generation completed!")

print("Generating T2 with bridge domains...")
T2(X_0, X_1, X_2, X_3, transfer_num)
print("T2 generation completed!")

print("Generating T3 with bridge domains...")
T3(X_0, X_1, X_2, X_3, transfer_num)
print("T3 generation completed!")

print("Generating T4 with bridge domains...")
T4(X_0, X_1, X_2, X_3, transfer_num)
print("T4 generation completed!")

print("All enhanced OSDT tasks with bridge domains have been generated successfully!")