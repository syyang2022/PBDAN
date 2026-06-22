import shutil

import numpy as np
from sklearn.metrics import accuracy_score

import torch


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# def cal_acc(true_y, pred_y, NUM_CLASSES):
#     gt_list = true_y
#     predict_list = pred_y
#     num = NUM_CLASSES
#     acc_sum = 0
#     cal = 0
#     sum_cal = 0
#     for n in range(num):
#         y = []
#         predict_y = []
#         for i in range(len(gt_list)):
#             gt = gt_list[i]
#             predict = predict_list[i]
#             if gt == n:
#                 y.append(gt)
#                 predict_y.append(predict)
#         if n < num-1:
#             cal += accuracy_score(y, predict_y)*len(y)
#     #         print(n)
#         else:
#             sum_cal = cal + accuracy_score(y, predict_y)*len(y)
#     #     print(cal)
#
#     Known_Acc = (cal / (len(gt_list)-len(y)))
#     All_Acc = sum_cal / len(gt_list)
#
#     return All_Acc,Known_Acc
# def cal_acc(true_y, pred_y, NUM_CLASSES):
#     gt_list, predict_list, num = true_y, pred_y, NUM_CLASSES
#
#     n_known_correct = 0          # 已知类预测对的样本数
#     n_known_total   = 0          # 已知类总样本数
#     n_all_correct   = 0          # 全部预测对的样本数
#
#     for n in range(num):
#         mask = np.array(gt_list) == n
#         if mask.sum() == 0:      # 该类没有样本，跳过
#             continue
#         class_correct = (np.array(predict_list)[mask] == n).sum()
#         n_all_correct += class_correct
#         if n < num - 1:          # 已知类
#             n_known_correct += class_correct
#             n_known_total   += mask.sum()
#
#     Known_Acc = n_known_correct / n_known_total if n_known_total else 0.0
#     All_Acc   = n_all_correct / len(gt_list)
#     return Known_Acc, All_Acc

# def cal_acc(true_y, pred_y, NUM_CLASSES):
#     gt_list, predict_list, num = true_y, pred_y, NUM_CLASSES
#
#     n_known_correct = 0          # 已知类预测对的样本数
#     n_known_total   = 0          # 已知类总样本数
#     n_all_correct   = 0          # 全部预测对的样本数
#     n_unknown_correct = 0        # 未知类预测对的样本数
#     n_unknown_total = 0          # 未知类总样本数
#
#     for n in range(num):
#         mask = np.array(gt_list) == n
#         if mask.sum() == 0:      # 该类没有样本，跳过
#             continue
#         class_correct = (np.array(predict_list)[mask] == n).sum()
#         n_all_correct += class_correct
#         if n < num - 1:          # 已知类
#             n_known_correct += class_correct
#             n_known_total   += mask.sum()
#         else:                    # 假设 "Unknown" 类是最后一类
#             n_unknown_correct += class_correct
#             n_unknown_total += mask.sum()
#
#     Known_Acc = n_known_correct / n_known_total if n_known_total else 0.0
#     Unknown_Acc = n_unknown_correct / n_unknown_total if n_unknown_total else 0.0
#     All_Acc   = n_all_correct / len(gt_list)
#
#     return Known_Acc, Unknown_Acc, All_Acc

# def cal_acc(true_y, pred_y, NUM_CLASSES):
#     gt_list, predict_list, num = true_y, pred_y, NUM_CLASSES
#
#     n_known_correct = 0
#     n_known_total = 0
#     n_all_correct = 0
#     n_unknown_correct = 0
#     n_unknown_total = 0
#
#     for n in range(num):
#         mask = np.array(gt_list) == n
#         if mask.sum() == 0:
#             continue
#         class_correct = (np.array(predict_list)[mask] == n).sum()
#         n_all_correct += class_correct
#         if n < num - 1:  # 已知类别
#             n_known_correct += class_correct
#             n_known_total += mask.sum()
#         else:  # 假设未知类别是最后一类
#             n_unknown_correct += class_correct
#             n_unknown_total += mask.sum()
#
#         # 打印调试信息
#         print(f"Class {n}: Correct: {class_correct}, Total: {mask.sum()}")
#
#     Known_Acc = n_known_correct / n_known_total if n_known_total else 0.0
#     Unknown_Acc = n_unknown_correct / n_unknown_total if n_unknown_total else 0.0
#     All_Acc = n_all_correct / len(gt_list)
#
#     print(f"Known Acc: {n_known_correct}/{n_known_total} = {Known_Acc}")
#     print(f"Unknown Acc: {n_unknown_correct}/{n_unknown_total} = {Unknown_Acc}")
#
#     return Known_Acc, Unknown_Acc, All_Acc

def cal_acc(true_y, pred_y, NUM_CLASSES):
    gt_list, predict_list, num = true_y, pred_y, NUM_CLASSES

    n_known_correct = 0
    n_known_total = 0
    n_all_correct = 0
    n_unknown_correct = 0
    n_unknown_total = 0

    for n in range(num):
        mask = np.array(gt_list) == n
        if mask.sum() == 0:
            continue
        class_correct = ((np.array(predict_list) == n) & (np.array(gt_list) == n)).sum()
        n_all_correct += class_correct
        if n < num - 1:  # 已知类别
            n_known_correct += class_correct
            n_known_total += mask.sum()
        else:  # 假设未知类别是最后一类
            n_unknown_correct += class_correct
            n_unknown_total += mask.sum()
         # 打印调试信息
        print(f"Class {n}: Correct: {class_correct}, Total: {mask.sum()}")

    Known_Acc = n_known_correct / n_known_total if n_known_total else 0.0
    Unknown_Acc = n_unknown_correct / n_unknown_total if n_unknown_total else 0.0
    All_Acc = n_all_correct / len(gt_list)

    print(f"Known Acc: {n_known_correct}/{n_known_total} = {Known_Acc}")
    print(f"Unknown Acc: {n_unknown_correct}/{n_unknown_total} = {Unknown_Acc}")

    return Known_Acc, Unknown_Acc, All_Acc


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))