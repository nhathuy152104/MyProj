#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import covid19_iid
import covid19
import numpy as np
import constants

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])



trans_covid19_train = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.2,0.2),
                transforms.RandomAffine(degrees=10, scale=(0.8,1.1), translate=(0.0625,0.0625)),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
            )
def get_dataset(args, n_list, k_list):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir + args.dataset
    
    # Bổ sung khởi tạo các biến mặc định để tránh lỗi UnboundLocalError 
    # khi chạy nhánh IID (vì nhánh IID không tạo ra classes_list, user_groups_lt...)
    user_groups_lt = None
    classes_list = None
    classes_list_gt = None

    if args.dataset == 'covid19':
        # Sửa lại Transforms: Ảnh y tế thường cần Resize về kích thước chuẩn (ví dụ 224x224) 
        # và Normalize theo kênh màu RGB chuẩn của ImageNet
        apply_transform = trans_covid19_train

        # Đã SỬA: Thay datasets.MNIST bằng covid19.COVID19 (gọi từ file covid19.py của bạn)
        train_dataset = covid19.COVID19(args, data_dir, train=True, transform=apply_transform)

        # sample training data amongst users
        user_groups = covid19_iid(train_dataset, args.num_users)

    return train_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != '....':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_sem(w, n_list):
    """
    Returns the average of the weights.
    """
    k = 2
    model_dict = {}
    for i in range(k):
        model_dict[i] = []

    idx = 0
    for i in n_list:
        if i< np.mean(n_list):
            model_dict[0].append(idx)
        else:
            model_dict[1].append(idx)
        idx += 1

    ww = copy.deepcopy(w)
    for cluster_id in model_dict.keys():
        model_id_list = model_dict[cluster_id]
        w_avg = copy.deepcopy(w[model_id_list[0]])
        for key in w_avg.keys():
            for j in range(1, len(model_id_list)):
                w_avg[key] += w[model_id_list[j]][key]
            w_avg[key] = torch.true_divide(w_avg[key], len(model_id_list))
            # w_avg[key] = torch.div(w_avg[key], len(model_id_list))
        for model_id in model_id_list:
            for key in ww[model_id].keys():
                ww[model_id][key] = w_avg[key]

    return ww

def average_weights_per(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:2] != 'fc':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            # w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_het(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != 'fc2.':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}\n')
    return