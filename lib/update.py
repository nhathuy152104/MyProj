#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset,WeightedRandomSampler,Subset
import copy
import numpy as np
from .losses import MultiProtoSupConLoss
import os
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, val_dataset, idxs):
        self.args = args
        self.trainloader = DataLoader(
            dataset, 
            batch_size=self.args.local_bs, 
            shuffle=True,       # Đảm bảo các batch trộn đều cả 4 class
            drop_last=True
        )
        self.device = args.device
        counts = torch.tensor([3600.0, 6000.0, 10000.0, 1300.0])
        weights = 1.0 / torch.sqrt(counts)
        weights = (weights / weights.sum()) * 4.0

        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        self.contrastive_loss = MultiProtoSupConLoss().to(self.device)
        self.testloader = DataLoader(val_dataset, batch_size=32, drop_last=False)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(1.0 * len(idxs))]
        
        # Tạo tập dataset con (Subset) cho riêng client này
        train_dataset = Subset(dataset, idxs_train)

        # 1. Trích xuất nhãn (targets) của riêng phần dữ liệu thuộc Client này
        # dataset.targets là một tensor chứa toàn bộ nhãn, ta dùng idxs_train để slice
        train_targets = dataset.targets[idxs_train].numpy()

        # 2. Đếm số lượng ảnh của từng class (Giả sử có 4 class từ 0 đến 3)
        class_counts = np.bincount(train_targets, minlength=4)
        
        # 3. Tính trọng số cho từng class (nghịch đảo của số lượng). 
        # Nếu số lượng = 0 thì gán trọng số = 0 để tránh lỗi chia cho 0
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        
        # 4. Gán trọng số tương ứng cho từng ảnh (sample)
        samples_weight = class_weights[train_targets]
        samples_weight = torch.from_numpy(samples_weight).double()

        # 5. Khởi tạo Sampler
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        # 6. Truyền sampler vào DataLoader
        # QUAN TRỌNG: Khi dùng sampler, BẮT BUỘC phải bỏ tham số shuffle=True
        trainloader = DataLoader(
            train_dataset, 
            batch_size=self.args.local_bs, 
            sampler=sampler, 
            drop_last=True # Có thể giữ lại drop_last nếu muốn batch size luôn cố định
        )

        return trainloader


    def update_weights_fedfm(self, idx, global_protos, model, global_round):
        epoch_loss = {'total': [], '1': [], '2': []}    
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.train_ep, eta_min=1e-5)
        
        # Danh sách tên các nhãn theo đúng thứ tự index
        categories = ["COVID", "Lung_Op", "Normal", "Viral_Pneu"]
        best_balanced_acc = 0.0
        for iter in range(self.args.train_ep):
            model.train()

            batch_loss = {'total': [], '1': [], '2': []}
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                log_probs, protos = model.forward(images)
                loss1 = self.criterion(log_probs, labels)
                loss2 = 0.3 * self.contrastive_loss.forward(protos, labels, global_protos)

                loss = loss1
                loss.backward()
                optimizer.step()
                _, y_hat = log_probs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f} \tLoss1: {:.3f} | Loss2: {:.3f}'.format(
                            global_round, 
                            idx, 
                            iter, 
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader),
                            loss.item(),
                            acc_val.item(), 
                            loss1.item(),
                            loss2.item()
                        )
                    )
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())

            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print("lr: ", current_lr)

            # Đánh giá và in chi tiết từng class
            val_acc, val_loss, val_loss1, val_loss2, per_class_acc = self.inference(model, global_protos)
            
            # Format chuỗi hiển thị theo từng nhãn
            class_acc_str = " | ".join([f"{cat}: {acc*100:.2f}%" for cat, acc in zip(categories, per_class_acc)])
            balanced_acc = per_class_acc.mean() * 100

            print(f"Val Loss: {val_loss:.4f} | Overall Acc: {val_acc*100:.2f}% (Loss1: {val_loss1:.4f} | Loss2: {val_loss2:.4f})")
            print(f"   -> Balanced Acc: {balanced_acc:.2f}%")
            print(f"   -> Per-class: [{class_acc_str}]")

            # 2. LOGIC LƯU BEST MODEL DỰA TRÊN BALANCED ACCURACY
            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                # Lưu vào RAM để gửi lên Server
                best_model_weights = copy.deepcopy(model.state_dict())
                
                # Lưu ra ổ cứng để backup
                import os
                os.makedirs("local_checkpoints", exist_ok=True)
                save_path = f"local_checkpoints/client_{idx}_best_round_{global_round}.pth"
                torch.save(best_model_weights, save_path)
                print(f"   *** [NEW BEST] Đã lưu model đạt Balanced Acc: {best_balanced_acc:.2f}% tại {save_path} ***")

        # Tính trung bình loss của cả round

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        import os
        os.makedirs("local_checkpoints", exist_ok=True) # Tạo thư mục lưu trữ nếu chưa có
        save_path = f"local_checkpoints/client_{idx}_round_{global_round}.pth"
        
        torch.save(model.state_dict(), save_path)
        print(f"[SAVE] Đã lưu trọng số cục bộ tại: {save_path}")
        # -----------------------------------
        return model.state_dict(), epoch_loss, acc_val.item()
                
    def inference(self, model, global_protos):
        """ Returns the true inference overall accuracy, average losses,
            and per-class accuracy.
        """
        model.eval()
        total_loss, total_loss1, total_loss2 = 0.0, 0.0, 0.0
        total, correct = 0.0, 0.0

        # Giả sử có 4 classes: 0: COVID, 1: Lung_Opacity, 2: Normal, 3: Viral Pneumonia
        num_classes = 4
        class_correct = torch.zeros(num_classes, device=self.device)
        class_total = torch.zeros(num_classes, device=self.device)

        # Tắt gradient khi kiểm thử để tiết kiệm VRAM và tăng tốc
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                batch_sz = labels.size(0)

                # Inference
                log_probs, protos = model.forward(images)
                l1 = self.criterion(log_probs, labels)
                l2 = self.contrastive_loss.forward(protos, labels, global_protos)
                
                # Nhân lại với batch_sz để tính đúng tổng loss tích lũy
                total_loss1 += l1.item() * batch_sz
                total_loss2 += l2.item() * batch_sz
                total_loss += (l1.item() + l2.item()) * batch_sz

                # Prediction
                _, pred_labels = torch.max(log_probs, 1)
                pred_labels = pred_labels.view(-1)
                labels = labels.view(-1)

                correct_mask = torch.eq(pred_labels, labels)
                correct += torch.sum(correct_mask).item()
                total += batch_sz

                # Đếm số lượng mẫu đúng và tổng số mẫu theo từng class
                for c in range(num_classes):
                    c_mask = (labels == c)
                    class_total[c] += torch.sum(c_mask).item()
                    class_correct[c] += torch.sum(correct_mask & c_mask).item()

        overall_acc = correct / total
        avg_loss = total_loss / total
        avg_loss1 = total_loss1 / total
        avg_loss2 = total_loss2 / total

        # Tính Accuracy cho từng lớp (tránh chia cho 0 nếu class_total = 0)
        per_class_acc = (class_correct / torch.clamp(class_total, min=1.0)).cpu().numpy()

        return overall_acc, avg_loss, avg_loss1, avg_loss2, per_class_acc

class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=64, shuffle=False)
        return testloader

    def get_result(self, args, idx, classes_list, model):
        # Set mode to train model
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            model.zero_grad()
            outputs, protos = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            outputs = outputs[: , 0 : args.num_classes]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total

        return loss, acc

    def fine_tune(self, args, dataset, idxs, model):
        trainloader = self.test_split(dataset, list(idxs))
        device = args.device
        criterion = nn.NLLLoss().to(device)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        model.train()
        for i in range(args.ft_round):
            for batch_idx, (images, label_g) in enumerate(trainloader):
                images, labels = images.to(device), label_g.to(device)

                # compute loss
                model.zero_grad()
                log_probs, protos = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()


def test_inference(args, model, test_dataset, global_protos):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs, protos = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def test_inference_new(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        cnt = np.zeros(10)
        for i in range(10):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(10):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc

def test_inference_new_cifar(args, local_model_list, test_dataset, classes_list, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        outputs = torch.zeros(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        cnt = np.zeros(100)
        for i in range(100):
            for idx in range(args.num_users):
                if i in classes_list[idx]:
                    tmp = np.where(classes_list[idx] == i)[0][0]
                    outputs[:,i] += prob_list[idx][:,tmp]
                    cnt[i]+=1
        for i in range(100):
            if cnt[i]!=0:
                outputs[:, i] = outputs[:,i]/cnt[i]

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)


    acc = correct/total

    return loss, acc


def test_inference_new_het(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        protos_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            _, protos = model(images)
            protos_list.append(protos)

        ensem_proto = torch.zeros(size=(images.shape[0], protos.shape[1])).to(device)
        # protos ensemble
        for protos in protos_list:
            ensem_proto += protos
        ensem_proto /= len(protos_list)

        a_large_num = 100
        outputs = a_large_num * torch.ones(size=(images.shape[0], 10)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(10):
                if j in global_protos.keys():
                    dist = loss_mse(ensem_proto[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        # Prediction
        _, pred_labels = torch.min(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    acc = correct/total

    return acc

def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)

        # test (use global proto)
        if global_protos!=[]:
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                outputs, protos = model(images)

                # compute the dist between protos and global_protos
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                # compute loss
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.detach().numpy()

            acc = correct / total
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            loss_list.append(loss2)

    return acc_list_l, acc_list_g, loss_list


def save_protos(args, local_model_list, test_dataset, user_groups_gt):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.device
    criterion = nn.NLLLoss().to(device)

    agg_protos_label = {}
    for idx in range(args.num_users):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label[idx]:
                    agg_protos_label[idx][labels[i].item()].append(protos[i, :])
                else:
                    agg_protos_label[idx][labels[i].item()] = [protos[i, :]]

    x = []
    y = []
    d = []
    for i in range(args.num_users):
        for label in agg_protos_label[i].keys():
            for proto in agg_protos_label[i][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")

def test_inference_new_het_cifar(args, local_model_list, test_dataset, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnt = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        prob_list = []
        for idx in range(args.num_users):
            images = images.to(args.device)
            model = local_model_list[idx]
            probs, protos = model(images)  # outputs 64*6
            prob_list.append(probs)

        a_large_num = 1000
        outputs = a_large_num * torch.ones(size=(images.shape[0], 100)).to(device)  # outputs 64*10
        for i in range(images.shape[0]):
            for j in range(100):
                if j in global_protos.keys():
                    dist = loss_mse(protos[i,:],global_protos[j][0])
                    outputs[i,j] = dist

        _, pred_labels = torch.topk(outputs, 5)
        for i in range(pred_labels.shape[1]):
            correct += torch.sum(torch.eq(pred_labels[:,i], labels)).item()
        total += len(labels)

        cnt+=1
        if cnt==20:
            break

    acc = correct/total

    return acc