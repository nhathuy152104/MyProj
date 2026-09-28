import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
from lib.utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
from lib.options import args_parser
from lib.update import LocalUpdate
from lib.models.models import ClientModel
import os

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))


def FedFM(args, train_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    protos_path = "/kaggle/working/MyProj/exps/global_protos_dict.pt"
    
    if os.path.exists(protos_path):
        # Đọc Dictionary chứa Tensor từ file vào bộ nhớ
        global_protos = torch.load(protos_path)
        print(f"[*] Đã tải thành công tâm cụm toàn cục từ '{protos_path}'")
        print(f"    Số class được load: {len(global_protos)}")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy '{protos_path}'. Sẽ khởi tạo rỗng.")
        # Khởi tạo Dictionary rỗng (không dùng List [] để tránh lỗi TypeError ở hàm Loss)
        global_protos = {}

    idxs_users = np.arange(args.num_users)
    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users: 
            local_model = LocalUpdate(args = args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc = local_model.update_weights_fedfm(idx, global_protos, model = copy.deepcopy(local_model_list[idx]), global_round = round)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            print('Train/Loss/user' + str(idx + 1), loss['total'], round)
            print('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            print('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            print('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        local_weights_list = local_weights

        global_head_weights = {}

        head_keys = [k for k in local_weights_list[0].keys() if 'head' in k]

        for key in head_keys:
            global_head_weights[key] = sum(local_weights[idx][key] for idx in range(len(idxs_users))) / len(idxs_users)

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])

            client_weights = local_weights_list[idx]

            for key in head_keys:
                client_weights[key] = copy.deepcopy(global_head_weights[key])

            local_model.load_state_dict(client_weights, strict=True)
            local_model_list[idx] = local_model 
        # update global weights

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    # acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    # print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    # print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    # print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    # save protos




if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()

    exp_details(args)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)

    else:
        torch.manual_seed(args.seed)

    np.random.seed(args.seed)
    random.seed(args.seed)

    n_list = 3
    print("n_list")
    print(n_list)
    if args.dataset == 'covid19':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev - 1, args.num_users)
    print("k_list")
    print(k_list)
    train_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list) 
    print(len(train_dataset))
    local_model_list = []
    for i in range(args.num_users):
        args.out_channels = 4
        local_model = ClientModel()

        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    FedFM(args, train_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    
