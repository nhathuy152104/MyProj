import torch
import torch.nn as nn
import torch.nn.functional as F

def format_global_protos_for_loss(global_protos_dict, num_classes, feature_dim, device):
    """
    Biến đổi Dictionary các sub-prototypes không đồng đều thành 3D Tensor hoàn hảo.
    """
    if not global_protos_dict or len(global_protos_dict) == 0:
        return torch.empty(0, device=device) 

    max_protos = max([len(protos) for protos in global_protos_dict.values()])
    formatted_tensor = torch.zeros((num_classes, max_protos, feature_dim), device=device)

    for class_idx in range(num_classes):
        if class_idx in global_protos_dict:
            class_protos = global_protos_dict[class_idx] 
            num_existing = len(class_protos)
            
            for i in range(max_protos):
                if i < num_existing:
                    formatted_tensor[class_idx, i, :] = class_protos[i].to(device)
                else:
                    formatted_tensor[class_idx, i, :] = class_protos[-1].to(device)
        else:
            pass
            
    return formatted_tensor

class MultiProtoSupConLoss(nn.Module):
    def __init__(self, temperature = 0.97):
        super().__init__()
        self.temp = temperature

    def forward(self, z, labels, global_prototypes):
        batch_size = z.size(0)

        # 1. CHUYỂN ĐỔI DICT SANG TENSOR
        if isinstance(global_prototypes, dict):
            # Lưu ý: num_classes = 4 dựa trên bộ dữ liệu COVID-19 của bạn
            global_prototypes = format_global_protos_for_loss(
                global_prototypes, 
                num_classes=4, 
                feature_dim=z.size(1), 
                device=z.device
            )
            print(global_prototypes)
        # 2. XỬ LÝ VÒNG 1 (Nếu chưa có dữ liệu, trả về Loss = 0)
        if not isinstance(global_prototypes, torch.Tensor) or global_prototypes.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=z.device)

        global_prototypes = global_prototypes.to(z.device)

        # 3. TÍNH TOÁN LOSS
        # Lúc này global_prototypes CHẮC CHẮN là 3D Tensor, hàm unsqueeze sẽ hoạt động
        sim_matrix = F.cosine_similarity(
            z.unsqueeze(1).unsqueeze(1), 
            global_prototypes.unsqueeze(0),
            dim = -1
        )
        sim_matrix = sim_matrix / self.temp

        pos_sims = sim_matrix[torch.arange(batch_size), labels]
        pos_logits, _ = torch.max(pos_sims, dim = 1)

        all_logits = sim_matrix.view(batch_size, -1)
        loss = -pos_logits + torch.logsumexp(all_logits, dim = 1)

        return loss.mean()