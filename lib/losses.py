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
    def __init__(self, temperature = 0.01):
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
        # 2. XỬ LÝ VÒNG 1 (Nếu chưa có dữ liệu, trả về Loss = 0)
        if not isinstance(global_prototypes, torch.Tensor) or global_prototypes.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=z.device)

        global_prototypes = global_prototypes.to(z.device)

        M = global_prototypes.size(1) # Kích thước M (Số lượng sub-prototypes mỗi class)

        # 1. Tính Cosine Similarity
        # sim_matrix: [batch_size, num_classes, M]
        sim_matrix = F.cosine_similarity(
            z.unsqueeze(1).unsqueeze(1), 
            global_prototypes.unsqueeze(0),
            dim = -1
        )
        sim_matrix = sim_matrix / self.temp
        
        # Đưa về dạng ma trận 2D cho toàn bộ logit: [batch_size, num_classes * M]
        all_logits = sim_matrix.view(batch_size, -1)
        
        # 2. Tính Log-Softmax của logits (tương đương với vế -log(p) trong công thức)
        log_probs = F.log_softmax(all_logits, dim=1)

        # 3. Tạo Soft Targets (Mục tiêu mềm) theo chuẩn MedCLIP
        soft_targets = torch.zeros_like(log_probs)
        
        # Tìm index của tất cả M sub-prototypes thuộc class đúng của từng sample
        # Ví dụ: M=3, nhãn=1 => index = [3, 4, 5]
        label_indices = labels.unsqueeze(1) * M + torch.arange(M, device=z.device)
        
        # Rải đều trọng số (1/M) cho tất cả các tâm thuộc cùng class
        # Điều này loại bỏ hoàn toàn False Negatives trong nội bộ class
        soft_targets.scatter_(1, label_indices, 1.0 / M)
        
        # (Tuỳ chọn) Nếu bạn muốn Soft-Target Tự thích ứng dựa trên khoảng cách hiện tại:
        # Thay vì chia đều 1/M, ta lấy chính Softmax của độ đo hiện tại (như MedCLIP dùng Text-Sim)
        # pos_sims = sim_matrix[torch.arange(batch_size), labels] # [batch_size, M]
        # soft_weights = F.softmax(pos_sims.detach(), dim=1)
        # soft_targets.scatter_(1, label_indices, soft_weights)

        # 4. Tính Cross Entropy Loss với phân phối mềm (KL-Divergence)
        # Công thức: Loss = Trung bình ( Tổng ( - Target_i * Log_Prob_i ) )
        loss = torch.sum(-soft_targets * log_probs, dim=1)

        return loss.mean()