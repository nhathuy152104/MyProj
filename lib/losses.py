import torch
import torch.nn as nn
import torch.nn.functional as F

def format_global_protos_for_loss(global_protos_dict, num_classes, feature_dim, device):
    """
    Biến đổi Dictionary các sub-prototypes không đồng đều thành 3D Tensor.
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
                    # Lặp lại tâm cuối cùng để padding cho đủ max_protos
                    formatted_tensor[class_idx, i, :] = class_protos[-1].to(device)
                    
    return formatted_tensor

class MultiProtoSupConLoss(nn.Module):
    def __init__(self, num_classes=4, temperature=0.01):
        super().__init__()
        # Đưa num_classes ra khởi tạo để tái sử dụng dễ dàng cho dataset khác
        self.num_classes = num_classes 
        self.temp = temperature

    def forward(self, z, labels, global_prototypes):
        batch_size = z.size(0)
        
        if batch_size == 0:
            return torch.tensor(0.0, requires_grad=True, device=z.device)

        # 1. CHUYỂN ĐỔI DICT SANG TENSOR
        if isinstance(global_prototypes, dict):
            global_prototypes = format_global_protos_for_loss(
                global_prototypes, 
                num_classes=self.num_classes, 
                feature_dim=z.size(1), 
                device=z.device
            )
            
        # Nếu chưa có prototype (ví dụ vòng đầu tiên), trả về Loss 0
        if not isinstance(global_prototypes, torch.Tensor) or global_prototypes.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=z.device)

        global_prototypes = global_prototypes.to(z.device)
        M = global_prototypes.size(1) # Số sub-prototypes lớn nhất mỗi class

        # 2. TÍNH COSINE SIMILARITY
        # sim_matrix: [batch_size, num_classes, M]
        sim_matrix = F.cosine_similarity(
            z.unsqueeze(1).unsqueeze(1), 
            global_prototypes.unsqueeze(0),
            dim=-1
        )
        
        # Scale với temperature cho Logits
        sim_matrix_scaled = sim_matrix / self.temp
        all_logits = sim_matrix_scaled.view(batch_size, -1)
        log_probs = F.log_softmax(all_logits, dim=1)

        # 3. TẠO ADAPTIVE SOFT TARGETS
        soft_targets = torch.zeros_like(log_probs)
        
        # Trích xuất độ tương đồng của điểm ảnh hiện tại với M tâm của RIÊNG class đúng
        # Kích thước pos_sims: [batch_size, M]
        pos_sims = sim_matrix_scaled[torch.arange(batch_size), labels] 
        
        # Dùng Softmax để tạo trọng số phân bổ. 
        # Chú ý: Cần dùng .detach() để không truyền gradient ngược qua nhánh tạo label
        soft_weights = F.softmax(pos_sims.detach(), dim=1)
        
        # Tính toán index của class đúng trên ma trận đã duỗi (2D)
        label_indices = labels.unsqueeze(1) * M + torch.arange(M, device=z.device)
        
        # Rải trọng số thích ứng vào mục tiêu
        soft_targets.scatter_(1, label_indices, soft_weights)

        # 4. TÍNH KL-DIVERGENCE / CROSS ENTROPY
        loss = torch.sum(-soft_targets * log_probs, dim=1)

        return loss.mean()