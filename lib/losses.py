import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiProtoSupConLoss(nn.Module):
    def __init__(self, temperature = 0.97):
        super().__init__()
        self.temp = temperature

    def forward(self, z , labels, global_prototypes):

        batch_size = z.size(0)

        sim_matrix = F.cosine_similarity(z.unsqueeze(1).unsqueeze(1), 
                                         global_prototypes.unsqueeze(0),
                                         dim = -1)
        sim_matrix = sim_matrix / self.temp

        pos_sims = sim_matrix[torch.arange(batch_size), labels]

        pos_logits, _ = torch.max(pos_sims, dim = 1)


        all_logits = sim_matrix.view(batch_size, -1)

        loss = -pos_logits + torch.logsumexp(all_logits, dim = 1)

        return loss.mean()