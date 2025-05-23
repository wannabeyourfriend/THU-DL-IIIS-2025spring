import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        """
        y_hat: (B, T, vocab_size)
        y: (B, T)
        """
        # Convert y_hat to (B*T, vocab_size), y to (B*T)
        return F.cross_entropy(y_hat.view(-1, y_hat.size(-1)),
                               y.view(-1),
                               ignore_index=-1)


class DPOLoss(nn.Module):

    def __init__(self, kl_beta=0.02):
        super().__init__()
        self.kl_beta = kl_beta

    def forward(self, positive_log_probs, negative_log_probs, positive_log_probs_sft, negative_log_probs_sft):
        """
        positive_log_probs: (B, 1)
        negative_log_probs: (B, 1)
        positive_log_probs_sft: (B, 1)
        negative_log_probs_sft: (B, 1)
        """
        ############################ Your code here ############################
        # TODO: Implement the DPO loss
        kl_positive = positive_log_probs - positive_log_probs_sft
        kl_negative = negative_log_probs - negative_log_probs_sft
        
        # 计算偏好差异
        logits = kl_positive - kl_negative
        
        # 计算DPO损失
        loss = -F.logsigmoid(logits / self.kl_beta).mean()
        
        # 计算准确率（正样本得分高于负样本的比例）
        acc = (kl_positive > kl_negative).float().mean()
        
        return loss, acc
        ########################################################################
