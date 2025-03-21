import torch


class ContinuousVowelBoundaryLoss(torch.nn.Module):
    def __init__(self, vowel_vocab, margin=0.3):
        super().__init__()
        self.register_buffer('vowel_ids_tensor', torch.tensor(list(vowel_vocab.values()), dtype=torch.long))
        self.margin = margin
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, frame_features, boundaries, ph_labels):
        """
        frame_features: (B, T, D) - 特征张量
        boundaries: (B, T) - 边界概率
        ph_labels: (B, T) - 音素标签
        """
        # 使用torch.isin在GPU上执行
        is_vowel = torch.isin(ph_labels, self.vowel_ids_tensor)  # (B, T)
        next_is_vowel = torch.cat([is_vowel[:, 1:], torch.zeros_like(is_vowel[:, :1])], dim=1)  # (B, T)

        # 边界掩码
        boundary_mask = (boundaries > 0.1) & is_vowel & next_is_vowel  # (B, T)

        # 如果没有边界，返回伪损失
        if boundary_mask.sum() == 0:
            return torch.tensor(1e-6, device=frame_features.device)

        # 获取边界位置的索引
        batch_indices, time_indices = torch.where(boundary_mask)  # (N,), (N,)

        # 负采样：从非边界位置随机采样
        non_boundary = torch.where(~boundary_mask)  # (N_non_boundary,)
        if len(non_boundary[0]) == 0:
            return torch.tensor(1e-6, device=frame_features.device)

        # 随机采样负样本
        rand_idx = torch.randint(0, len(non_boundary[0]), (len(batch_indices),), device=frame_features.device)
        negatives = frame_features[non_boundary[0][rand_idx], non_boundary[1][rand_idx]]  # (N, D)

        # 获取前一个时间步的特征作为正样本
        prev_time = torch.clamp(time_indices - 1, min=0)  # (N,)
        positives = frame_features[batch_indices, prev_time]  # (N, D)

        # 获取当前时间步的特征作为锚点
        anchors = frame_features[batch_indices, time_indices]  # (N, D)

        # 计算Triplet Loss
        return self.loss_fn(anchors, positives, negatives)
