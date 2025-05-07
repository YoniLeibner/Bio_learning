import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrelatedGroupSelector(nn.Module):
    def __init__(self, input_dim, num_groups, group_size, temperature=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.group_size = group_size
        self.temperature = temperature

        # Learnable logits for group membership: [num_groups, input_dim]
        self.group_logits = nn.Parameter(torch.randn(num_groups, input_dim))

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        Returns:
            grouped_inputs: list of tensors [(batch, group_size), ...]
            selection_mask: (num_groups, input_dim)
        """
        # Gumbel-Softmax over inputs to create a soft group selection
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.group_logits)))
        logits = self.group_logits + gumbel_noise
        probs = F.softmax(logits / self.temperature, dim=-1)

        # Use top-k per group to select group members
        topk = torch.topk(probs, self.group_size, dim=-1)
        selection_mask = torch.zeros_like(probs)
        selection_mask.scatter_(1, topk.indices, 1.0)  # hard selection mask

        grouped_inputs = []
        for i in range(self.num_groups):
            group = selection_mask[i] * x  # broadcasted (batch, input_dim)
            grouped_inputs.append(group)

        return grouped_inputs, selection_mask

def compute_group_correlation(group):
    """
    group: (batch_size, input_dim) where non-grouped values are 0
    Returns: scalar correlation score (avg pairwise cosine)
    """
    # Only non-zero cols
    nonzero = (group.abs().sum(0) > 0)
    group_vars = group[:, nonzero]
    if group_vars.shape[1] < 2:
        return torch.tensor(0.0, device=group.device)
    normed = F.normalize(group_vars, dim=0)
    corr = (normed.T @ normed) / normed.shape[0]
    upper = torch.triu(corr, diagonal=1)
    avg_corr = upper.sum() / (nonzero.sum() * (nonzero.sum() - 1) / 2 + 1e-6)
    return avg_corr

class SelectiveGroupModel(nn.Module):
    def __init__(self, input_dim, num_groups=4, group_size=4):
        super().__init__()
        self.selector = CorrelatedGroupSelector(input_dim, num_groups, group_size)
        self.group_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ) for _ in range(num_groups)
        ])

    def forward(self, x):
        groups, mask = self.selector(x)

        # Compute correlation per group
        correlations = torch.stack([compute_group_correlation(g) for g in groups])
        _, selected_indices = torch.topk(-correlations, k=2)  # lowest 2 correlations

        # Only update selected groups
        outputs = []

        for i, group in enumerate(groups):
            if i in selected_indices:
                out = self.group_mlp[i](group)
            else:
                with torch.no_grad():
                    out = self.group_mlp[i](group)
            outputs.append(out)

        out = torch.cat(outputs, dim=1)
        return out, correlations