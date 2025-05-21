
import torch
from torch import nn
import itertools
from typing import List, Tuple


def compute_corr_over_T(x: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """
    Given x of shape (B, T, w), compute for each b a w×w Pearson matrix
    over the T timepoints.  Returns a tensor of shape (B, w, w).
    """
    B, T, w = x.shape
    # helper to do correlation on (T, w) → (w, w)
    def corr2d(tensor: torch.Tensor):
        # tensor: (T, w)
        mean = tensor.mean(dim=0, keepdim=True)         # (1, w)
        centered = tensor - mean                        # (T, w)
        denom = (T - 1) if unbiased else T
        cov = centered.T @ centered / denom             # (w, w)
        std = tensor.std(dim=0, unbiased=unbiased)      # (w,)
        return torch.clamp(cov / (std[:, None] * std[None, :] + 1e-8),
                           -1.0, 1.0)

    # stack per‑batch correlations
    return torch.stack([corr2d(x[b]) for b in range(B)], dim=0)  # (B, w, w)


class TimeOnlyBurstEMA:
    """
    EMA of the w×w corr matrix computed over T only.
    We only update pairs (k,l) when they exceed burst_threshold
    in at least min_batch_burst out of the B samples.
    """
    def __init__(
        self,
        w: int,
        momentum: float = 0.9,
        burst_threshold: float = 0.5,
        device=None,
    ):
        self.w = w
        self.m = momentum
        self.burst_threshold = burst_threshold
        self.device = device or torch.device('cpu')
        self.ema = torch.zeros((w, w), device=self.device)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """
        x: (B, T, w)
        1) Compute corr_batch: (B, w, w)
        2) For each b in B, if |corr_batch[b,k,l]|>th, update EMA_{k,l} with that sample's corr
        """
        B, T, w = x.shape
        assert w == self.w, "dimension mismatch"

        # 1) correlations over T → (B, w, w)
        corr_batch = compute_corr_over_T(x).to(x.device)
        if self.ema.device != x.device:
            self.ema = self.ema.to(x.device)
        # 2) for each sample b, update any (k,l) where that sample bursts
        for b in range(B):
            # boolean mask of size (w, w) for this sample
            burst_mask = corr_batch[b].abs() > self.burst_threshold
            # EMA update on just those entries, using corr_batch[b]
            self.ema[burst_mask] = (
                self.m * self.ema[burst_mask]
                + (1 - self.m) * corr_batch[b][burst_mask]
            )

    def get(self) -> torch.Tensor:
        """Return the current EMA correlation matrix."""
        return self.ema

def best_and_trim_group(
    ema: torch.Tensor,
    s: int,
    trim_threshold: float = 0.0,
    other_groups: Optional[List[List[int]]] = None,
    div_weight: float = 1.0
) -> Tuple[List[int], torch.Tensor]:
    """
    1. Find the subset G of indices of size exactly s that maximizes
       the mean intra-group absolute correlation while discouraging
       high cross-correlation with other_groups.
    2. Trim any member whose avg abs-corr to the rest of the group is below trim_threshold.

    Args:
        ema: (w, w) symmetric correlation matrix.
        s:   target group size.
        trim_threshold: threshold for trimming low-correlation members.
        other_groups: list of existing groups to diverge from.
        div_weight: weight for penalizing cross-group correlation.

    Returns:
        final_group: List of selected indices (<= s).
        submatrix:   ema submatrix of final_group.
    """
    w = ema.size(0)
    assert 1 <= s <= w, "s must be between 1 and w"

    C = ema.abs().clone()
    C.fill_diagonal_(0)

    best_group: List[int] = []
    best_score = -float('inf')
    iu = torch.triu_indices(s, s, offset=1)

    # Goal: maximize mean intra-group corr minus div_weight * max cross-group corr
    for group in itertools.combinations(range(w), s):
        sub = C[list(group)][:, list(group)]
        intra_mean = sub[iu[0], iu[1]].mean().item()
        cross_penalty = 0.0
        if other_groups:
            # compute mean cross-corr to each other_group, take max
            penalties = []
            for og in other_groups:
                if og:
                    cross_vals = C[list(group)][:, og].flatten()
                    penalties.append(cross_vals.mean().item())
            if penalties:
                cross_penalty = div_weight * max(penalties)
        score = intra_mean - cross_penalty
        if score > best_score:
            best_score = score
            best_group = list(group)

    # Trim low-correlation members
    final_group = best_group
    while True:
        avgs = torch.tensor([
            C[i, [j for j in final_group if j != i]].mean()
            for i in final_group
        ], device=ema.device)
        mask = avgs >= trim_threshold
        new_group = [i for i, keep in zip(final_group, mask) if keep]
        if len(new_group) == len(final_group):
            break
        final_group = new_group

    submatrix = ema[final_group][:, final_group]
    return final_group, submatrix

def group_similarity(
    ema: torch.Tensor,
    group1: List[int],
    group2: List[int]
) -> float:
    """
    Compute similarity in [0,1] between two groups based on
    average absolute correlation between their members.

    Args:
        ema: (w, w) symmetric correlation matrix.
        group1, group2: lists of indices.
    Returns:
        similarity: average |ema[i,j]| over i in group1, j in group2.
    """
    if not group1 or not group2:
        return 0.0
    C = ema.abs()
    vals = C[group1][:, group2].flatten()
    return float(vals.mean().item())

def update_group_members(
    ema: torch.Tensor,
    current_group: List[int],
    add_threshold: float,
    remove_threshold: float,
    max_size: int
) -> List[int]:
    """
    Prune and grow an existing group based on add/remove thresholds.

    Args:
        ema: (w, w) symmetric correlation matrix.
        current_group: List of current indices.
        add_threshold: min avg abs-corr to add new member.
        remove_threshold: min avg abs-corr to keep existing member.
        max_size: maximum group size.

    Returns:
        updated_group: new list of indices.
    """
    w = ema.size(0)
    C = ema.abs().clone()
    C.fill_diagonal_(0)

    # Remove low-correlation members
    updated = [
        i for i in current_group
        if not current_group or C[i, [j for j in current_group if j != i]].mean().item() >= remove_threshold
    ]

    # Add new members
    candidates = [k for k in range(w) if k not in updated]
    scores = []
    for k in candidates:
        score = C[k].mean().item() if not updated else C[k, updated].mean().item()
        if score >= add_threshold:
            scores.append((k, score))
    # add top scorers until max_size
    for k, _ in sorted(scores, key=lambda x: x[1], reverse=True):
        if len(updated) >= max_size:
            break
        updated.append(k)

    return updated

def update_multiple_groups(
    ema: torch.Tensor,
    groups: List[List[int]],
    add_threshold: float,
    remove_threshold: float,
    max_size: int
) -> List[List[int]]:
    """
    Apply `update_group_members` to a list of groups.

    Args:
        ema: (w, w) symmetric correlation matrix.
        groups: List of groups (each a list of indices).
        add_threshold: threshold for adding new members.
        remove_threshold: threshold for removing members.
        max_size: maximum size for each updated group.

    Returns:
        List of updated groups.
    """
    updated_groups = []
    for group in groups:
        new_group = update_group_members(
            ema,
            current_group=group,
            add_threshold=add_threshold,
            remove_threshold=remove_threshold,
            max_size=max_size
        )
        updated_groups.append(new_group)
    return updated_groups

class SelectiveGroupModel(nn.Module):
    def __init__(self, input_dim, num_groups=4, group_size=4):
        super().__init__()
        self.selector = TimeOnlyBurstEMA(
        w=input_dim,
        momentum = 0.9,
        burst_threshold = 0.5,
        min_batch_burst = 1)

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