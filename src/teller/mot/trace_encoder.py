"""
TracePairEncoder: GCN over call graph (parent_idx) + depth/duration encoding; output (B, L_trace, d).
Uses PyTorch Geometric for batching and GCN. Falls back to mean-pool when parent_idx/duration/depth are not provided.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    _HAS_TORCH_GEOMETRIC = False
    Batch = None
    Data = None
    GCNConv = None
    global_mean_pool = None


def _build_edge_index_from_parent_idx(
    parent_idx: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """
    parent_idx: (L,) int, parent_idx[i] = parent of node i, -1 for root.
    valid_mask: (L,) 1=valid, 0=pad.
    Returns: (edge_index (2, E) in 0..num_valid-1, num_valid).
    """
    L = parent_idx.shape[0]
    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    num_valid = valid_idx.numel()
    if num_valid == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device), 0

    global_to_local = torch.full((L,), -1, dtype=torch.long, device=device)
    global_to_local[valid_idx] = torch.arange(num_valid, device=device)

    edges: list[tuple[int, int]] = []
    for i in range(L):
        if not valid_mask[i]:
            continue
        p = parent_idx[i].item()
        if p >= 0 and valid_mask[p]:
            edges.append((global_to_local[p].item(), global_to_local[i].item()))

    if not edges:
        return torch.zeros(2, 0, dtype=torch.long, device=device), num_valid
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index, num_valid


class TracePairEncoder(nn.Module):
    """
    Encode trace steps: TPE token ids + call graph (parent_idx) + depth/duration.
    - Token embedding + depth encoder + duration encoder -> node features.
    - PyG GCN over graph built from parent_idx (parent -> child edges).
    - Readout: global_mean_pool per step -> (B, L_trace, d).
    When step_parent_idx / step_duration / step_depth are None, falls back to mean-pool over tokens.
    Requires: torch-geometric (optional dependency).
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_hidden_size: int,
        d: int,
        pad_token_id: int = 0,
        num_gcn_layers: int = 2,
        max_depth: int = 64,
        duration_log_scale: float = 1e-9,
    ):
        super().__init__()
        if not _HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "TracePairEncoder with GCN requires torch-geometric. "
                "Install with: pip install torch-geometric"
            )
        self.pad_token_id = pad_token_id
        self.encoder_hidden_size = encoder_hidden_size
        self.num_gcn_layers = num_gcn_layers
        self.duration_log_scale = duration_log_scale

        self.embed = nn.Embedding(vocab_size, encoder_hidden_size, padding_idx=pad_token_id)
        self.depth_encoder = nn.Sequential(
            nn.Linear(1, encoder_hidden_size),
            nn.LayerNorm(encoder_hidden_size),
            nn.GELU(),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
        )
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, encoder_hidden_size),
            nn.LayerNorm(encoder_hidden_size),
            nn.GELU(),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
        )
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(encoder_hidden_size, encoder_hidden_size, add_self_loops=True))
        self.proj = nn.Linear(encoder_hidden_size, d)

    def _encode_depth_duration(
        self,
        step_depth: torch.Tensor,
        step_duration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """step_depth (B, L_trace, L_tok), step_duration (B, L_trace, L_tok). Return (B,L_trace,L_tok,H), (B,L_trace,L_tok,H)."""
        depth = step_depth.clamp(min=0).float().unsqueeze(-1)
        duration = step_duration.float().clamp(min=0)
        duration = torch.log1p(duration * self.duration_log_scale).unsqueeze(-1)
        depth_enc = self.depth_encoder(depth)
        duration_enc = self.duration_encoder(duration)
        return depth_enc, duration_enc

    def _build_pyg_batch(
        self,
        node_feat: torch.Tensor,
        step_token_mask: torch.Tensor,
        step_parent_idx: torch.Tensor,
        device: torch.device,
    ) -> tuple[Batch | None, list[tuple[int, int]]]:
        """
        node_feat: (B, L_trace, L_tok, H)
        Returns: (Batch of Data, list of (b, s) for each graph in order) or (None, []) if no valid graphs.
        """
        B, L_trace, L_tok, H = node_feat.shape
        data_list: list[Data] = []
        indices: list[tuple[int, int]] = []

        for b in range(B):
            for s in range(L_trace):
                valid = step_token_mask[b, s]
                num_valid = (valid > 0.5).sum().item()
                if num_valid == 0:
                    continue
                parent = step_parent_idx[b, s].long()
                edge_index, _ = _build_edge_index_from_parent_idx(parent, (valid > 0.5), device)
                x = node_feat[b, s][:num_valid]
                data_list.append(Data(x=x, edge_index=edge_index))
                indices.append((b, s))

        if not data_list:
            return None, []
        batch = Batch.from_data_list(data_list)
        return batch, indices

    def forward(
        self,
        step_token_ids: torch.Tensor,
        step_token_mask: torch.Tensor | None = None,
        step_parent_idx: torch.Tensor | None = None,
        step_duration: torch.Tensor | None = None,
        step_depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        step_token_ids: (B, L_trace, L_tok) int
        step_token_mask: (B, L_trace, L_tok) 1=valid, 0=pad
        step_parent_idx: (B, L_trace, L_tok) int, -1 for root/pad
        step_duration: (B, L_trace, L_tok) float
        step_depth: (B, L_trace, L_tok) int
        Returns: (B, L_trace, d)
        """
        B, L_trace, L_tok = step_token_ids.shape
        device = step_token_ids.device
        if step_token_mask is None:
            step_token_mask = (step_token_ids != self.pad_token_id).float()

        token_emb = self.embed(step_token_ids)
        if step_depth is not None and step_duration is not None:
            depth_enc, duration_enc = self._encode_depth_duration(step_depth, step_duration)
            node_feat = token_emb + depth_enc + duration_enc
        else:
            node_feat = token_emb

        use_gcn = step_parent_idx is not None and self.num_gcn_layers > 0

        if not use_gcn:
            mask = step_token_mask.unsqueeze(-1)
            x = (node_feat * mask).sum(dim=2) / (mask.sum(dim=2).clamp(min=1e-9))
            return self.proj(x)

        pyg_batch, indices = self._build_pyg_batch(
            node_feat, step_token_mask, step_parent_idx, device
        )

        out = torch.zeros(B, L_trace, self.encoder_hidden_size, device=device, dtype=node_feat.dtype)
        if pyg_batch is not None:
            x = pyg_batch.x
            for gcn in self.gcn_layers:
                x = gcn(x, pyg_batch.edge_index)
                x = F.relu(x)
            graph_vecs = global_mean_pool(x, pyg_batch.batch)
            for i, (b, s) in enumerate(indices):
                out[b, s] = graph_vecs[i]

        return self.proj(out)
