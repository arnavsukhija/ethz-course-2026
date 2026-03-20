"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, d_model: int = 256, depth: int = 3, dropout: float = 0.2):
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        in_dim = state_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Mish(),
                nn.Dropout(dropout),
            ])
            in_dim = d_model
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(d_model, chunk_size * action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        h = self.backbone(state)
        actions = self.action_head(h)
        return actions.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        # BC Loss: Mean Squared Error between predicted and expert action chunks
        predicted_chunk = self.forward(state)
        return F.mse_loss(predicted_chunk, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Inference mode: returns the predicted action chunk."""
        self.eval()
        with torch.no_grad():
            return self.forward(state)


# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, d_model: int = 256, depth: int = 3, dropout: float = 0.2):
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        in_dim = state_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Mish(),
                nn.Dropout(dropout),
            ])
            in_dim = d_model
        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(d_model, chunk_size * action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        h = self.network(state)
        actions = self.action_head(h)
        return actions.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        # MSE loss for goal-conditioned BC
        return F.mse_loss(self.forward(state), action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model: int = 256,
    depth: int = 3,
    **kwargs,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            **kwargs,
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            **kwargs,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
