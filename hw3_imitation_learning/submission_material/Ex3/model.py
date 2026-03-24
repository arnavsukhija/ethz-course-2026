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
    def __init__(self, state_dim: int = 19, action_dim: int = 4, chunk_size: int = 16, d_model: int = 256, depth: int = 3, dropout: float = 0.2, **kwargs):
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
    """Goal-conditioned policy with explicit target cube routing."""
    def __init__(self, state_dim: int = 19, action_dim: int = 4, chunk_size: int = 32, d_model: int = 256, depth: int = 3, dropout: float = 0.3, **kwargs):
        super().__init__(state_dim, action_dim, chunk_size)

        in_dim = state_dim
        if state_dim == 19:
            in_dim = state_dim + 9 # we add the cube's target pos (3), relative cube pos (3), relative bin pos (3)
        elif state_dim == 31:
            in_dim = state_dim + 7  # same but with the full dimensions            
        layers = []
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
        if state.shape[1] == 19:
            # ee(3), gripper(1), red(3), green(3), blue(3), goal(3), bin(3)
            ee = state[:, 0:3]
            red = state[:, 4:7]
            green = state[:, 7:10]
            blue = state[:, 10:13]
            goal = state[:, 13:16]
            bin_pos = state[:, 16:19]
            
            target_cube = goal[:, 0:1] * red + goal[:, 1:2] * green + goal[:, 2:3] * blue
            rel_cube = target_cube - ee
            rel_bin = bin_pos - ee
            
            enhanced_state = torch.cat([state, target_cube, rel_cube, rel_bin], dim=1)
        elif state.shape[1] == 31:
            # Assumes order: red(7), green(7), blue(7), goal(3), bin(3), ee(3), gripper_jaw(1)
            red = state[:, 0:7]
            green = state[:, 7:14]
            blue = state[:, 14:21]
            goal = state[:, 21:24]
            
            target_cube = goal[:, 0:1] * red + goal[:, 1:2] * green + goal[:, 2:3] * blue
            enhanced_state = torch.cat([state, target_cube], dim=1)
        else:
            enhanced_state = state

        h = self.network(enhanced_state)
        actions = self.action_head(h)
        return actions.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
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
