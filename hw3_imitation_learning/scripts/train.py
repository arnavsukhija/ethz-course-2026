"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
    _parse_key_spec,
)
from hw3.model import BasePolicy, build_policy

# TODO: Any imports you want from torch or other libraries we use. Not allowed: libraries we don't use
from torch.utils.data import DataLoader, random_split

# Choose your own hyperparameters!
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 5e-4
VAL_SPLIT = 0.2


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    normalizer=None,
    key_to_slice=None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Pre-calculate mean/std tensors on the target device only once per epoch
    mean, std = None, None
    if normalizer is not None:
        mean = torch.tensor(normalizer.state_mean, device=device, dtype=torch.float32)
        std = torch.tensor(normalizer.state_std, device=device, dtype=torch.float32)

    for batch in loader:
        states, action_chunks = batch
        # TODO: Implement the training step for one batch here.
        # This mostly: Get states and action_chunks onto the correct device, compute the loss, and step the optimizer.
        # we also perform data augmentation here for each batch
        states, action_chunks = states.to(device).float(), action_chunks.to(device).float()

        if normalizer is not None and key_to_slice is not None:
            B = states.shape[0]
            # Unnormalize state to perform augmentations
            states_unnorm = states * std + mean
            
            # 1. Translation Augmentation
            dx = (torch.rand((B, 1), device=device) - 0.5) * 0.1
            dy = (torch.rand((B, 1), device=device) - 0.5) * 0.1
            for key in ["state_ee_xyz", "original_pos_cube_red", "original_pos_cube_green", "original_pos_cube_blue", "goal_pos"]:
                if key in key_to_slice:
                    sl = key_to_slice[key]
                    if sl.stop - sl.start >= 2:
                        states_unnorm[:, sl.start:sl.start+1] += dx
                        states_unnorm[:, sl.start+1:sl.start+2] += dy

            # 2. Goal Relabeling / Permutation Augmentation
            has_cubes = all(k in key_to_slice for k in ["original_pos_cube_red", "original_pos_cube_green", "original_pos_cube_blue", "state_goal"])
            if has_cubes:
                for i in range(B):
                    # random permutation of [0, 1, 2]
                    perm = torch.randperm(3, device=device)
                    r_sl, g_sl, b_sl = key_to_slice["original_pos_cube_red"], key_to_slice["original_pos_cube_green"], key_to_slice["original_pos_cube_blue"]
                    c_slices = [r_sl, g_sl, b_sl]
                    vals = [states_unnorm[i, sl].clone() for sl in c_slices]
                    
                    for j, p in enumerate(perm):
                        states_unnorm[i, c_slices[j]] = vals[p]
                    
                    goal_sl = key_to_slice["state_goal"]
                    orig_goal = states_unnorm[i, goal_sl].clone()
                    
                    target_idx = torch.argmax(orig_goal)
                    # Find which new cube received the position of the old target
                    match_idx = torch.where(perm == target_idx)[0]
                    if len(match_idx) > 0:
                        new_target_idx = match_idx[0]
                        new_goal = torch.zeros_like(orig_goal)
                        new_goal[new_target_idx] = 1.0
                        states_unnorm[i, goal_sl] = new_goal

            # Renormalize
            states = (states_unnorm - mean) / std
        
        optimizer.zero_grad()
        loss = model.compute_loss(states, action_chunks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states, action_chunks = states.to(device).float(), action_chunks.to(device).float()
        loss = model.compute_loss(states, action_chunks)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    # You may add any cli arguments that make life easier for you like learning rate etc.
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--extra-zarr",
        type=Path,
        nargs="+",
        help="Path(s) to additional processed .zarr stores to merge (e.g. for DAgger).",
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate (default: 0.15).")
    parser.add_argument("--d-model", type=int, default=256, help="Hidden dimension (default: 256).")
    parser.add_argument("--depth", type=int, default=3, help="Number of layers (default: 3).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu). Auto-detected if None.")
    parser.add_argument("--no-padding", action="store_true", help="Disable padding for episode endings (default: padding enabled).")
    parser.add_argument("--augment-multicube", action="store_true", help="Enable PyTorch data augmentation for multicube.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
        use_padding=not args.no_padding,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}, padding={'enabled' if not args.no_padding else 'disabled'}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")

    def parse_key_spec_local(spec: str):
        if "[" not in spec: return spec, slice(None)
        name, rest = spec.split("[", 1)
        parts = rest.rstrip("]").split(":")
        start = int(parts[0]) if len(parts) == 2 and parts[0] else None
        stop = int(parts[1]) if len(parts) == 2 and parts[1] else None
        return name, slice(start, stop)

    key_to_slice = None
    if args.augment_multicube and args.state_keys:
        root = zarr_lib.open_group(str(zarr_paths[0]), mode="r")
        state_dims = []
        for spec in args.state_keys:
            name, col_slice = parse_key_spec_local(spec)
            arr = root["data"][name][:1]
            sliced = arr[:, col_slice] if col_slice != slice(None) else arr
            state_dims.append(sliced.shape[1])
        
        key_to_slice = {}
        curr = 0
        for spec, dim in zip(args.state_keys, state_dims):
            name = parse_key_spec_local(spec)[0]
            key_to_slice[name] = slice(curr, curr + dim)
            curr += dim
        print(f"  Enabled Multicube Data Augmentation with slices: {key_to_slice}")

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # implement an optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    save_name = f"best_model_{action_space}_{args.policy}.pt"

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        save_name = f"best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt"
    # Default: checkpoints/<task>/
    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    save_path = ckpt_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            normalizer if args.augment_multicube else None,
            key_to_slice if args.augment_multicube else None,
        )
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalizer": {
                        "state_mean": normalizer.state_mean,
                        "state_std": normalizer.state_std,
                        "action_mean": normalizer.action_mean,
                        "action_std": normalizer.action_std,
                    },
                    "chunk_size": args.chunk_size,
                    "policy_type": args.policy,
                    "state_keys": args.state_keys,
                    "action_keys": args.action_keys,
                    "state_dim": int(states.shape[1]),
                    "action_dim": int(actions.shape[1]),
                    "d_model": args.d_model,
                    "depth": args.depth,
                    "val_loss": val_loss,
                },
                save_path,
            )
            tag = " ✓ saved"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
