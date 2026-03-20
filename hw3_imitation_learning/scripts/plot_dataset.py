import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import zarr

def plot_zarr(zarr_path: Path, num_episodes: int = 5):
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]
    
    ep_ends = np.asarray(meta["episode_ends"])
    starts = np.concatenate([[0], ep_ends[:-1]])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Identify position key
    pos_key = next((k for k in data.keys() if "ee" in k and "xyz" in k), None)
    if not pos_key and "state_ee" in data: pos_key = "state_ee"
    
    grip_key = "action_gripper" if "action_gripper" in data else "state_gripper"

    for i in range(min(len(starts), num_episodes)):
        s, e = starts[i], ep_ends[i]
        
        if pos_key:
            pos = data[pos_key][s:e]
            axes[0].plot(pos[:, 0], label=f"Ep {i} X", alpha=0.7)
            axes[0].plot(pos[:, 1], label=f"Ep {i} Y", alpha=0.7, linestyle='--')
            axes[0].plot(pos[:, 2], label=f"Ep {i} Z", alpha=0.7, linestyle=':')
        
        if grip_key:
            grip = data[grip_key][s:e]
            axes[1].plot(grip, label=f"Ep {i} Grip", alpha=0.7)

    axes[0].set_title(f"EE Trajectories ({pos_key})")
    axes[0].set_ylabel("Position (m)")
    axes[0].legend(ncol=3, fontsize='small')
    
    axes[1].set_title(f"Gripper {grip_key}")
    axes[1].set_ylabel("Value")
    axes[1].set_xlabel("Timestep")
    
    plt.tight_layout()
    out_img = zarr_path.parent / f"{zarr_path.stem}_plot.png"
    plt.savefig(out_img)
    print(f"Plot saved to: {out_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr", type=Path, help="Path to .zarr file")
    parser.add_argument("--num-eps", type=int, default=5, help="Number of episodes to plot")
    args = parser.parse_args()
    plot_zarr(args.zarr, args.num_eps)
