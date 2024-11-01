from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
from boundvor import BoundedVoronoi
from tqdm import tqdm
import ffmpeg

matplotlib.use("Agg")


# Get the most recent folder in the path
folder_path = Path("runs/")
sorted_dirs = sorted(folder_path.iterdir(), key=lambda x: x.stat().st_ctime)
if sorted_dirs[-1] != Path("runs/.DS_Store"):
    file_dir = sorted_dirs[-1]
else:
    file_dir = sorted_dirs[-2]

print(f'Visualizing {file_dir}')
arrays_dir = file_dir / "arrays"
output_dir = file_dir / "frames"
output_dir.mkdir(exist_ok=True)

files = list(arrays_dir.glob("*.npz"))

def plot_leaf(file):
    data = np.load(file)

    sources = data["sources"]
    X = data["X"]
    C = data["C"]
    edges = data["edges"]
    boundary = data["boundary"]
    power = data["power"]
    step = data["step"]
    N = len(X)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=200)
    fig.set_facecolor("#f4f0e8")
    leaf = patches.Polygon(boundary, closed=True, edgecolor="#383b3e", facecolor="#f4f0e8", lw=2)
    ax.add_patch(leaf)

    x = np.concatenate((sources, X))
    c = np.asarray(C)
    for e, (v1, v2) in enumerate(edges):
        lw = 10 * np.sqrt(c[e]) / np.max(np.sqrt(c))
        ax.plot([x[v1, 0], x[v2, 0]], [x[v1, 1], x[v2, 1]], lw=lw, c="#383b3e")
    ax.plot(x[:len(sources),0], x[:len(sources),1], "o", color="#383b3e", ms=2)
    ax.plot(x[len(sources):, 0], x[len(sources):, 1], "o", color="#383b3e", ms=2)

    vor = BoundedVoronoi(X, boundary)
    for _, region in enumerate(vor.regions):
        ax.add_patch(patches.Polygon(vor.vertices[region], closed=True, edgecolor="#383b3e", fill=False, lw=0.5, alpha=0.4, zorder=2))


    ax.text(0.95, 0.05, f"Step {step}", fontsize=8, color="gray", ha="right", transform=ax.transAxes)
    ax.text(0.95, 0.09, f"N = {N}", fontsize=8, color="gray", ha="right", transform=ax.transAxes)
    ax.text(0.95, 0.01, f"Power = {power:.6g}", fontsize=8, color="gray", ha="right", transform=ax.transAxes)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.01)
    plt.tight_layout()
    fig.savefig(output_dir / f"frames_{step:08d}.png", dpi=120, pad_inches=0.4, bbox_inches="tight")
    plt.close()


def main():
    with ProcessPoolExecutor() as executor:
        tasks = [executor.submit(plot_leaf, file) for file in files]
        for f in tqdm(as_completed(tasks), total=len(tasks), ncols=81):
            pass

    # Make a video (duration to be 10 seconds)
    num_frames = len(list(output_dir.glob("*.png")))
    fps = int(num_frames // 10)
    ffmpeg.input(
        str(output_dir / "*.png"), pattern_type='glob', framerate=fps
    ).output(
        str(file_dir / "animation.mp4"), pix_fmt='yuv420p',
        vf='scale=trunc(iw/2)*2:trunc(ih/2)*2'
    ).run(quiet=True, overwrite_output=True)


if __name__ == '__main__':
    main()
