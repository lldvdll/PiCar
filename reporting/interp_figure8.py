"""
interp_figure8.py — clean 3x2 figure: input, Grad-CAM, Occlusion for two frames.
Shows how attention shifts between consecutive frames during a misprediction.
"""
import os, glob
import cv2
import matplotlib.pyplot as plt

from interp import (model, preprocess, gradcam, occlusion, overlay,
                     parse_filename, RECORDINGS_DIR)

FRAME_INDICES = [38, 77]   # 0-indexed: frames 42 and 79
OUTPUT_NAME = "attention_shift.png"


def main():
    image_paths = sorted(glob.glob(os.path.join(RECORDINGS_DIR, "*.png")))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.02,
                         hspace=0.22, wspace=0.04)

    row_titles = ["Input", "Grad-CAM (angle head)", "Occlusion (angle head)"]

    for col, idx in enumerate(FRAME_INDICES):
        path = image_paths[idx]
        raw = cv2.imread(path)
        inp = preprocess(raw)
        inp_disp = (inp[0].numpy() * 255).astype('uint8')

        preds = model(inp, training=False)
        pa = float(preds[0][0, 0]) * 80.0 + 50.0
        _, ra, _ = parse_filename(path)

        cam_heat = gradcam(inp, head=0)
        # occ_heat = occlusion(inp, head=0)

        panels = [inp_disp, overlay(inp_disp, cam_heat)]#, overlay(inp_disp, occ_heat)]

        for row, panel in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"Frame {idx + 1}",#\nrecorded {ra}°,  predicted {pa:.1f}°",
                             fontsize=10)
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=10)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_NAME)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    print(f"[INFO] Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()