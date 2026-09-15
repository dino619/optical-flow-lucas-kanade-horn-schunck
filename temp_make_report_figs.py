from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ex1_utils import rotate_image, show_flow
from of_methods import iterative_lucaskanade, hornschunck, lucaskanade, pyramidal_lucaskanade


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "report_temp_figs"
OUT_DIR.mkdir(exist_ok=True)


def load_gray(path):
    # Vse vhodne slike pretvorimo v sivinsko obliko in normaliziramo na [0,1],
    # da je prikaz in izracun med primeri enoten.
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img.astype(np.float32) / 255.0


def make_synthetic_pair(seed=0, size=200, angle=-1):
    # Synthetic par je kontroliran primer za sanity check.
    np.random.seed(seed)
    im1 = np.random.rand(size, size).astype(np.float32)
    im2 = rotate_image(im1, angle)
    return im1, im2


def make_large_motion_pair(seed=7, size=220, dx=14, dy=10):
    # Ta par vsebuje vecji translacijski premik za test dodatnega dela (pyramidal LK).
    np.random.seed(seed)
    im1 = np.random.rand(size, size).astype(np.float32)
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    im2 = cv2.warpAffine(
        im1,
        M,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return im1, im2


def save_inputs_figure(im1, im2, out_name):
    # Posebej shranimo vhodni par, da ga lahko v porocilu postavimo brez stiskanja.
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.8))
    axs[0].imshow(im1, cmap="gray")
    axs[0].set_title("Image 1")
    axs[1].imshow(im2, cmap="gray")
    axs[1].set_title("Image 2")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_flow_figure(im1, im2, out_name, lk_n=3, hs_iters=1000, hs_lambda=0.5):
    # V eni sliki primerjamo LK in HS, da je razlika med metodama takoj vidna.
    u_lk, v_lk = lucaskanade(im1, im2, N=lk_n)
    u_hs, v_hs = hornschunck(im1, im2, n_iters=hs_iters, lmbd=hs_lambda)

    fig, axs = plt.subplots(2, 2, figsize=(8.2, 7.2))
    show_flow(u_lk, v_lk, axs[0, 0], type="angle")
    axs[0, 0].set_title("LK Flow (Angle)")
    show_flow(u_lk, v_lk, axs[0, 1], type="field", set_aspect=True)
    axs[0, 1].set_title("LK Flow (Field)")
    show_flow(u_hs, v_hs, axs[1, 0], type="angle")
    axs[1, 0].set_title("HS Flow (Angle)")
    show_flow(u_hs, v_hs, axs[1, 1], type="field", set_aspect=True)
    axs[1, 1].set_title("HS Flow (Field)")
    for ax in axs.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pyramidal_flow_figure(im1, im2, out_name):
    # Dodatni del: neposredna primerjava navadnega LK, iterativnega LK in pyramidalnega LK.
    u_lk, v_lk = lucaskanade(im1, im2, N=5)
    u_iter, v_iter = iterative_lucaskanade(im1, im2, N=5, n_iters=5)
    u_pyr, v_pyr = pyramidal_lucaskanade(im1, im2, N=5, n_levels=4, n_iters=3)

    fig, axs = plt.subplots(3, 2, figsize=(8.2, 10.0))
    show_flow(u_lk, v_lk, axs[0, 0], type="angle")
    axs[0, 0].set_title("LK (Angle)")
    show_flow(u_lk, v_lk, axs[0, 1], type="field", set_aspect=True)
    axs[0, 1].set_title("LK (Field)")

    show_flow(u_iter, v_iter, axs[1, 0], type="angle")
    axs[1, 0].set_title("Iterative LK (Angle)")
    show_flow(u_iter, v_iter, axs[1, 1], type="field", set_aspect=True)
    axs[1, 1].set_title("Iterative LK (Field)")

    show_flow(u_pyr, v_pyr, axs[2, 0], type="angle")
    axs[2, 0].set_title("Pyramidal LK (Angle)")
    show_flow(u_pyr, v_pyr, axs[2, 1], type="field", set_aspect=True)
    axs[2, 1].set_title("Pyramidal LK (Field)")

    for ax in axs.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


# Figure 1: synthetic primer (vhod + LK/HS rezultat).
im1_syn, im2_syn = make_synthetic_pair()
save_inputs_figure(
    im1_syn,
    im2_syn,
    out_name="figure1_inputs.png",
)
save_flow_figure(
    im1_syn,
    im2_syn,
    out_name="figure1_flows.png",
    lk_n=3,
    hs_iters=1000,
    hs_lambda=0.5,
)

# Figure 4: cporta primer (vhod + LK/HS rezultat).
im1_c = load_gray(ROOT / "disparity" / "cporta_left.png")
im2_c = load_gray(ROOT / "disparity" / "cporta_right.png")
save_inputs_figure(
    im1_c,
    im2_c,
    out_name="figure4_inputs.png",
)
save_flow_figure(
    im1_c,
    im2_c,
    out_name="figure4_flows.png",
    lk_n=3,
    hs_iters=1000,
    hs_lambda=0.5,
)

# Dodatni del: pyramidal LK na paru z vecjim premikom.
im1_l, im2_l = make_large_motion_pair()
save_inputs_figure(
    im1_l,
    im2_l,
    out_name="figure_pyr_inputs.png",
)
save_pyramidal_flow_figure(
    im1_l,
    im2_l,
    out_name="figure_pyr_flows.png",
)

print(f"Saved temporary report figures in: {OUT_DIR}")
