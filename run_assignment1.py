from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ex1_utils import gaussderiv, rotate_image, show_flow
from of_methods import hornschunck, iterative_lucaskanade, lucaskanade, pyramidal_lucaskanade


# Uporabimo osnovno pot do mape s skripto,
# da pač ni vazno, od kod program pozenemo
WORKSPACE_DIR = Path(__file__).resolve().parent


def load_gray_image(path):
    # Slike beremo v sivinah, ker obe metodi delata nad enim kanalom
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    # Sliko pretvorimo v float in normaliziramo na [0, 1],
    # da se metoda enako obnaša pri različnih tipih vhodov
    return img.astype(np.float32) / 255.0


def create_synthetic_images(seed=0, size=200, rotation_deg=-1):
    # Fiksen seed da vedno isti random vzorec,
    # zato da pač lahko teste in rezultate po potrebi ponovimo
    np.random.seed(seed)
    im1 = np.random.rand(size, size).astype(np.float32)

    # Drugo sliko dobimo z znano rotacijo.
    # To je nekak dober sanity test, ker približno vemo, kakšen tok pričakujemo
    im2 = rotate_image(im1, rotation_deg)
    return im1, im2


def flow_residual(im1, im2, U, V):
    Ix1, Iy1 = gaussderiv(im1, 1.0)
    Ix2, Iy2 = gaussderiv(im2, 1.0)
    Ix = 0.5 * (Ix1 + Ix2)
    Iy = 0.5 * (Iy1 + Iy2)
    It = im2 - im1
    return float(np.mean((Ix * U + Iy * V + It) ** 2))


def timed_call(fn, repeat=3):
    # Funkcijo večkrat izmerimo in vzamemo povprečje,
    # ker ena sama meritev lahko preveč niha zaradi sistema
    times = []
    out = None
    for _ in range(repeat):
        t0 = perf_counter()
        out = fn()
        times.append(perf_counter() - t0)
    return out, float(np.mean(times))


def plot_lk_hs_result(im1, im2, title, lk_N=3, hs_iters=1000, hs_lambda=0.5):
    # Obe metodi računamo na istem paru slik,
    # da potem lahko rezultate prikažemo in lažje primerjamo
    U_lk, V_lk = lucaskanade(im1, im2, N=lk_N)
    U_hs, V_hs = hornschunck(im1, im2, n_iters=hs_iters, lmbd=hs_lambda)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    axs[0, 0].imshow(im1, cmap="gray")
    axs[0, 0].set_title("Image 1")
    axs[0, 1].imshow(im2, cmap="gray")
    axs[0, 1].set_title("Image 2")
    show_flow(U_lk, V_lk, axs[0, 2], type="angle")
    axs[0, 2].set_title("LK Flow (Angle)")
    show_flow(U_lk, V_lk, axs[0, 3], type="field", set_aspect=True)
    axs[0, 3].set_title("LK Flow (Field)")

    axs[1, 0].imshow(im1, cmap="gray")
    axs[1, 0].set_title("Image 1")
    axs[1, 1].imshow(im2, cmap="gray")
    axs[1, 1].set_title("Image 2")
    show_flow(U_hs, V_hs, axs[1, 2], type="angle")
    axs[1, 2].set_title("HS Flow (Angle)")
    show_flow(U_hs, V_hs, axs[1, 3], type="field", set_aspect=True)
    axs[1, 3].set_title("HS Flow (Field)")

    for ax in axs.ravel():
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()


def plot_lk_variants_result(im1, im2, title, U_lk, V_lk, U_iter, V_iter, U_pyr, V_pyr):
    # Tu prikažemo tri različice LK skupaj,
    # da lažje vidimo, ali dodatne iteracije ali piramida res pomagajo
    fig, axs = plt.subplots(3, 2, figsize=(11, 12))

    show_flow(U_lk, V_lk, axs[0, 0], type="angle")
    axs[0, 0].set_title("LK (Angle)")
    show_flow(U_lk, V_lk, axs[0, 1], type="field", set_aspect=True)
    axs[0, 1].set_title("LK (Field)")

    show_flow(U_iter, V_iter, axs[1, 0], type="angle")
    axs[1, 0].set_title("Iterative LK (Angle)")
    show_flow(U_iter, V_iter, axs[1, 1], type="field", set_aspect=True)
    axs[1, 1].set_title("Iterative LK (Field)")

    show_flow(U_pyr, V_pyr, axs[2, 0], type="angle")
    axs[2, 0].set_title("Pyramidal LK (Angle)")
    show_flow(U_pyr, V_pyr, axs[2, 1], type="field", set_aspect=True)
    axs[2, 1].set_title("Pyramidal LK (Field)")

    for ax in axs.ravel():
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()


def create_large_motion_pair(seed=7, size=220, dx=14, dy=10):
    # Naredimo par z večjim premikom,
    # ker pri takem primeru navadni LK pogosto odpove ali postane nezanesljiv
    np.random.seed(seed)
    im1 = np.random.rand(size, size).astype(np.float32)
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    im2 = cv2.warpAffine(im1, M, (size, size),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return im1, im2


def run_additional_part():
    # (additional part seminarske) preveri ali večstopenjski pristop pomaga pri večjih premikih,
    # kjer en sam lokalni izračun navadno ni dovolj
    print("\n=== Pyramidal LK ===")
    im1, im2 = create_large_motion_pair()

    # Vse tri variante merimo na istem paru,
    # da je primerjava hitrosti in kakovosti poštena
    (U_lk, V_lk), t_lk = timed_call(lambda: lucaskanade(im1, im2, N=5), repeat=1)
    (U_iter, V_iter), t_iter = timed_call(
        lambda: iterative_lucaskanade(im1, im2, N=5, n_iters=5), repeat=1)
    (U_pyr, V_pyr), t_pyr = timed_call(
        lambda: pyramidal_lucaskanade(im1, im2, N=5, n_levels=4, n_iters=3),
        repeat=1,
    )

    r_lk = flow_residual(im1, im2, U_lk, V_lk)
    r_iter = flow_residual(im1, im2, U_iter, V_iter)
    r_pyr = flow_residual(im1, im2, U_pyr, V_pyr)

    # Izpis pokaže kompromis med kakovostjo in časom
    print(f"LK            | residual={r_lk:.6e} | time={t_lk * 1000:.2f} ms")
    print(
        f"Iterative LK  | residual={r_iter:.6e} | time={t_iter * 1000:.2f} ms")
    print(f"Pyramidal LK  | residual={r_pyr:.6e} | time={t_pyr * 1000:.2f} ms")

    plot_lk_hs_result(
        im1,
        im2,
        title="Large Motion Pair (Input + LK/HS layout)",
        lk_N=5,
        hs_iters=600,
        hs_lambda=0.5,
    )
    plot_lk_variants_result(
        im1,
        im2,
        title="LK vs Iterative LK vs Pyramidal LK",
        U_lk=U_lk,
        V_lk=V_lk,
        U_iter=U_iter,
        V_iter=V_iter,
        U_pyr=U_pyr,
        V_pyr=V_pyr,
    )


def run_parameter_study(im1, im2):
    print("\n=== Parameter Study (residual lower is better) ===")

    # Pri LK je glavni parameter velikost lokalnega okna
    # Večje okno bolj gladi rezultat, manjše pa bolje ohrani lokalne podrobnosti
    lk_neighborhoods = [3, 5, 7, 9]
    for n in lk_neighborhoods:
        (u_lk, v_lk), t_lk = timed_call(lambda: lucaskanade(im1, im2, N=n))
        r_lk = flow_residual(im1, im2, u_lk, v_lk)
        print(f"LK: N={n:>2} | residual={r_lk:.6e} | time={t_lk * 1000:.2f} ms")

    # Pri HS sta ključna lambda in število iteracij.
    # Lambda določa, kako gladek naj bo tok, iteracije pa koliko časa metodo izboljšujemo
    hs_lambdas = [0.1, 0.5, 1.0, 2.0]
    hs_iters = [200, 500, 1000]
    for lmbd in hs_lambdas:
        for n_iter in hs_iters:
            (u_hs, v_hs), t_hs = timed_call(
                lambda: hornschunck(im1, im2, n_iters=n_iter, lmbd=lmbd),
                repeat=1,
            )
            r_hs = flow_residual(im1, im2, u_hs, v_hs)
            print(
                f"HS: lambda={lmbd:<3} iters={n_iter:>4} | "
                f"residual={r_hs:.6e} | time={t_hs * 1000:.2f} ms"
            )


def run_runtime_comparison(im1, im2):
    print("\n=== Runtime Comparison ===")

    # Za časovno primerjavo vzamemo fiksne nastavitve,
    # da primerjamo metodi pod enakimi pogoji
    (_, _), t_lk = timed_call(lambda: lucaskanade(im1, im2, N=5), repeat=5)
    (_, _), t_hs = timed_call(
        lambda: hornschunck(im1, im2, n_iters=1000, lmbd=0.5),
        repeat=3,
    )

    print(f"Lucas-Kanade (N=5): {t_lk * 1000:.2f} ms (avg)")
    print(f"Horn-Schunck (1000 iters, lambda=0.5): {t_hs * 1000:.2f} ms (avg)")


def run_hs_lk_init_comparison(im1, im2):
    print("\n=== HS Initialization Comparison ===")

    # Ta primerjava pokaže, ali lahko HS pospešimo tako,
    # da mu kot začetni približek podamo rezultat metode LK
    (u_hs_zero, v_hs_zero), t_zero = timed_call(
        lambda: hornschunck(im1, im2, n_iters=1000, lmbd=0.5),
        repeat=1,
    )
    r_zero = flow_residual(im1, im2, u_hs_zero, v_hs_zero)

    # Tok iz LK uporabimo kot začetni približek za HS
    # Ideja je, da HS zato potrebuje manj iteracij za podoben rezultat
    (u_lk, v_lk), t_lk = timed_call(lambda: lucaskanade(im1, im2, N=5), repeat=1)

    def hs_with_init():
        # Podpremo obe podpisni varianti funkcije:
        # hornschunck(..., U0=..., V0=...) ali hornschunck(..., init_u=..., init_v=...)
        try:
            return hornschunck(im1, im2, n_iters=300, lmbd=0.5, U0=u_lk, V0=v_lk)
        except TypeError:
            return hornschunck(
                im1, im2, n_iters=300, lmbd=0.5, init_u=u_lk, init_v=v_lk
            )

    (u_hs_init, v_hs_init), t_hs_init = timed_call(hs_with_init, repeat=1)
    r_init = flow_residual(im1, im2, u_hs_init, v_hs_init)

    print(
        f"HS zero init: iters=1000 | residual={r_zero:.6e} | "
        f"time={t_zero * 1000:.2f} ms"
    )
    print(
        f"LK init + HS: LK(N=5)+HS(300) | residual={r_init:.6e} | "
        f"total_time={(t_lk + t_hs_init) * 1000:.2f} ms"
    )


# Glavno izvajane: najprej obvezni eksperimenti, nato meritve in additional del
# 1) Najprej zaženemo sanity test na sintetičnem primeru,
# kjer lahko hitro preverimo, ali obe metodi vračata smiseln flow
im1_syn, im2_syn = create_synthetic_images(
    seed=0, size=200, rotation_deg=-1)
plot_lk_hs_result(
    im1_syn,
    im2_syn,
    title="Optical Flow (Synthetic Rotated Random Noise)",
    lk_N=3,
    hs_iters=1000,
    hs_lambda=0.5,
)
# 2) Nato zaženemo še tri zahtevane realne pare iz mape disparity,
# da vidimo, kako se metodi obneseta na bolj realističnih podatkih
real_pairs = [
    ("disparity/office_left.png",
     "disparity/office_right.png", "Disparity: office"),
    ("disparity/office2_left.png",
     "disparity/office2_right.png", "Disparity: office2"),
    ("disparity/cporta_left.png",
     "disparity/cporta_right.png", "Disparity: cporta"),
]
for p1, p2, title in real_pairs:
    im1 = load_gray_image(WORKSPACE_DIR / p1)
    im2 = load_gray_image(WORKSPACE_DIR / p2)
    plot_lk_hs_result(
        im1,
        im2,
        title=f"Optical Flow: {title}",
        lk_N=3,
        hs_iters=1000,
        hs_lambda=0.5,
    )
# 3) 4) 5) Številčne primerjave naredimo na enem reprezentativnem paru,
# da dobimo pregled nad vplivom parametrov, hitrostjo in inicializacijo HS z LK
study_im1 = load_gray_image(WORKSPACE_DIR / "disparity/office_left.png")
study_im2 = load_gray_image(WORKSPACE_DIR / "disparity/office_right.png")
run_parameter_study(study_im1, study_im2)
run_runtime_comparison(study_im1, study_im2)
run_hs_lk_init_comparison(study_im1, study_im2)
run_additional_part()
# 6) Na koncu prikažemo vse figure, ki smo jih pripravili med zagonom
plt.show()
