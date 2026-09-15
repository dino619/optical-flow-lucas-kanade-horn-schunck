import cv2
import numpy as np

from ex1_utils import gaussderiv


def lucaskanade(im1, im2, N):
    # Sliki morata biti enake velikosti, ker flow računamo za iste piksle na obeh slikah.
    if im1.shape != im2.shape:
        raise ValueError("Input images must have the same shape")
    # Metoda je definirana za grayscale slike, zato pričakujemo samo en kanal.
    if im1.ndim != 2 or im2.ndim != 2:
        raise ValueError("Input images must be grayscale (2D)")
    # Velikost okna mora biti pozitivna in liha, da ima local neighborhood jasen center.
    if N <= 0 or N % 2 == 0:
        raise ValueError(
            "Neighborhood size N must be a positive and odd integer")

    im1_float = im1.astype(np.float32)
    im2_float = im2.astype(np.float32)

    # Sliki normaliziramo na primerljiv obseg, da je metoda bolj stable pri različnih vhodih
    if np.max(im1_float) > 1.0:
        im1_float /= 255.0
    if np.max(im2_float) > 1.0:
        im2_float /= 255.0

    sigma = 1.0

    # Gradiente izračunamo na obeh slikah in jih povprečimo,
    # ker je taka ocena praviloma bolj stabilna kot gradient samo ene slike.
    Ix_img1, Iy_img1 = gaussderiv(im1_float, sigma)
    Ix_img2, Iy_img2 = gaussderiv(im2_float, sigma)
    Ix = 0.5 * (Ix_img1 + Ix_img2)
    Iy = 0.5 * (Iy_img1 + Iy_img2)
    It = im2_float - im1_float

    # Pripravimo produkte gradientov, ki jih potrebujemo v lokalnem sistemu enačb metode LK.
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    Ixt = Ix * It
    Iyt = Iy * It

    # Lokalno vsoto naredimo z box filtrom brez zank,
    # ker tako hitro dobimo info. iz neighborhood-a za vsak piksel.
    border_type = cv2.BORDER_REFLECT
    Sxx = cv2.boxFilter(Ix2, -1, (N, N), normalize=False,
                        borderType=border_type)
    Syy = cv2.boxFilter(Iy2, -1, (N, N), normalize=False,
                        borderType=border_type)
    Sxy = cv2.boxFilter(Ixy, -1, (N, N), normalize=False,
                        borderType=border_type)
    Sxt = cv2.boxFilter(Ixt, -1, (N, N), normalize=False,
                        borderType=border_type)
    Syt = cv2.boxFilter(Iyt, -1, (N, N), normalize=False,
                        borderType=border_type)

    determinant = Sxx * Syy - Sxy * Sxy
    num_u = Syy * Sxt - Sxy * Syt
    num_v = Sxx * Syt - Sxy * Sxt

    # Če je determinanta zelo majhna, je rešitev nestabilna,
    # zato takih točk bolje da ne uporabimo.
    eps = 1e-8
    valid = np.abs(determinant) > eps

    # Harrisov odziv uporabimo kot dodaten test,
    # ali ima lokalno območje dovolj strukture za zanesljiv izračun toka.
    harris_k = 0.04
    harris_response = determinant - harris_k * (Sxx + Syy) * (Sxx + Syy)
    max_harris = np.max(harris_response)
    if max_harris > 0:
        # Nižji prag pusti več veljavnih točk pri realnih slikah.
        valid &= harris_response > (0.001 * max_harris)
    else:
        valid &= False

    U = np.zeros_like(im1_float, dtype=np.float32)
    V = np.zeros_like(im1_float, dtype=np.float32)
    U[valid] = -num_u[valid] / determinant[valid]
    V[valid] = -num_v[valid] / determinant[valid]

    # Izločimo ekstremno velike vektorje,
    # ker so pogosto posledica napak in zelo pokvarijo prikaz polja.
    magnitude = np.sqrt(U * U + V * V)
    mag_valid = magnitude[valid]
    if mag_valid.size > 0:
        max_mag = np.percentile(mag_valid, 98.0)
        outliers = magnitude > max_mag
        U[outliers] = 0.0
        V[outliers] = 0.0

    return U, V


def hornschunck(im1, im2, n_iters, lmbd, init_u=None, init_v=None):
    # Osnovne kontrole vhodov ustavijo napake takoj na začetku,
    # namesto da bi kasneje dobili slab rezultat brez jasnega razloga.
    if im1.shape != im2.shape:
        raise ValueError("Input images must have the same shape")
    if im1.ndim != 2 or im2.ndim != 2:
        raise ValueError("Input images must be grayscale (2D)")
    if n_iters <= 0:
        raise ValueError("n_iters must be a positive")
    if lmbd <= 0:
        raise ValueError("lmbd must be positive")

    im1_float = im1.astype(np.float32)
    im2_float = im2.astype(np.float32)

    # Tudi pri HS vhod normaliziramo,
    # da je obnašanje bolj stabilno pri različnih datotekah.
    if np.max(im1_float) > 1.0:
        im1_float /= 255.0
    if np.max(im2_float) > 1.0:
        im2_float /= 255.0

    sigma = 1.0
    Ix1, Iy1 = gaussderiv(im1_float, sigma)
    Ix2, Iy2 = gaussderiv(im2_float, sigma)
    Ix = 0.5 * (Ix1 + Ix2)
    Iy = 0.5 * (Iy1 + Iy2)
    It = im2_float - im1_float

    # Po želji lahko začnemo iz toka metode LK,
    # da preverimo, ali HS zato potrebuje manj iteracij.
    if init_u is not None and init_v is not None:
        if init_u.shape != im1_float.shape or init_v.shape != im1_float.shape:
            raise ValueError(
                "init_u and init_v must have the same shape as input images")
        U = init_u.astype(np.float32).copy()
        V = init_v.astype(np.float32).copy()
    elif init_u is None and init_v is None:
        U = np.zeros_like(im1_float, dtype=np.float32)
        V = np.zeros_like(im1_float, dtype=np.float32)
    else:
        raise ValueError("Both init_u and init_v must be provided together")

    # Jedro vzame povprečje štirih sosedov.
    # S tem dobimo gladek lokalni približek toka oz. flow-a.
    laplacian_kernel = np.array(
        [[0.0, 0.25, 0.0],
         [0.25, 0.0, 0.25],
         [0.0, 0.25, 0.0]],
        dtype=np.float32,
    )
    D = lmbd + Ix * Ix + Iy * Iy
    eps = 1e-8

    # V vsaki iteraciji tok najprej zgladimo,
    # nato pa ga popravimo glede na optično enačbo med slikama.
    for _ in range(n_iters):
        Ua = cv2.filter2D(U, -1, laplacian_kernel,
                          borderType=cv2.BORDER_REFLECT)
        Va = cv2.filter2D(V, -1, laplacian_kernel,
                          borderType=cv2.BORDER_REFLECT)

        P = Ix * Ua + Iy * Va + It
        denom = D + eps
        U = Ua - (Ix * P) / denom
        V = Va - (Iy * P) / denom

    return U, V


def _warp_with_flow(im, U, V):
    # Drugo sliko premaknemo glede na trenutno oceno flow-a,
    # da jo lažje poravnamo s prvo sliko.
    # Ta korak je ključen pri iterativnem in piramidalnem LK.
    h, w = im.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    map_x = x + U.astype(np.float32)
    map_y = y + V.astype(np.float32)
    return cv2.remap(im.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def iterative_lucaskanade(im1, im2, N=5, n_iters=5):
    # Iterativni LK večkrat ponovi isti postopek na isti skali.
    # Po vsakem koraku drugo sliko poravnamo, da naslednji popravek lovi manjšo napako.
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")

    im1_f = im1.astype(np.float32)
    im2_f = im2.astype(np.float32)
    if np.max(im1_f) > 1.0:
        im1_f /= 255.0
    if np.max(im2_f) > 1.0:
        im2_f /= 255.0

    # Tok začnemo z ničlo in ga nato po korakih popravljamo.
    U = np.zeros_like(im1_f, dtype=np.float32)
    V = np.zeros_like(im1_f, dtype=np.float32)
    for _ in range(n_iters):
        # Najprej poravnamo drugo sliko, nato izračunamo majhen popravek toka.
        im2_warp = _warp_with_flow(im2_f, U, V)
        dU, dV = lucaskanade(im1_f, im2_warp, N)
        U += dU
        V += dV
    return U, V


def pyramidal_lucaskanade(im1, im2, N=5, n_levels=4, n_iters=3):
    # Piramida omogoči, da večje premike najprej rešujemo na grobi skali,
    # kjer so videti manjši in jih je lažje oceniti.
    if n_levels <= 0:
        raise ValueError("n_levels must be positive")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")

    im1_f = im1.astype(np.float32)
    im2_f = im2.astype(np.float32)
    if np.max(im1_f) > 1.0:
        im1_f /= 255.0
    if np.max(im2_f) > 1.0:
        im2_f /= 255.0

    # Zgradimo piramido obeh slik, da imamo isto vsebino na več ločljivostih.
    pyr1 = [im1_f]
    pyr2 = [im2_f]
    for _ in range(1, n_levels):
        pyr1.append(cv2.pyrDown(pyr1[-1]))
        pyr2.append(cv2.pyrDown(pyr2[-1]))

    U = np.zeros_like(pyr1[-1], dtype=np.float32)
    V = np.zeros_like(pyr1[-1], dtype=np.float32)

    # Od grobe skale gremo proti fini in sproti izboljšujemo tok.
    for lvl in range(n_levels - 1, -1, -1):
        im1_l = pyr1[lvl]
        im2_l = pyr2[lvl]

        if lvl < n_levels - 1:
            # Tok iz grobe skale povečamo na trenutno ločljivost
            # in ga ustrezno pomnožimo zaradi spremembe skale.
            U = cv2.resize(
                U, (im1_l.shape[1], im1_l.shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0
            V = cv2.resize(
                V, (im1_l.shape[1], im1_l.shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0

        for _ in range(n_iters):
            # Na vsaki ravni naredimo več lokalnih popravkov,
            # da je ocena pred prehodom na bolj fino skalo boljša.
            im2_warp = _warp_with_flow(im2_l, U, V)
            dU, dV = lucaskanade(im1_l, im2_warp, N)
            U += dU
            V += dV

    return U, V
