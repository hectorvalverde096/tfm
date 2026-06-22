"""
prepare_patches.py — Pre-extrae parches LR/HR y los escribe en archivos binarios
accesibles via np.memmap. Elimina el cuello de botella de I/O en entrenamiento.

Uso:
    python prepare_patches.py --hr_dir data/train/hr --lr_dir data/train/lr
                              --out_dir data/patches/train
                              --patches_per_img 3 --patch_hr 256

Salida:
    out_dir/hr.bin    — array (N, patch_hr, patch_hr, 3) uint8
    out_dir/lr.bin    — array (N, patch_lr, patch_lr, 3) uint8
    out_dir/meta.npy  — [N, patch_hr, patch_lr]
"""

import argparse
import os
import glob
import cv2
import numpy as np


def extract_patches(hr, lr, patch_hr, n, rng):
    patch_lr = patch_hr // 2
    h_lr, w_lr = lr.shape[:2]
    if h_lr < patch_lr or w_lr < patch_lr:
        return [], []

    patches_hr, patches_lr = [], []
    for _ in range(n):
        top  = rng.integers(0, h_lr - patch_lr + 1)
        left = rng.integers(0, w_lr - patch_lr + 1)
        patches_lr.append(lr[top:top + patch_lr, left:left + patch_lr])
        patches_hr.append(hr[top * 2:(top + patch_lr) * 2, left * 2:(left + patch_lr) * 2])
    return patches_hr, patches_lr


def build_memmap(hr_dir, lr_dir, out_dir, patches_per_img=3, patch_hr=256):
    hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
    lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
    assert len(hr_paths) == len(lr_paths) and len(hr_paths) > 0, \
        f"Se necesitan pares HR/LR iguales. Encontrados: {len(hr_paths)} HR, {len(lr_paths)} LR"

    patch_lr = patch_hr // 2
    N = len(hr_paths) * patches_per_img
    os.makedirs(out_dir, exist_ok=True)

    hr_path   = os.path.join(out_dir, 'hr.bin')
    lr_path   = os.path.join(out_dir, 'lr.bin')
    meta_path = os.path.join(out_dir, 'meta.npy')

    hr_gb = N * patch_hr * patch_hr * 3 / 1024**3
    lr_gb = N * patch_lr * patch_lr * 3 / 1024**3
    print(f"Generando {N} parches ({patches_per_img} por imagen, {len(hr_paths)} imagenes)")
    print(f"Tamano estimado: HR={hr_gb:.1f} GB  LR={lr_gb:.1f} GB  Total={hr_gb+lr_gb:.1f} GB")
    print(f"Destino: {out_dir}")

    mmap_hr = np.memmap(hr_path, dtype='uint8', mode='w+', shape=(N, patch_hr, patch_hr, 3))
    mmap_lr = np.memmap(lr_path, dtype='uint8', mode='w+', shape=(N, patch_lr, patch_lr, 3))

    rng = np.random.default_rng(0)
    idx = 0
    for i, (hp, lp) in enumerate(zip(hr_paths, lr_paths)):
        hr = cv2.imread(hp)
        lr = cv2.imread(lp)
        if hr is None or lr is None:
            continue

        patches_h, patches_l = extract_patches(hr, lr, patch_hr, patches_per_img, rng)
        for p_hr, p_lr in zip(patches_h, patches_l):
            if idx >= N:
                break
            mmap_hr[idx] = p_hr
            mmap_lr[idx] = p_lr
            idx += 1

        if (i + 1) % 500 == 0:
            mmap_hr.flush()
            mmap_lr.flush()
            pct = (i + 1) / len(hr_paths) * 100
            print(f"  {i+1}/{len(hr_paths)} ({pct:.0f}%)  ->  {idx} parches escritos", end='\r')

    if idx < N:
        print(f"\nAviso: {N - idx} slots vacios. Recortando a {idx}.")
        N = idx

    np.save(meta_path, np.array([N, patch_hr, patch_lr], dtype=np.int64))
    del mmap_hr, mmap_lr

    print(f"\nListo. {N} parches guardados en {out_dir}")
    print(f"  hr.bin : {os.path.getsize(hr_path)/1024**3:.2f} GB")
    print(f"  lr.bin : {os.path.getsize(lr_path)/1024**3:.2f} GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-extrae parches para AnimeSR-EdgeGAN')
    parser.add_argument('--hr_dir',          default='data/train/hr')
    parser.add_argument('--lr_dir',          default='data/train/lr')
    parser.add_argument('--out_dir',         default='data/patches/train')
    parser.add_argument('--patches_per_img', default=3, type=int)
    parser.add_argument('--patch_hr',        default=256, type=int)
    args = parser.parse_args()

    build_memmap(args.hr_dir, args.lr_dir, args.out_dir,
                 args.patches_per_img, args.patch_hr)
