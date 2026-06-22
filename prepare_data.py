"""
prepare_data.py — Genera pares LR/HR para el entrenamiento a partir de frames HD.
Integrado con el pipeline existente de pruebas.py.

Uso:
    python prepare_data.py --input_dir pruebas/in --output_dir data/train --scale 2
"""

import argparse
import os
import glob
import cv2
import numpy as np
from pathlib import Path


def add_anime_degradation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Degradación realista para animación antigua:
      1. Downscale bicubic (principal)
      2. Ruido AWGN ligero (simula grain de VHS/TV)
      3. Compresión JPEG ligera (simula artefactos de distribución)
      4. Ligero blur gaussiano (simula captura analógica)

    Nota: para animación es preferible una degradación leve —
    queremos que el modelo aprenda a reconstruir bordes, no a eliminar ruido severo.
    """
    # 1. Aplicar blur antes de downscale (simula antialiasing del master original)
    sigma = rng.uniform(0.2, 0.8)
    ksize = int(sigma * 6) | 1  # impar
    if ksize > 1:
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    # 2. Downscale bicubic
    h, w = img.shape[:2]
    lr = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)

    # 3. Ruido AWGN ligero
    noise_std = rng.uniform(0, 8)
    noise = rng.normal(0, noise_std, lr.shape).astype(np.float32)
    lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Compresión JPEG ligera
    quality = int(rng.uniform(75, 98))
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, enc = cv2.imencode('.jpg', lr, encode_params)
    lr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return lr


def prepare_data(input_dir: str, output_dir: str, scale: int = 2, val_split: float = 0.05):
    """
    Genera pares LR/HR a partir de frames HD.

    Args:
        input_dir:  Directorio con imágenes HR (frames GT)
        output_dir: Directorio de salida
        scale:      Factor de downscale (2 = x2)
        val_split:  Fracción para validación
    """
    hr_paths = sorted(
        glob.glob(os.path.join(input_dir, '*.png')) +
        glob.glob(os.path.join(input_dir, '*.jpg'))
    )
    if not hr_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {input_dir}")

    rng = np.random.default_rng(42)
    indices = np.arange(len(hr_paths))
    rng.shuffle(indices)
    val_n = max(1, int(len(hr_paths) * val_split))
    val_idx = set(indices[:val_n].tolist())

    splits = {'train': [], 'val': []}
    for i, p in enumerate(hr_paths):
        splits['val' if i in val_idx else 'train'].append(p)

    for split, paths in splits.items():
        hr_out = os.path.join(output_dir, split, 'hr')
        lr_out = os.path.join(output_dir, split, 'lr')
        os.makedirs(hr_out, exist_ok=True)
        os.makedirs(lr_out, exist_ok=True)

        for idx, path in enumerate(paths):
            img_hr = cv2.imread(path)
            if img_hr is None:
                print(f"  Saltando (no se pudo leer): {path}")
                continue

            # Asegurar que las dimensiones sean múltiplo de scale*8
            h, w = img_hr.shape[:2]
            h_new = (h // (scale * 8)) * (scale * 8)
            w_new = (w // (scale * 8)) * (scale * 8)
            if h_new != h or w_new != w:
                img_hr = img_hr[:h_new, :w_new]

            img_lr = add_anime_degradation(img_hr, rng)

            name = Path(path).stem
            cv2.imwrite(os.path.join(hr_out, f'{name}.png'), img_hr)
            cv2.imwrite(os.path.join(lr_out, f'{name}.png'), img_lr)

            if (idx + 1) % 100 == 0:
                print(f"  [{split}] {idx+1}/{len(paths)} procesados...", end='\r')

        print(f"  [{split}] {len(paths)} pares generados → {hr_out}")

    print(f"\nDatos listos en: {output_dir}")
    print(f"  Train: {len(splits['train'])} pares")
    print(f"  Val:   {len(splits['val'])} pares")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparar datos LR/HR para AnimeSR-EdgeGAN')
    parser.add_argument('--input_dir',  default='pruebas/in',  help='Directorio con imágenes GT')
    parser.add_argument('--output_dir', default='data',        help='Directorio de salida')
    parser.add_argument('--scale',      default=2, type=int,   help='Factor de escala')
    parser.add_argument('--val_split',  default=0.05, type=float, help='Fracción para validación')
    args = parser.parse_args()

    prepare_data(args.input_dir, args.output_dir, args.scale, args.val_split)
