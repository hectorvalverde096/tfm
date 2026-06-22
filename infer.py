"""
infer.py — Inferencia con AnimeSR-EdgeGAN
Drop-in replacement para el bloque de inferencia de pruebas.py.
Soporta x2 (una pasada) o x4 (dos pasadas), igual que el pipeline original.
"""

import torch
import cv2
import glob
import numpy as np
import os
import argparse
from model import AnimeEdgeGenerator
from utils import to_tensor, to_image, calcular_psnr, calcular_ssim, calcular_lpips, medir_nitidez_bordes


def load_generator(checkpoint_path: str, device: str) -> AnimeEdgeGenerator:
    """Carga el generador desde un checkpoint de entrenamiento."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    G = AnimeEdgeGenerator(num_features=64, num_rrdb=16, growth=32).to(device)

    # El checkpoint puede contener el state_dict directamente o dentro de 'generator'
    state = ckpt.get('generator', ckpt)
    G.load_state_dict(state)
    G.eval()
    print(f"Modelo cargado desde: {checkpoint_path}")
    if 'epoch' in ckpt:
        print(f"  Época de entrenamiento: {ckpt['epoch']}")
    return G


@torch.no_grad()
def upscale(model: AnimeEdgeGenerator, img_tensor: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """
    Aplica el modelo (x2) una o dos veces para obtener x2 o x4.
    Idéntico a la lógica de pruebas.py.
    """
    if scale == 2:
        return model(img_tensor)
    elif scale == 4:
        x2 = model(img_tensor)
        return model(x2)
    else:
        raise ValueError(f"Escala {scale} no soportada. Usa 2 o 4.")


def infer_single(args):
    """Inferencia sobre una imagen individual con métricas opcionales."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_generator(args.checkpoint, device)

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {args.input}")

    img_tensor = to_tensor(img).to(device)
    out_tensor = upscale(model, img_tensor, scale=args.scale)
    out_img = to_image(out_tensor)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, out_img)
    print(f"Resultado guardado: {args.output}")
    print(f"Nitidez (Laplacian): {medir_nitidez_bordes(out_img):.2f}")

    # Métricas contra GT si se proporciona
    if args.ref and os.path.exists(args.ref):
        img_ref = cv2.imread(args.ref)
        tensor_ref = to_tensor(img_ref).to(device)
        print(f"\n--- Métricas de Calidad ---")
        print(f"PSNR:  {calcular_psnr(img_ref, out_img):.2f} dB")
        print(f"SSIM:  {calcular_ssim(img_ref, out_img):.4f}")
        print(f"LPIPS: {calcular_lpips(tensor_ref, out_tensor, device=device):.4f}")


def infer_video(args):
    """Inferencia por lotes sobre frames de video — idéntico a pruebas.py sección 5."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_generator(args.checkpoint, device)

    frames = sorted(glob.glob(os.path.join(args.video_dir, '*.png')))
    if not frames:
        print(f"No se encontraron frames en {args.video_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Configurar VideoWriter
    img0 = cv2.imread(frames[0])
    h, w = img0.shape[:2]
    factor = args.scale
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(
        os.path.join(args.output_dir, f'resultado_edgegan_{factor}x.mp4'),
        fourcc, 30.0, (w * factor, h * factor)
    )

    prev_frame = None
    temp_errors = []

    for frame_path in frames:
        img = cv2.imread(frame_path)
        t = to_tensor(img).to(device)
        out_t = upscale(model, t, scale=args.scale)
        out_img = to_image(out_t)

        name = os.path.basename(frame_path).replace('.png', f'_x{factor}.png')
        cv2.imwrite(os.path.join(args.output_dir, name), out_img)
        video_out.write(out_img)

        if prev_frame is not None:
            from utils import calcular_error_temporal
            temp_errors.append(calcular_error_temporal(out_img, prev_frame))

        prev_frame = out_img
        print(f"Frame {name} completado.", end='\r')

    video_out.release()
    print(f"\nVideo guardado en: {args.output_dir}")
    if temp_errors:
        print(f"Error temporal medio: {np.mean(temp_errors):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AnimeSR-EdgeGAN Inferencia')
    sub = parser.add_subparsers(dest='mode')

    # Modo imagen individual
    p_single = sub.add_parser('image', help='Inferencia sobre una imagen')
    p_single.add_argument('--input',      required=True, help='Imagen de entrada')
    p_single.add_argument('--output',     required=True, help='Imagen de salida')
    p_single.add_argument('--checkpoint', required=True, help='Ruta al checkpoint .pth')
    p_single.add_argument('--scale',      default=2, type=int, choices=[2, 4])
    p_single.add_argument('--ref',        default=None, help='Imagen GT para métricas (opcional)')

    # Modo video
    p_video = sub.add_parser('video', help='Inferencia sobre directorio de frames')
    p_video.add_argument('--video_dir',   required=True, help='Directorio con frames .png')
    p_video.add_argument('--output_dir',  required=True, help='Directorio de salida')
    p_video.add_argument('--checkpoint',  required=True, help='Ruta al checkpoint .pth')
    p_video.add_argument('--scale',       default=2, type=int, choices=[2, 4])

    args = parser.parse_args()

    if args.mode == 'image':
        infer_single(args)
    elif args.mode == 'video':
        infer_video(args)
    else:
        parser.print_help()
