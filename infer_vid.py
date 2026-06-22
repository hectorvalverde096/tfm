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


# ─────────────────────────────────────────────
#  POST-PROCESO: ESTILIZACIÓN CEL (CAMINO 3)
# ─────────────────────────────────────────────

def crispen_celart(
    img_bgr: np.ndarray,
    edge_thr:   int   = 40,    # Umbral Canny inferior (superior = 3×). Sube → menos bordes detectados.
    line_gamma: float = 2.2,   # Gamma en zona de borde. >1 → aplasta sombras → líneas más negras.
    flatten_d:  int   = 7,     # Diámetro del filtro bilateral para aplanar rellenos.
    flatten_sc: float = 80.0,  # Sigma-color bilateral. Sube para aplanar colores más distintos.
    dilate_px:  int   = 2,     # Expansión en px de la máscara de borde (evita cortes duros).
    strength:   float = 1.0,   # 0.0 = sin efecto, 1.0 = efecto completo. Valores intermedios = blend.
) -> np.ndarray:
    """
    Post-proceso de estilización cel para la salida del modelo.

    Efecto:
      - Detecta contornos con Canny.
      - En zonas SIN borde: aplana el relleno con filtro bilateral,
        eliminando el antialiasing residual.
      - En zonas DE borde: aplica gamma > 1, que aplasta los píxeles
        grises de la transición hacia el negro, potenciando la línea.
      - Mezcla el resultado con la imagen original según `strength`.

    Parámetros recomendados para animación:
        edge_thr   30-50   (50+ para material muy limpio)
        line_gamma 1.8-2.8 (2.2 es un buen punto de partida)
        flatten_d  5-9     (más alto = más plano, más lento)
        strength   0.6-1.0 (empieza en 0.7 para ver el efecto sin pasarte)
    """
    if strength <= 0.0:
        return img_bgr.copy()

    # ── 1. Detección de bordes ──────────────────────────────────────
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_thr, edge_thr * 3)

    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        edges  = cv2.dilate(edges, kernel)

    # Máscara suavizada para blending sin artefactos de borde duro
    mask_hard = (edges > 0).astype(np.float32)
    mask      = cv2.GaussianBlur(mask_hard, (5, 5), 1.2)[..., None]  # (H,W,1)

    img_f = img_bgr.astype(np.float32) / 255.0

    # ── 2. Relleno aplanado (bilateral en imagen original) ──────────
    # El bilateral respeta bordes fuertes pero aplana gradientes suaves
    flat = cv2.bilateralFilter(
        img_bgr, flatten_d, flatten_sc, flatten_sc
    ).astype(np.float32) / 255.0

    # ── 3. Línea potenciada (gamma crush sobre zona de borde) ───────
    # power(x, gamma>1) aplasta los grises hacia 0, dejando blancos intactos.
    # Resultado: la transición gris del antialiasing se convierte en negro sólido.
    line = np.power(np.clip(img_f, 1e-6, 1.0), line_gamma)

    # ── 4. Composición: línea potenciada | relleno plano ────────────
    styled = mask * line + (1.0 - mask) * flat

    # ── 5. Blend final con imagen original según strength ───────────
    result = strength * styled + (1.0 - strength) * img_f
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  ESTABILIZACIÓN TEMPORAL (Nivel 0b)
# ─────────────────────────────────────────────

def _warp_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Deforma una imagen BGR según un campo de flujo denso.
      img:  (H, W, 3) uint8
      flow: (H, W, 2) float32 — desplazamiento en píxeles (dx, dy)
    Devuelve imagen deformada (H, W, 3) uint8.
    """
    H, W = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                  np.arange(H, dtype=np.float32))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _occlusion_mask(flow_fw: np.ndarray, flow_bw: np.ndarray,
                    scale: int, occ_thr: float = 20.0) -> np.ndarray:
    """
    Máscara de fiabilidad del warp (float32, H_lr×W_lr×1, rango [0,1]).
    Se calcula en resolución LR (la del flujo) y se reescala fuera.

    Vale 0 donde el warp NO es fiable, por cualquiera de estas razones:
      - El flujo apunta fuera de los límites de la imagen.
      - La magnitud del flujo supera occ_thr (movimiento demasiado rápido).
      - El flujo forward y backward son INCONSISTENTES. Este es el filtro
        clave para animación: en zonas planas (cielo, rellenos) Farneback
        inventa flujo de magnitud pequeña pero incoherente; la comprobación
        forward-backward (Sundaram et al.) lo detecta y lo descarta, mientras
        que el chequeo por magnitud solo no lo pillaba.

    Parámetro occ_thr (en px de resolución SR):
      8-15   → conservador, descarta mucho movimiento rápido (recomendado anime)
      20-35  → más permisivo, usa más del warp (riesgo de ghosting)
    """
    H, W = flow_fw.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                  np.arange(H, dtype=np.float32))

    # Magnitud (en px LR; occ_thr viene en px SR → comparamos con occ_thr/scale)
    mag   = np.linalg.norm(flow_fw, axis=2)                       # (H, W)
    dst_x = grid_x + flow_fw[..., 0]
    dst_y = grid_y + flow_fw[..., 1]
    in_bounds = ((dst_x >= 0) & (dst_x < W) &
                 (dst_y >= 0) & (dst_y < H)).astype(np.float32)

    # Consistencia forward-backward:
    # se sigue el flujo forward y, en el destino, se lee el flujo backward.
    # Si el warp es fiable, fw(x) + bw(x + fw(x)) ≈ 0.
    bw_x = cv2.remap(flow_bw[..., 0], dst_x, dst_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    bw_y = cv2.remap(flow_bw[..., 1], dst_x, dst_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    fb_err2  = (flow_fw[..., 0] + bw_x) ** 2 + (flow_fw[..., 1] + bw_y) ** 2
    mag_bw2  = bw_x ** 2 + bw_y ** 2
    # Umbral relativo: tolera más error donde el movimiento es grande.
    thr      = 0.01 * (mag ** 2 + mag_bw2) + 0.5
    consistent = (fb_err2 < thr).astype(np.float32)

    valid = in_bounds * (mag < (occ_thr / float(scale))).astype(np.float32) * consistent

    # Suavizado para blend sin aristas
    valid = cv2.GaussianBlur(valid, (7, 7), 2.0)
    return valid[..., None]                                        # (H, W, 1)


def _is_scene_cut(prev_gray: np.ndarray, curr_gray: np.ndarray,
                  cut_thr: float = 30.0) -> bool:
    """
    Detección de corte de escena por diferencia media de luminancia.
    Si la diferencia supera cut_thr, se asume corte y se reinicia el estado.
    Ajusta cut_thr según el material:
      20-25  → sensible, bueno para anime con transiciones suaves
      30-40  → estándar
      50+    → solo detecta cortes muy abruptos
    """
    return float(np.mean(cv2.absdiff(curr_gray, prev_gray))) > cut_thr


class TemporalStabilizer:
    """
    Estabilización temporal frame a frame mediante flujo óptico denso
    (Gunnar Farneback) + warp del frame SR anterior + blending ponderado
    por máscara de oclusión.

    Funcionamiento:
      1. Calcula flujo óptico entre los frames LR consecutivos (más rápido
         que sobre SR y suficientemente preciso para escala 2×).
      2. Escala el flujo a resolución SR (×scale).
      3. Deforma el frame SR anterior para alinearlo con el actual.
      4. Genera máscara de oclusión que excluye regiones con movimiento
         muy rápido o flujo que sale de los límites.
      5. Mezcla: zonas fiables → promedio ponderado (warp + SR puro),
                 zonas ocluidas → SR puro.
      6. Detecta cortes de escena y reinicia el estado.

    Parámetros de Farneback:
      pyr_scale  escala de pirámide (0.5 estándar)
      levels     niveles de pirámide; más niveles → capta movimiento rápido
      winsize    ventana de suavizado; más grande → flujo más suave pero menos
                 detallado (15 es un buen equilibrio)
      iterations iteraciones por nivel
      poly_n     vecindad del polinomio (5 o 7)
      poly_sigma suavizado del polinomio (1.1-1.5)
    """

    def __init__(
        self,
        scale:      int   = 2,
        strength:   float = 0.75,
        occ_thr:    float = 20.0,
        cut_thr:    float = 30.0,
        pyr_scale:  float = 0.5,
        levels:     int   = 3,
        winsize:    int   = 15,
        iterations: int   = 3,
        poly_n:     int   = 5,
        poly_sigma: float = 1.2,
    ):
        self.scale     = scale
        self.strength  = strength
        self.occ_thr   = occ_thr
        self.cut_thr   = cut_thr
        self.fb_kw     = dict(
            pyr_scale  = pyr_scale,
            levels     = levels,
            winsize    = winsize,
            iterations = iterations,
            poly_n     = poly_n,
            poly_sigma = poly_sigma,
            flags      = 0,
        )
        self._prev_gray: np.ndarray | None = None
        self._prev_sr:   np.ndarray | None = None
        self._scene_cut_flag = False   # True cuando el último frame fue un corte

    def reset(self):
        """Reinicia el estado (útil ante cortes manuales o cambio de clip)."""
        self._prev_gray = None
        self._prev_sr   = None

    def __call__(
        self,
        lr_frame: np.ndarray,
        sr_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Estabiliza sr_frame usando el contexto del frame anterior.
          lr_frame: frame de entrada LR (BGR uint8) — para calcular el flujo
          sr_frame: salida del modelo para este frame (BGR uint8)
        Devuelve sr_frame estabilizado (BGR uint8).
        """
        curr_gray = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2GRAY)

        # Primer frame o estabilización desactivada
        if self._prev_gray is None or self.strength <= 0.0:
            self._prev_gray = curr_gray
            self._prev_sr   = sr_frame.copy()
            return sr_frame

        # ── Detección de corte ────────────────────────────────────────────
        if _is_scene_cut(self._prev_gray, curr_gray, self.cut_thr):
            self._prev_gray = curr_gray
            self._prev_sr   = sr_frame.copy()
            self._scene_cut_flag = True
            return sr_frame          # frame limpio, sin mezcla con estado viejo

        self._scene_cut_flag = False

        # ── 1. Flujo óptico forward y backward sobre LR ──────────────────
        # Necesitamos ambos sentidos para la comprobación de consistencia.
        flow_fw = cv2.calcOpticalFlowFarneback(
            self._prev_gray, curr_gray, None, **self.fb_kw
        )                            # prev → curr   (H_lr, W_lr, 2)
        flow_bw = cv2.calcOpticalFlowFarneback(
            curr_gray, self._prev_gray, None, **self.fb_kw
        )                            # curr → prev   (H_lr, W_lr, 2)

        # ── 2. Escalar flujo forward → resolución SR ─────────────────────
        H_sr, W_sr = sr_frame.shape[:2]
        flow_sr = cv2.resize(flow_fw, (W_sr, H_sr),
                             interpolation=cv2.INTER_LINEAR) * float(self.scale)

        # ── 3. Warp del frame SR anterior ────────────────────────────────
        sr_warped = _warp_flow(self._prev_sr, flow_sr)

        # ── 4. Máscara de oclusión (consistencia FB, calculada en LR) ────
        occ_lr   = _occlusion_mask(flow_fw, flow_bw, self.scale, self.occ_thr)
        occ_mask = cv2.resize(occ_lr[..., 0], (W_sr, H_sr),
                              interpolation=cv2.INTER_LINEAR)[..., None]

        # ── 5. Blend ponderado ───────────────────────────────────────────
        # alpha = 0 → usa SR puro; alpha = strength → usa warp
        alpha      = self.strength * occ_mask
        sr_f       = sr_frame.astype(np.float32)
        warp_f     = sr_warped.astype(np.float32)
        stabilized = np.clip(alpha * warp_f + (1.0 - alpha) * sr_f, 0, 255).astype(np.uint8)

        # ── 6. Actualizar estado ──────────────────────────────────────────
        # CLAVE: guardamos el SR RAW del frame actual, NO el estabilizado.
        # Guardar el estabilizado creaba un bucle de retroalimentación que
        # acumulaba los errores del warp frame a frame (efecto "derretido").
        self._prev_gray = curr_gray
        self._prev_sr   = sr_frame.copy()

        return stabilized


def load_generator(checkpoint_path: str, device: str) -> AnimeEdgeGenerator:
    """Carga el generador desde un checkpoint o desde un state_dict suelto."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    # El checkpoint puede contener el state_dict directamente o dentro de 'generator'
    state = ckpt.get('generator', ckpt) if isinstance(ckpt, dict) else ckpt

    # Detectar num_rrdb a partir de las claves body.N.*  (evita el mismatch 8 vs 16)
    indices  = {int(k.split('.')[1]) for k in state if k.startswith('body.')}
    num_rrdb = (max(indices) + 1) if indices else 8

    G = AnimeEdgeGenerator(num_features=64, num_rrdb=num_rrdb, growth=32).to(device)
    G.load_state_dict(state)
    G.eval()
    print(f"Modelo cargado desde: {checkpoint_path}  |  num_rrdb={num_rrdb}")
    if isinstance(ckpt, dict) and 'epoch' in ckpt:
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

    # Post-proceso de estilización cel (opcional)
    if args.crisp > 0.0:
        out_img = crispen_celart(
            out_img,
            edge_thr   = args.crisp_edge_thr,
            line_gamma = args.crisp_gamma,
            flatten_d  = args.crisp_flatten_d,
            flatten_sc = args.crisp_flatten_sc,
            dilate_px  = args.crisp_dilate,
            strength   = args.crisp,
        )
        print(f"Crispen aplicado  (strength={args.crisp}, gamma={args.crisp_gamma})")

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
    """Inferencia por lotes sobre frames de video con estabilización temporal opcional."""
    from utils import calcular_error_temporal

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_generator(args.checkpoint, device)

    frames = sorted(glob.glob(os.path.join(args.video_dir, '*.png')))
    if not frames:
        print(f"No se encontraron frames en {args.video_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    img0   = cv2.imread(frames[0])
    h, w   = img0.shape[:2]
    factor = args.scale
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # VideoWriter para la salida estabilizada (y opcionalmente para el raw)
    path_stable = os.path.join(args.output_dir, f'resultado_edgegan_{factor}x_stable.mp4')
    video_stable = cv2.VideoWriter(path_stable, fourcc, 30.0, (w * factor, h * factor))

    # Si la estabilización está activa, generamos también el video raw para comparar
    video_raw = None
    if args.temporal > 0.0:
        path_raw = os.path.join(args.output_dir, f'resultado_edgegan_{factor}x_raw.mp4')
        video_raw = cv2.VideoWriter(path_raw, fourcc, 30.0, (w * factor, h * factor))

    # ── Inicializar estabilizador ─────────────────────────────────────────
    stabilizer = TemporalStabilizer(
        scale      = args.scale,
        strength   = args.temporal,
        occ_thr    = args.temporal_occ_thr,
        cut_thr    = args.temporal_cut_thr,
        winsize    = args.temporal_winsize,
    ) if args.temporal > 0.0 else None

    prev_raw    = None
    prev_stable = None
    raw_errors, stable_errors = [], []
    scene_cuts = 0

    for frame_path in frames:
        img = cv2.imread(frame_path)
        t   = to_tensor(img).to(device)

        # ── Inferencia SR ─────────────────────────────────────────────────
        out_t   = upscale(model, t, scale=args.scale)
        out_raw = to_image(out_t)

        # ── Crispen opcional ──────────────────────────────────────────────
        if args.crisp > 0.0:
            out_raw = crispen_celart(
                out_raw,
                edge_thr   = args.crisp_edge_thr,
                line_gamma = args.crisp_gamma,
                flatten_d  = args.crisp_flatten_d,
                flatten_sc = args.crisp_flatten_sc,
                dilate_px  = args.crisp_dilate,
                strength   = args.crisp,
            )

        # ── Estabilización temporal ───────────────────────────────────────
        if stabilizer is not None:
            cut_before = stabilizer._scene_cut_flag
            out_stable = stabilizer(img, out_raw)
            # El estabilizador marca _scene_cut_flag=True cuando reinicia por corte
            if stabilizer._scene_cut_flag and not cut_before:
                scene_cuts += 1
        else:
            out_stable = out_raw

        # ── Guardar frames y video ────────────────────────────────────────
        name = os.path.basename(frame_path).replace('.png', f'_x{factor}.png')
        cv2.imwrite(os.path.join(args.output_dir, name), out_stable)
        video_stable.write(out_stable)
        if video_raw is not None:
            video_raw.write(out_raw)

        # ── Métricas temporales (acumular) ───────────────────────────────
        if prev_raw is not None:
            raw_errors.append(calcular_error_temporal(out_raw,    prev_raw))
        if prev_stable is not None:
            stable_errors.append(calcular_error_temporal(out_stable, prev_stable))

        prev_raw    = out_raw
        prev_stable = out_stable
        print(f"Frame {name} completado.", end='\r')

    video_stable.release()
    if video_raw:
        video_raw.release()

    # ── Resumen ───────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Video guardado en:  {args.output_dir}")
    if args.temporal > 0.0:
        mean_raw    = np.mean(raw_errors)    if raw_errors    else 0.0
        mean_stable = np.mean(stable_errors) if stable_errors else 0.0
        reduccion   = (1.0 - mean_stable / mean_raw) * 100 if mean_raw > 0 else 0.0
        print(f"\n--- Estabilidad temporal ---")
        print(f"  Sin estabilización : {mean_raw:.4f}")
        print(f"  Con estabilización : {mean_stable:.4f}  ({reduccion:+.1f}%)")
        print(f"  Cortes de escena detectados: {scene_cuts}")
        print(f"  strength={args.temporal}  occ_thr={args.temporal_occ_thr}  "
              f"cut_thr={args.temporal_cut_thr}")
    else:
        if raw_errors:
            print(f"Error temporal medio: {np.mean(raw_errors):.4f}")



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
    # Estilización cel
    p_single.add_argument('--crisp',            default=0.0,  type=float, help='Fuerza del crispen (0=off, 1=completo)')
    p_single.add_argument('--crisp-edge-thr',   default=40,   type=int,   help='Umbral Canny inferior')
    p_single.add_argument('--crisp-gamma',      default=2.2,  type=float, help='Gamma de línea (>1 = más oscuro)')
    p_single.add_argument('--crisp-flatten-d',  default=7,    type=int,   help='Diámetro bilateral')
    p_single.add_argument('--crisp-flatten-sc', default=80.0, type=float, help='Sigma-color bilateral')
    p_single.add_argument('--crisp-dilate',     default=2,    type=int,   help='Dilatación de máscara (px)')

    # Modo video
    p_video = sub.add_parser('video', help='Inferencia sobre directorio de frames')
    p_video.add_argument('--video_dir',   required=True, help='Directorio con frames .png')
    p_video.add_argument('--output_dir',  required=True, help='Directorio de salida')
    p_video.add_argument('--checkpoint',  required=True, help='Ruta al checkpoint .pth')
    p_video.add_argument('--scale',       default=2, type=int, choices=[2, 4])
    # Estilización cel
    p_video.add_argument('--crisp',            default=0.0,  type=float, help='Fuerza del crispen (0=off, 1=completo)')
    p_video.add_argument('--crisp-edge-thr',   default=40,   type=int,   help='Umbral Canny inferior')
    p_video.add_argument('--crisp-gamma',      default=2.2,  type=float, help='Gamma de línea (>1 = más oscuro)')
    p_video.add_argument('--crisp-flatten-d',  default=7,    type=int,   help='Diámetro bilateral')
    p_video.add_argument('--crisp-flatten-sc', default=80.0, type=float, help='Sigma-color bilateral')
    p_video.add_argument('--crisp-dilate',     default=2,    type=int,   help='Dilatación de máscara (px)')

    # Estabilización temporal
    p_video.add_argument('--temporal',          default=0.0,  type=float, help='Fuerza estabilización temporal (0=off, 0.75=recomendado)')
    p_video.add_argument('--temporal-occ-thr',  default=20.0, type=float, help='Umbral de oclusión por flujo (px). Baja → más conservador.')
    p_video.add_argument('--temporal-cut-thr',  default=30.0, type=float, help='Umbral de detección de corte de escena.')
    p_video.add_argument('--temporal-winsize',  default=15,   type=int,   help='Ventana Farneback (15=estándar, 21=movimiento rápido)')

    args = parser.parse_args()

    if args.mode == 'image':
        infer_single(args)
    elif args.mode == 'video':
        infer_video(args)
    else:
        parser.print_help()