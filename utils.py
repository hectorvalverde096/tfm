import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import lpips

# Inicializamos el modelo LPIPS una sola vez para ahorrar memoria y tiempo
_lpips_vgg = None

def to_tensor(img_bgr):
    """Convierte una imagen BGR (OpenCV) a un Tensor de PyTorch (RGB)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

def to_image(tensor):
    """Convierte un Tensor de PyTorch de vuelta a una imagen BGR (OpenCV)."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def calcular_psnr(img1, img2):
    """Calcula el Peak Signal-to-Noise Ratio entre dos imágenes."""
    return cv2.PSNR(img1, img2)

def calcular_ssim(img1, img2):
    """Calcula el Structural Similarity Index (SSIM) entre dos imágenes."""
    # channel_axis=2 indica que la imagen es (H, W, C)
    return ssim(img1, img2, channel_axis=2)

def calcular_lpips(tensor1, tensor2, device="cuda"):
    """Calcula la distancia perceptual LPIPS entre dos tensores [0, 1]."""
    global _lpips_vgg
    if _lpips_vgg is None:
        _lpips_vgg = lpips.LPIPS(net='vgg').to(device)
    
    # LPIPS espera que los tensores estén en el rango [-1, 1]
    t1 = tensor1 * 2 - 1
    t2 = tensor2 * 2 - 1
    with torch.no_grad():
        dist = _lpips_vgg(t1, t2)
    return dist.item()

def medir_nitidez_bordes(img):
    """Mide la nitidez de los bordes usando la varianza del Laplaciano."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calcular_error_temporal(frame_actual, frame_previo):
    """Calcula la diferencia media entre frames para evaluar estabilidad."""
    if frame_previo is None:
        return 0.0
    return np.mean(cv2.absdiff(frame_actual, frame_previo))

def aplicar_clahe(img):
    """Mejora el contraste local usando CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img_clahe = cv2.merge((cl, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)

def calcular_tof(frame_actual, frame_previo):
    """
    Temporal Optical Flow error — mide inconsistencia temporal
    compensando el movimiento. Más justo que MAE en escenas con movimiento.
    Referencia: EDVR, BasicVSR papers.
    """
    prev_gray = cv2.cvtColor(frame_previo, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Warp del frame anterior hacia el actual
    H, W = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                  np.arange(H, dtype=np.float32))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    prev_warped = cv2.remap(frame_previo, map_x, map_y,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return np.mean(cv2.absdiff(frame_actual, prev_warped))