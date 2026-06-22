"""
losses.py — Funciones de pérdida para AnimeSR-EdgeGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ─────────────────────────────────────────────
#  PERCEPTUAL LOSS (VGG)
# ─────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Pérdida perceptual usando capas intermedias de VGG-19.
    Captura estructuras de alto nivel sin difuminar bordes.
    """
    def __init__(self, layer_ids=(2, 7, 16, 25), weights=(0.1, 0.1, 1.0, 1.0)):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList()
        prev = 0
        for idx in layer_ids:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:idx+1]))
            prev = idx + 1
        for p in self.parameters():
            p.requires_grad = False
        self.weights = weights

        # Normalización ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = self._normalize(pred.clamp(0,1))
        target = self._normalize(target.clamp(0,1))
        loss = 0.0
        for w, layer in zip(self.weights, self.slices):
            pred   = layer(pred)
            target = layer(target)
            loss  += w * F.l1_loss(pred, target)
        return loss


# ─────────────────────────────────────────────
#  EDGE LOSS (Sobel diferenciable)
# ─────────────────────────────────────────────

class EdgeLoss(nn.Module):
    """
    Pérdida directa sobre mapas de bordes.
    Maximiza la similitud entre los bordes del output y el GT.
    Es la pérdida más importante para animación.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],  dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        # 1e-6 underflows to 0 in FP16 (min ~6.1e-5); use 1e-4 to keep sqrt stable
        return torch.sqrt(gx**2 + gy**2 + 1e-4)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(pred), self._edges(target))


# ─────────────────────────────────────────────
#  FREQUENCY LOSS (FFT)
# ─────────────────────────────────────────────

class FrequencyLoss(nn.Module):
    """
    Pérdida en el dominio de frecuencias.
    Penaliza la pérdida de altas frecuencias (detalles finos, textura).
    Complementa la perceptual loss, que es más sensible a estructuras.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT overflows silently in FP16; cast to FP32 regardless of autocast context
        pred_fft   = torch.fft.rfft2(pred.float(),   norm='ortho')
        target_fft = torch.fft.rfft2(target.float(), norm='ortho')
        loss = F.l1_loss(pred_fft.abs(), target_fft.abs())
        return loss


# ─────────────────────────────────────────────
#  ADVERSARIAL LOSS (LSGAN)
# ─────────────────────────────────────────────

class LSGANLoss(nn.Module):
    """
    Least-Squares GAN loss — más estable que BCE, menos mode collapse.
    Úsalo tanto para el generador como el discriminador.
    """
    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)


# ─────────────────────────────────────────────
#  PÉRDIDA TOTAL DEL GENERADOR
# ─────────────────────────────────────────────

class GeneratorLoss(nn.Module):
    """
    Combina todas las pérdidas del generador con pesos configurables.

    Pesos por defecto recomendados para animación:
      - pixel:       1.0   (L1 directo, da estabilidad base)
      - perceptual:  1.0   (estructura de alto nivel)
      - edge:        2.0   (PRIORIDAD: bordes nítidos)
      - frequency:   0.5   (altas frecuencias / detalles)
      - adversarial: 0.1   (no dominar al principio del entrenamiento)
    """
    def __init__(
        self,
        w_pixel=1.0,
        w_perceptual=1.0,
        w_edge=2.0,
        w_frequency=0.5,
        w_adversarial=0.1,
    ):
        super().__init__()
        self.w_pixel       = w_pixel
        self.w_perceptual  = w_perceptual
        self.w_edge        = w_edge
        self.w_frequency   = w_frequency
        self.w_adversarial = w_adversarial

        self.pixel_loss      = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss() if w_perceptual > 0 else None
        self.edge_loss       = EdgeLoss()
        self.freq_loss       = FrequencyLoss() if w_frequency > 0 else None
        self.adv_loss        = LSGANLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        disc_pred_patch=None,    # salida del PatchDiscriminator para el pred
        disc_pred_edge=None,     # salida del EdgeDiscriminator para el pred
    ) -> dict:
        losses = {}

        losses['pixel'] = self.w_pixel * self.pixel_loss(pred, target)
        # Skip expensive/unstable losses when their weight is 0.
        # Calling VGG in fp16 with w=0 produces 0*inf=NaN and wastes ~500ms/batch.
        if self.w_perceptual > 0:
            losses['perceptual'] = self.w_perceptual * self.perceptual_loss(pred, target)
        losses['edge'] = self.w_edge * self.edge_loss(pred, target)
        if self.w_frequency > 0:
            losses['frequency'] = self.w_frequency * self.freq_loss(pred, target)

        if disc_pred_patch is not None:
            losses['adv_patch'] = self.w_adversarial * self.adv_loss(disc_pred_patch, True)
        if disc_pred_edge is not None:
            losses['adv_edge']  = self.w_adversarial * self.adv_loss(disc_pred_edge,  True)

        losses['total'] = sum(losses.values())
        return losses
