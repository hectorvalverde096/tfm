"""
AnimeSR-EdgeGAN — Generador + Discriminadores para super-resolución de animación
Prioridad: detección y preservación de bordes nítidos
Escala: x2 o x4 (aplicando el generador dos veces)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchmodels


# ─────────────────────────────────────────────
#  BLOQUES BASE
# ─────────────────────────────────────────────

class ResidualDenseBlock(nn.Module):
    """RRDB-style dense block, adaptado para animación (canales más anchos)."""
    def __init__(self, channels=64, growth=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels,           growth, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channels + growth,  growth, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channels + growth*2,growth, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channels + growth*3,growth, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channels + growth*4,channels,3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x  # residual scaling


class RRDB(nn.Module):
    """Bloque Residual-in-Residual Dense (3 RDB anidados)."""
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth)
        self.rdb2 = ResidualDenseBlock(channels, growth)
        self.rdb3 = ResidualDenseBlock(channels, growth)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


# ─────────────────────────────────────────────
#  MÓDULO DE ATENCIÓN A BORDES
# ─────────────────────────────────────────────

class EdgeAttention(nn.Module):
    """
    Genera un mapa de atención basado en bordes (Sobel differentiable)
    y lo aplica como gate sobre las features del generador.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Filtros Sobel — fijos, no entrenables
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],  dtype=torch.float32)
        # shape: (out_ch, in_ch/groups, kH, kW)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))

        # Red ligera que aprende a refinar el mapa de atención
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        # Proyección para aplicar la atención al tensor de features
        self.gate = nn.Conv2d(channels, channels, 1)

    def _sobel_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula magnitud de gradiente sobre el canal de luminancia."""
        # Convertir a escala de grises: media de canales (aproximación rápida)
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, feat: torch.Tensor, img_lr: torch.Tensor) -> torch.Tensor:
        edge_map = self._sobel_magnitude(img_lr)              # (B,1,H,W)
        # Redimensionar al tamaño de las features si difiere
        if edge_map.shape[-2:] != feat.shape[-2:]:
            edge_map = F.interpolate(edge_map, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        attn = self.refine(edge_map)                          # (B,1,H,W)
        return feat + self.gate(feat) * attn                  # modulation


# ─────────────────────────────────────────────
#  GENERADOR PRINCIPAL
# ─────────────────────────────────────────────

class AnimeEdgeGenerator(nn.Module):
    """
    Generador x2 para animación con atención a bordes.
    Para x4: aplicar el modelo dos veces (igual que en pruebas.py).

    Args:
        in_channels:  Canales de entrada (3 = RGB)
        num_features: Ancho del espacio de features (64 recomendado)
        num_rrdb:     Número de bloques RRDB (16 para calidad, 8 para velocidad)
        growth:       Growth rate del bloque denso
    """
    def __init__(self, in_channels=3, num_features=64, num_rrdb=16, growth=32):
        super().__init__()

        # Extracción inicial de features
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        # Cuerpo residual: N bloques RRDB
        self.body = nn.Sequential(*[RRDB(num_features, growth) for _ in range(num_rrdb)])
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)

        # Módulo de atención a bordes (se inyecta tras el cuerpo)
        self.edge_attn = EdgeAttention(num_features)

        # Upsampling x2 con pixel-shuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Salida final
        self.conv_last = nn.Conv2d(num_features, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat                        # skip connection global

        feat = self.edge_attn(feat, x)                # refuerzo de bordes

        feat = self.upsample(feat)
        return self.conv_last(feat)


# ─────────────────────────────────────────────
#  DISCRIMINADORES
# ─────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    Discriminador PatchGAN estándar — evalúa parches 70x70.
    Decide si cada parche local es real o generado.
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=4):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = ndf
        for i in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            stride = 1 if i == n_layers - 1 else 2
            layers += [
                nn.Conv2d(ch, next_ch, 4, stride, 1, bias=False),
                nn.InstanceNorm2d(next_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch
        layers.append(nn.Conv2d(ch, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EdgeDiscriminator(nn.Module):
    """
    Discriminador especializado en bordes.
    Recibe SOLO el mapa de bordes (Canny aproximado con Sobel diferenciable).
    Fuerza al generador a producir bordes realistas y nítidos.
    """
    def __init__(self, ndf=32):
        super().__init__()
        # Sobel diferenciable (igual que EdgeAttention)
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],  dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3))

        self.model = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1, 4, 1, 1),
        )

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = self._edge_map(x)
        return self.model(edge)
