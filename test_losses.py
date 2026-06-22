"""
test_losses.py — Verifica el fix de G=inf antes de lanzar entrenamiento completo.
Crea tensores sinteticos, corre 5 pasos de generador y mide velocidad.
Ejecutar: python test_losses.py
"""
import time, sys
import torch
from torch.cuda.amp import GradScaler

sys.path.insert(0, '.')
from model import AnimeEdgeGenerator, PatchDiscriminator, EdgeDiscriminator
from losses import GeneratorLoss, LSGANLoss

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = DEVICE == 'cuda'
BATCH   = 4      # pequeno para el test
LR_SIZE = 64     # 64x64 LR -> 128x128 SR (rapido)

print(f"Device: {DEVICE}  AMP: {USE_AMP}")

G       = AnimeEdgeGenerator(num_features=64, num_rrdb=8, growth=32).to(DEVICE)
D_patch = PatchDiscriminator().to(DEVICE)
D_edge  = EdgeDiscriminator().to(DEVICE)

criterion_g = GeneratorLoss(
    w_pixel=1.0, w_perceptual=0.0,  # VGG desactivado
    w_edge=2.0,  w_frequency=0.5, w_adversarial=0.1,
).to(DEVICE)
criterion_d = LSGANLoss()
opt_g = torch.optim.Adam(G.parameters(), lr=1e-4)
scaler = GradScaler(enabled=USE_AMP)

print("\nEjecutando 5 pasos de prueba...")
for step in range(5):
    t0 = time.time()

    lr_b = torch.rand(BATCH, 3, LR_SIZE,      LR_SIZE,      device=DEVICE)
    hr_b = torch.rand(BATCH, 3, LR_SIZE * 2,  LR_SIZE * 2,  device=DEVICE)

    opt_g.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', enabled=USE_AMP):
        sr = G(lr_b)
        losses = criterion_g(sr, hr_b,
                             disc_pred_patch=D_patch(sr),
                             disc_pred_edge=D_edge(sr))

    scaler.scale(losses['total']).backward()
    scaler.step(opt_g)
    scaler.update()

    elapsed = time.time() - t0
    total_val = losses['total'].item()
    ok = "OK" if (not torch.isnan(losses['total']) and not torch.isinf(losses['total'])) else "FALLO"
    print(f"  Paso {step+1}: G_total={total_val:.4f}  edge={losses['edge'].item():.4f}"
          f"  tiempo={elapsed*1000:.0f}ms  [{ok}]")

print("\nResultado esperado: G_total finito (no inf/nan), tiempo <500ms por paso.")
print("Si ves FALLO, el problema persiste. Si ves OK, el fix funciona.")