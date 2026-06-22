"""
train.py — Bucle de entrenamiento para AnimeSR-EdgeGAN
Compatible con el pipeline de pruebas.py
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import glob
from model import AnimeEdgeGenerator, PatchDiscriminator, EdgeDiscriminator
from losses import GeneratorLoss, LSGANLoss


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────

class AnimePairDataset(Dataset):
    """
    Espera pares (LR, HR) de imágenes.
    LR debe ser exactamente la mitad de resolución que HR (para x2).

    Estructura esperada:
        data/train/hr/  ← imágenes Ground Truth en alta resolución
        data/train/lr/  ← imágenes degradadas (bicubic downscale x2)

    Si solo tienes frames HR, usa prepare_data.py para generar los LR.
    """
    def __init__(self, hr_dir: str, lr_dir: str, patch_size: int = 256):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.png')))
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
        assert len(self.hr_paths) == len(self.lr_paths), \
            f"Mismatch: {len(self.hr_paths)} HR vs {len(self.lr_paths)} LR"
        self.patch_size = patch_size  # Tamaño del parche HR (LR = patch_size // 2)

    def __len__(self):
        return len(self.hr_paths)

    def _random_crop_pair(self, hr, lr):
        """Recorte alineado LR/HR."""
        h_lr, w_lr = lr.shape[:2]
        ps_lr = self.patch_size // 2
        top  = np.random.randint(0, h_lr - ps_lr)
        left = np.random.randint(0, w_lr - ps_lr)
        lr_crop = lr[top:top+ps_lr, left:left+ps_lr]
        hr_crop = hr[top*2:(top+ps_lr)*2, left*2:(left+ps_lr)*2]
        return hr_crop, lr_crop

    def _augment(self, hr, lr):
        """Data augmentation: flip horizontal + vertical + rotación 90°."""
        if np.random.rand() > 0.5:
            hr, lr = hr[:, ::-1], lr[:, ::-1]
        if np.random.rand() > 0.5:
            hr, lr = hr[::-1, :], lr[::-1, :]
        if np.random.rand() > 0.5:
            hr, lr = np.rot90(hr, 1), np.rot90(lr, 1)
        return hr.copy(), lr.copy()

    def _to_tensor(self, img):
        """BGR uint8 → RGB float32 tensor (C,H,W) en [0,1]."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

    def __getitem__(self, idx):
        hr = cv2.imread(self.hr_paths[idx])
        lr = cv2.imread(self.lr_paths[idx])
        hr, lr = self._random_crop_pair(hr, lr)
        hr, lr = self._augment(hr, lr)
        return self._to_tensor(lr), self._to_tensor(hr)


class MemmapDataset(Dataset):
    """
    Dataset de alta velocidad que lee parches pre-extraidos desde archivos
    binarios np.memmap. Elimina el cuello de botella de I/O de disco.

    Requiere ejecutar prepare_patches.py antes del entrenamiento:
        python prepare_patches.py --out_dir data/patches/train

    Con preload=True carga todo en RAM (~10 GB); usar cuando se tiene
    suficiente RAM libre para eliminar completamente el acceso a disco.
    """
    def __init__(self, patches_dir: str, preload: bool = False):
        self.patches_dir = patches_dir
        self.preload = preload

        meta = np.load(os.path.join(patches_dir, 'meta.npy'))
        self.n   = int(meta[0])
        self.p_hr = int(meta[1])
        self.p_lr = int(meta[2])

        self._open_arrays()

    def _open_arrays(self):
        hr_path = os.path.join(self.patches_dir, 'hr.bin')
        lr_path = os.path.join(self.patches_dir, 'lr.bin')
        if self.preload:
            self.hr = np.fromfile(hr_path, dtype='uint8').reshape(self.n, self.p_hr, self.p_hr, 3)
            self.lr = np.fromfile(lr_path, dtype='uint8').reshape(self.n, self.p_lr, self.p_lr, 3)
        else:
            self.hr = np.memmap(hr_path, dtype='uint8', mode='r',
                                shape=(self.n, self.p_hr, self.p_hr, 3))
            self.lr = np.memmap(lr_path, dtype='uint8', mode='r',
                                shape=(self.n, self.p_lr, self.p_lr, 3))

    # np.memmap cannot be pickled across Windows worker processes.
    # Drop the arrays from state; each worker re-opens them after unpickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        if not self.preload:
            del state['hr']
            del state['lr']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not self.preload:
            self._open_arrays()

    def __len__(self):
        return self.n

    def _augment(self, hr, lr):
        if np.random.rand() > 0.5:
            hr, lr = hr[:, ::-1], lr[:, ::-1]
        if np.random.rand() > 0.5:
            hr, lr = hr[::-1, :], lr[::-1, :]
        if np.random.rand() > 0.5:
            hr, lr = np.rot90(hr, 1), np.rot90(lr, 1)
        return hr.copy(), lr.copy()

    def _to_tensor(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

    def __getitem__(self, idx):
        hr = self.hr[idx].copy()
        lr = self.lr[idx].copy()
        hr, lr = self._augment(hr, lr)
        return self._to_tensor(lr), self._to_tensor(hr)


# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────

CFG = {
    # Datos
    'hr_dir':       'data/train/hr',
    'lr_dir':       'data/train/lr',
    'patch_size':   256,
    'batch_size':   8,
    'num_workers':  4,

    # Modelo
    'num_features': 64,
    'num_rrdb':     16,     # Reducir a 8 si la GPU tiene poca VRAM
    'growth':       32,

    # Entrenamiento
    'epochs':       200,
    'lr_g':         1e-4,
    'lr_d':         1e-4,
    'save_dir':     'checkpoints',
    'save_every':   10,     # guardar checkpoint cada N épocas
    'log_every':    100,    # loguear cada N iteraciones

    # Pérdidas del generador
    'w_pixel':       1.0,
    'w_perceptual':  1.0,
    'w_edge':        2.0,   # Peso alto para priorizar bordes
    'w_frequency':   0.5,
    'w_adversarial': 0.1,

    # Scheduler
    'decay_epoch':   100,   # A partir de aquí, lr decae linealmente
}


# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  → Checkpoint guardado: {path}")


def load_checkpoint(path: str, generator, disc_patch, disc_edge, opt_g, opt_d):
    ckpt = torch.load(path, map_location='cpu')
    generator.load_state_dict(ckpt['generator'])
    disc_patch.load_state_dict(ckpt['disc_patch'])
    disc_edge.load_state_dict(ckpt['disc_edge'])
    opt_g.load_state_dict(ckpt['opt_g'])
    opt_d.load_state_dict(ckpt['opt_d'])
    print(f"  → Checkpoint cargado desde época {ckpt['epoch']}")
    return ckpt['epoch']


# ─────────────────────────────────────────────
#  BUCLE PRINCIPAL
# ─────────────────────────────────────────────

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Entrenando en: {device}")

    # — Dataset & Loader —
    dataset = AnimePairDataset(CFG['hr_dir'], CFG['lr_dir'], CFG['patch_size'])
    loader  = DataLoader(
        dataset,
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} pares | {len(loader)} batches/época")

    # — Modelos —
    G          = AnimeEdgeGenerator(num_features=CFG['num_features'],
                                    num_rrdb=CFG['num_rrdb'],
                                    growth=CFG['growth']).to(device)
    D_patch    = PatchDiscriminator().to(device)
    D_edge     = EdgeDiscriminator().to(device)

    # — Optimizadores —
    opt_g = optim.Adam(G.parameters(),       lr=CFG['lr_g'], betas=(0.9, 0.999))
    opt_d = optim.Adam(
        list(D_patch.parameters()) + list(D_edge.parameters()),
        lr=CFG['lr_d'], betas=(0.9, 0.999)
    )

    # Scheduler: lr constante hasta decay_epoch, luego lineal hasta 0
    def lr_lambda(epoch):
        if epoch < CFG['decay_epoch']:
            return 1.0
        return max(0.0, 1.0 - (epoch - CFG['decay_epoch']) / (CFG['epochs'] - CFG['decay_epoch']))

    sched_g = optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    sched_d = optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)

    # — Pérdidas —
    criterion_g = GeneratorLoss(
        w_pixel=CFG['w_pixel'],
        w_perceptual=CFG['w_perceptual'],
        w_edge=CFG['w_edge'],
        w_frequency=CFG['w_frequency'],
        w_adversarial=CFG['w_adversarial'],
    ).to(device)
    criterion_d = LSGANLoss()

    start_epoch = 0

    # — Entrenamiento —
    for epoch in range(start_epoch, CFG['epochs']):
        G.train(); D_patch.train(); D_edge.train()
        running = {k: 0.0 for k in ['g_total', 'g_edge', 'g_pixel', 'd_patch', 'd_edge']}

        for i, (lr_imgs, hr_imgs) in enumerate(loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # ── Paso del Generador ──────────────────────────────
            sr_imgs = G(lr_imgs)

            disc_fake_patch = D_patch(sr_imgs)
            disc_fake_edge  = D_edge(sr_imgs)

            g_losses = criterion_g(
                sr_imgs, hr_imgs,
                disc_pred_patch=disc_fake_patch,
                disc_pred_edge=disc_fake_edge,
            )

            opt_g.zero_grad()
            g_losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_g.step()

            # ── Paso del Discriminador ──────────────────────────
            # Detach para no backpropagar al generador
            sr_detached = sr_imgs.detach()

            # PatchDiscriminator
            loss_d_patch = (
                criterion_d(D_patch(hr_imgs),    is_real=True) +
                criterion_d(D_patch(sr_detached), is_real=False)
            ) * 0.5

            # EdgeDiscriminator
            loss_d_edge = (
                criterion_d(D_edge(hr_imgs),    is_real=True) +
                criterion_d(D_edge(sr_detached), is_real=False)
            ) * 0.5

            opt_d.zero_grad()
            (loss_d_patch + loss_d_edge).backward()
            opt_d.step()

            # Acumular métricas
            running['g_total']  += g_losses['total'].item()
            running['g_edge']   += g_losses['edge'].item()
            running['g_pixel']  += g_losses['pixel'].item()
            running['d_patch']  += loss_d_patch.item()
            running['d_edge']   += loss_d_edge.item()

            if (i + 1) % CFG['log_every'] == 0:
                n = CFG['log_every']
                print(
                    f"Época [{epoch+1}/{CFG['epochs']}] Iter [{i+1}/{len(loader)}] | "
                    f"G: {running['g_total']/n:.4f} "
                    f"(pixel={running['g_pixel']/n:.4f} edge={running['g_edge']/n:.4f}) | "
                    f"D_patch: {running['d_patch']/n:.4f} D_edge: {running['d_edge']/n:.4f}"
                )
                running = {k: 0.0 for k in running}

        sched_g.step()
        sched_d.step()

        # Checkpoint periódico
        if (epoch + 1) % CFG['save_every'] == 0 or epoch == CFG['epochs'] - 1:
            save_checkpoint({
                'epoch':      epoch + 1,
                'generator':  G.state_dict(),
                'disc_patch': D_patch.state_dict(),
                'disc_edge':  D_edge.state_dict(),
                'opt_g':      opt_g.state_dict(),
                'opt_d':      opt_d.state_dict(),
            }, os.path.join(CFG['save_dir'], f'checkpoint_epoch{epoch+1:04d}.pth'))

    print("Entrenamiento completado.")


if __name__ == '__main__':
    train()
