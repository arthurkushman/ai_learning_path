import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ========== 1. NeRF модель (как в прошлом уроке) ==========
class MiniNeRF(nn.Module):
    def __init__(self, input_dim=3, hidden=128, layers=4):
        super().__init__()
        ins = input_dim * 6 + 3  # позиционное кодирование L=3
        self.net = nn.Sequential(
            nn.Linear(ins, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4)  # RGB + density
        )

    def pe(self, x, L=3):
        pe = [x]
        for i in range(L):
            f = 2 ** i
            pe.append(torch.sin(f * torch.pi * x))
            pe.append(torch.cos(f * torch.pi * x))
        return torch.cat(pe, dim=-1)

    def forward(self, pts):
        enc = self.pe(pts)
        out = self.net(enc)
        rgb = torch.sigmoid(out[..., :3])
        density = torch.relu(out[..., 3:4])
        return rgb, density


def volume_rendering(nerf, origins, dirs, near=0.5, far=2.0, N=32):
    device = origins.device
    t = torch.linspace(near, far, N, device=device)
    t = t + (torch.rand_like(t) * (far - near) / N)
    pts = origins[:, None, :] + dirs[:, None, :] * t[None, :, None]
    flat = pts.view(-1, 3)
    rgb_flat, dens_flat = nerf(flat)
    rgb = rgb_flat.view(len(origins), N, 3)
    dens = dens_flat.view(len(origins), N, 1)
    delta = t[1] - t[0]
    alpha = 1 - torch.exp(-dens * delta)
    cum = torch.cumprod(1 - alpha + 1e-10, dim=1)
    T = torch.cat([torch.ones_like(alpha[:, :1]), cum[:, :-1]], dim=1)
    color = torch.sum(T * alpha * rgb, dim=1)
    return color


# ========== 2. Эмуляция диффузионной модели (Score Distillation) ==========
class PseudoDiffusion:
    """
    Имитация предобученной диффузионной модели.
    В реальности здесь должна быть Stable Diffusion, возвращающая градиент.
    Мы подменяем: хотим, чтобы изображения были похожи на целевое понятие (например, "красная сфера").
    """

    def __init__(self, target_text="red sphere", device='cpu'):
        self.device = device
        # Эмулируем "понятие": для целевого текста мы хотим, чтобы изображение было красным.
        self.target_color = torch.tensor([1.0, 0.2, 0.2], device=device)  # красноватый

    def sds_loss_grad(self, image):
        """
        Имитируем Score Distillation Sampling (SDS) loss.
        Реальный метод добавляет шум к изображению и просит диффузионную модель предсказать шум.
        Здесь: просто MSE между изображением и целевым цветом, масштабированное.
        """
        # image: (H,W,3)
        img_mean = image.mean(dim=(0, 1))  # средний цвет
        loss = torch.mean((img_mean - self.target_color) ** 2)
        # Градиент по входному изображению (будет передан в NeRF через рендеринг)
        grad = torch.autograd.grad(loss, image, create_graph=True)[0]
        return grad


# ========== 3. Оптимизация NeRF через SDS ==========
def train_dreamfusion(epochs=200, batch_size=1024):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nerf = MiniNeRF().to(device)
    optimizer = optim.Adam(nerf.parameters(), lr=5e-4)

    # Целевой текст (задаём эмуляцией)
    diffusion = PseudoDiffusion(target_text="red sphere", device=device)

    # Камера будет вращаться вокруг центра
    def random_camera_rays(num_rays):
        # Камера на сфере радиусом 2
        theta = torch.rand(num_rays, device=device) * 2 * torch.pi
        phi = torch.rand(num_rays, device=device) * torch.pi - torch.pi / 2
        cam_pos = torch.stack([
            torch.cos(theta) * torch.cos(phi),
            torch.sin(theta) * torch.cos(phi),
            torch.sin(phi)
        ], dim=-1) * 2.0
        # Направление в начало координат
        dirs = -cam_pos / torch.norm(cam_pos, dim=-1, keepdim=True)
        return cam_pos, dirs

    loss_history = []
    print(f"Запуск DreamFusion на {device}")
    print("Цель: сгенерировать 3D-объект, похожий на красную сферу")

    for epoch in range(epochs):
        optimizer.zero_grad()
        # Рендерим изображение со случайного ракурса (только один луч = одно изображение?)
        # В реальном DreamFusion рендерят целое изображение (64x64) и считают SDS loss по пикселям.
        # Для упрощения: создадим маленькое "изображение" из 32x32 лучей.
        H, W = 32, 32
        origins = torch.zeros(H * W, 3, device=device)
        dirs = torch.zeros(H * W, 3, device=device)
        # Камера на фиксированном расстоянии, но вращаемся
        theta = epoch / epochs * 2 * torch.pi  # плавное вращение
        cam = torch.tensor([np.cos(theta), np.sin(theta), 0.5], device=device) * 2.0
        cam[2] = 1.0
        # Направления на пиксели
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                u = (j / W) * 2 - 1
                v = (i / H) * 2 - 1
                dir = torch.tensor([u, v, -1.0], device=device)
                dir = dir / torch.norm(dir)
                dirs[idx] = dir
                origins[idx] = cam
        # Рендерим
        image = volume_rendering(nerf, origins, dirs, near=0.5, far=3.0)
        image = image.view(H, W, 3)
        # Вычисляем SDS градиент (в реальности диффузия)
        grad_img = diffusion.sds_loss_grad(image)  # (H,W,3)
        # Передаём градиент в параметры NeRF через обратное распространение
        # Для этого нужно сделать backward от loss, который сгенерирует grad_img
        # Но grad_img уже есть, мы можем применить его к image, а затем backward
        fake_loss = (image * grad_img.detach()).sum()
        fake_loss.backward()
        optimizer.step()

        loss_history.append(fake_loss.item())
        if epoch % 50 == 0:
            # Покажем текущий рендер
            with torch.no_grad():
                vis = image.detach().cpu().numpy().clip(0, 1)
                plt.imshow(vis)
                plt.title(f"Epoch {epoch}")
                plt.axis('off')
                plt.show(block=False)
                plt.pause(0.1)
            print(f"Epoch {epoch:3d} | loss: {fake_loss.item():.4f}")

    # Финальный рендер с разных ракурсов
    nerf.eval()
    with torch.no_grad():
        angles = [0, 90, 180, 270]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for ax, deg in zip(axes, angles):
            rad = deg * torch.pi / 180
            cam = torch.tensor([np.cos(rad), np.sin(rad), 1.0], device=device) * 2.0
            origins = cam.expand(H * W, -1)
            dirs = torch.zeros(H * W, 3, device=device)
            for i in range(H):
                for j in range(W):
                    idx = i * W + j
                    u = (j / W) * 2 - 1;
                    v = (i / H) * 2 - 1
                    d = torch.tensor([u, v, -1.0], device=device)
                    dirs[idx] = d / torch.norm(d)
            img = volume_rendering(nerf, origins, dirs, near=0.5, far=3.0)
            ax.imshow(img.view(H, W, 3).cpu().numpy().clip(0, 1))
            ax.set_title(f"{deg}°")
            ax.axis('off')
        plt.suptitle("DreamFusion: 3D object from text 'red sphere'")
        plt.tight_layout()
        plt.show()
    return nerf


if __name__ == "__main__":
    print("=" * 60)
    print("День 29: DreamFusion — генерация 3D из текста")
    print("=" * 60)
    print("""
    Идея: Используем диффузионную модель как судья.
    NeRF рендерит изображение → диффузия говорит, похоже ли на текст → градиент идёт в NeRF.
    Результат: 3D-модель, которая со всех сторон похожа на описание.
    """)
    model = train_dreamfusion(epochs=200)