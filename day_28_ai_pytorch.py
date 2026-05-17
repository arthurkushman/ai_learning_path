import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple


# ========== 1. Генерация лучей (чистый PyTorch) ==========
def generate_rays(num_rays: int = 1024, img_h: int = 64, img_w: int = 64,
                  radius: float = 1.0, device: torch.device = torch.device('cpu')) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    origins: (N, 3) — позиции камеры (0,0,2)
    directions: (N, 3) — нормализованные направления
    target_colors: (N, 3) — цвет пикселя (истинный цвет для обучения)
    """
    origins = torch.full((num_rays, 3), 0.0, device=device)
    origins[:, 2] = 2.0  # камера в (0,0,2)

    # Случайные углы в пределах ±30 градусов
    theta = (torch.rand(num_rays, device=device) - 0.5) * (torch.pi / 3)
    phi = (torch.rand(num_rays, device=device) - 0.5) * (torch.pi / 3)

    # Направления (не нормированные)
    dirs_xyz = torch.stack([
        torch.tan(theta),
        torch.tan(phi),
        -torch.ones(num_rays, device=device)
    ], dim=-1)
    directions = dirs_xyz / torch.norm(dirs_xyz, dim=-1, keepdim=True)

    # Вычисляем пересечение луча с плоскостью z=0 (приближение для сферы)
    t = -origins[:, 2] / directions[:, 2]  # расстояние до плоскости
    hit_points = origins + directions * t.unsqueeze(-1)
    dist_to_center = torch.norm(hit_points, dim=-1)
    hit_mask = (dist_to_center <= radius) & (t > 0)

    target_colors = torch.zeros(num_rays, 3, device=device)
    # Цвет сферы: зависит от нормализованных координат точки
    norm_pos = hit_points[hit_mask] / radius
    target_colors[hit_mask] = (norm_pos + 1) / 2  # диапазон -1..1 → 0..1

    return origins, directions, target_colors


# ========== 2. NeRF модель ==========
class MiniNeRF(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = input_dim * 6 + 3  # позиционное кодирование L=3
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 4
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def positional_encoding(self, x: torch.Tensor, L: int = 3) -> torch.Tensor:
        pe = [x]
        for i in range(L):
            freq = 2 ** i
            pe.append(torch.sin(freq * torch.pi * x))
            pe.append(torch.cos(freq * torch.pi * x))
        return torch.cat(pe, dim=-1)

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.positional_encoding(points)
        out = self.net(encoded)
        rgb = torch.sigmoid(out[..., :3])
        density = torch.relu(out[..., 3:4])
        return rgb, density


# ========== 3. Объёмный рендеринг ==========
def volume_rendering(nerf: nn.Module, origins: torch.Tensor, directions: torch.Tensor,
                     near: float = 0.5, far: float = 2.0, num_samples: int = 32) -> torch.Tensor:
    device = origins.device
    batch_size = origins.shape[0]
    t_vals = torch.linspace(near, far, num_samples, device=device)
    t_vals = t_vals + (torch.rand_like(t_vals) * (far - near) / num_samples)

    pts = origins[:, None, :] + directions[:, None, :] * t_vals[None, :, None]  # (N, S, 3)
    flat_pts = pts.view(-1, 3)
    rgb_flat, density_flat = nerf(flat_pts)
    rgb = rgb_flat.view(batch_size, num_samples, 3)
    density = density_flat.view(batch_size, num_samples, 1)

    delta = t_vals[1] - t_vals[0]
    alpha = 1.0 - torch.exp(-density * delta)
    cum_prod = torch.cumprod(1 - alpha + 1e-10, dim=1)
    T = torch.cat([torch.ones_like(alpha[:, :1]), cum_prod[:, :-1]], dim=1)
    color = torch.sum(T * alpha * rgb, dim=1)
    return color


# ========== 4. Обучение ==========
def train_nerf(epochs: int = 300, batch_size: int = 2048, lr: float = 5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nerf = MiniNeRF().to(device)
    optimizer = optim.Adam(nerf.parameters(), lr=lr)
    loss_history = []

    print(f"Обучение NeRF на {device}...")
    for epoch in range(epochs):
        origins, dirs, target = generate_rays(batch_size, device=device)
        pred = volume_rendering(nerf, origins, dirs)
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | loss: {loss.item():.6f}")

    # Визуализация результата
    nerf.eval()
    with torch.no_grad():
        H, W = 64, 64
        origins = torch.full((H * W, 3), 0.0, device=device);
        origins[:, 2] = 2.0
        dirs_grid = torch.zeros(H * W, 3, device=device)
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                u = (j / W) * 2 - 1
                v = (i / H) * 2 - 1
                d = torch.tensor([u, v, -1.0], device=device)
                dirs_grid[idx] = d / torch.norm(d)
        rendered = volume_rendering(nerf, origins, dirs_grid)
        rendered = rendered.view(H, W, 3).cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1);
    plt.plot(loss_history);
    plt.title('Loss');
    plt.grid()
    plt.subplot(1, 2, 2);
    plt.imshow(rendered.clip(0, 1));
    plt.title('NeRF Output');
    plt.axis('off')
    plt.tight_layout();
    plt.show()
    return nerf


if __name__ == "__main__":
    print("=" * 60)
    print("День 28: NeRF — Neural Radiance Fields")
    print("=" * 60)
    print("Идея: MLP учится отображать (x,y,z) -> (r,g,b,плотность)")
    print("Объёмный рендеринг интегрирует цвета вдоль лучей.")
    nerf_model = train_nerf(epochs=300, batch_size=2048)