import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ========== 1. Симуляция 4-битной квантизации ==========
class Fake4BitLinear(nn.Module):
    """
    Эмулирует 4-битный линейный слой.
    Веса хранятся в 4-битном формате (int4), но для вычислений распаковываются в FP16.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Симулируем 4-битные веса: храним как uint8 (0..15), но представляем что это 4 бита
        # В реальности используется bitsandbytes, здесь мы просто эмулируем сжатие.
        self.in_features = in_features
        self.out_features = out_features

        # Сохраняем веса как FP16 (для эмуляции), но с пометкой, что они "4-битные"
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None

        # Инициализация
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Имитируем 4-битное хранение: квантуем веса в 16 уровней
        self._quantize_weights()

    def _quantize_weights(self):
        """Квантизация весов в 4 бита (16 уровней)"""
        with torch.no_grad():
            # Находим min/max
            w_min, w_max = self.weight.min(), self.weight.max()
            # Масштабируем в [0, 15]
            scale = 15.0 / (w_max - w_min + 1e-8)
            quantized = torch.round((self.weight - w_min) * scale)
            # Ограничиваем
            quantized = torch.clamp(quantized, 0, 15).byte()
            # Сохраняем для эмуляции
            self.register_buffer('quantized_weight', quantized)
            self.register_buffer('scale', torch.tensor(scale))
            self.register_buffer('w_min', w_min)

    def forward(self, x):
        # Деквантизация: из 4-бит обратно в FP16
        with torch.no_grad():
            weight_fp = self.quantized_weight.float() / self.scale + self.w_min
        # Для градиентов мы не можем использовать деквантизованные веса напрямую,
        # поэтому в реальном QLoRA градиенты идут только через LoRA адаптеры.
        # Здесь мы просто используем их для forward (эмуляция).
        return F.linear(x, weight_fp, self.bias)


# ========== 2. QLoRA слой (4-битная база + LoRA) ==========
class QLoRALayer(nn.Module):
    """
    Слой, сочетающий 4-битную базовую матрицу и LoRA адаптеры.
    Базовая матрица заморожена и хранится в 4 битах.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        # Базовая 4-битная матрица (заморожена)
        self.base = Fake4BitLinear(in_features, out_features, bias=True)
        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA адаптеры (обучаемые)
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x):
        # 4-битный базовый слой (заморожен)
        base_out = self.base(x)

        # LoRA адаптация
        lora_out = (x @ self.A.T) @ self.B.T
        lora_out = lora_out * (self.alpha / self.rank)

        return base_out + lora_out


# ========== 3. Сравнение QLoRA с обычной LoRA ==========
def compare_qloRa_vs_lora():
    print("=" * 60)
    print("QLoRA: 4-битная LoRA для больших моделей")
    print("=" * 60)

    # Параметры
    in_f, out_f = 64, 128
    rank = 4

    # Обычный линейный слой (FP32)
    linear_fp32 = nn.Linear(in_f, out_f)
    fp32_params = sum(p.numel() for p in linear_fp32.parameters())

    # LoRA-слой (FP32 база + LoRA)
    lora_layer = LoRALayer(linear_fp32, rank=rank)  # используем из прошлого урока
    lora_trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

    # QLoRA-слой (4-битная база + LoRA)
    qlora_layer = QLoRALayer(in_f, out_f, rank=rank)
    qlora_trainable = sum(p.numel() for p in qlora_layer.parameters() if p.requires_grad)
    # Память для 4-битной базы: out_f * in_f * 0.5 байт (4 бита = 0.5 байта)
    base_memory_4bit = out_f * in_f * 0.5 / 1024  # KB
    base_memory_fp32 = out_f * in_f * 4 / 1024  # KB

    print(f"\nСравнение памяти и параметров:")
    print(f"  FP32 линейный слой:        {fp32_params:,} параметров (~{base_memory_fp32:.1f} KB)")
    print(f"  LoRA (FP32 база):          {lora_trainable:,} обучаемых параметров (база заморожена)")
    print(
        f"  QLoRA (4-бит база):        {qlora_trainable:,} обучаемых параметров (база 4-бит, ~{base_memory_4bit:.1f} KB)")
    print(f"  Экономия памяти базы:      {base_memory_fp32 / base_memory_4bit:.1f}x")

    # Имитация обучения на синтетической задаче
    X = torch.randn(100, in_f)
    y = torch.randint(0, 5, (100,))

    # Обучаем QLoRA слой (только LoRA)
    model = nn.Sequential(qlora_layer, nn.Linear(out_f, 5))  # простой классификатор
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    losses = []
    print("\nОбучение QLoRA (обучаются только LoRA параметры)...")
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | loss: {loss.item():.4f}")

    plt.plot(losses)
    plt.xlabel('Epoch');
    plt.ylabel('Loss')
    plt.title('QLoRA обучение')
    plt.grid();
    plt.show()

    print("\nКлючевые преимущества QLoRA:")
    print("  • Экономия памяти: база в 4 битах (8× меньше FP32)")
    print("  • Можно донастраивать модели 65B+ на одной GPU")
    print("  • Точность близка к полной FP16-донастройке")
    print("  • Используется в реальных проектах (например, Llama 2 fine-tuning)")


# ========== 4. Воспроизведение LoRALayer (из прошлого урока) ==========
class LoRALayer(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.rank = rank
        self.alpha = alpha

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        lora_out = lora_out * (self.alpha / self.rank)
        return base_out + lora_out


# ========== 5. Запуск ==========
if __name__ == "__main__":
    compare_qloRa_vs_lora()