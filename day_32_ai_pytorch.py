import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ========== 1. LoRA слой ==========
class LoraLayer(nn.Module):
    """
     Low-Rank Adaptation слой.
     Оборачивает существующий линейный слой.
     """
    def __init__(self, linear_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear_layer = linear_layer
        self.rank = rank
        self.alpha = alpha

        # Исходные веса замораживаем
        self.linear_layer.weight.requires_grad = False
        self.linear_layer.bias.requires_grad = False

        # LoRA матрицы: A (ранг → вход) и B (выход → ранг)
        in_features = self.linear_layer.in_features
        out_features = self.linear_layer.out_features

        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, rank) * 0.01)

    def forward(self, x):
        base_out = self.linear_layer(x)

        # LoRA адаптация: B * A * x
        lora_out = (x @ self.A.T) @ self.B.T
        lora_out = lora_out * (self.alpha / self.rank)

        return base_out + lora_out

# ========== 2. Простая модель для демонстрации ==========
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ========== 3. Замена слоёв на LoRA ==========
def apply_lora(model: nn.Module, rank: int = 4):
    """Заменяет линейные слои на LoRA-версии"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lora_layer = LoraLayer(module, rank=rank)
            setattr(model, name, lora_layer)
    return model


# ========== 4. Демонстрация обучения с LoRA ==========
def demo_lora_training():
    print("=" * 60)
    print("LoRA: Low-Rank Adaptation")
    print("=" * 60)

    # Создаём базовую модель
    base_model = SimpleModel()
    total_params = sum(p.numel() for p in base_model.parameters())

    # Создаём LoRA-версию
    lora_model = SimpleModel()
    apply_lora(lora_model, rank=4)

    # Считаем параметры
    frozen_params = sum(p.numel() for p in lora_model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

    print(f"\nИсходная модель: {total_params:,} параметров")
    print(f"LoRA модель: {trainable_params:,} обучаемых параметров")
    print(f"Сжатие: {total_params / trainable_params:.1f}x")
    print(f"Доля обучаемых: {trainable_params / total_params * 100:.1f}%")

    # Генерируем синтетические данные
    X = torch.randn(100, 10)
    y = torch.randint(0, 5, (100,))

    # Обучаем только LoRA параметры
    optimizer = torch.optim.Adam(lora_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    losses = []
    print("\nОбучение с LoRA (обучаются только ~5% параметров)...")
    for epoch in range(100):
        optimizer.zero_grad()
        pred = lora_model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | loss: {loss.item():.4f}")

    # Визуализация
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LoRA обучение')
    plt.grid()
    plt.show()

    # Показываем структуру LoRA
    print("\nСтруктура LoRA слоя:")
    for name, module in lora_model.named_modules():
        if isinstance(module, LoRALayer):
            print(f"  {name}")
            print(f"    A: {module.A.shape} | B: {module.B.shape}")

    return lora_model

# ========== 5. Практический пример: донастройка для новой задачи ==========
def lora_for_new_task():
    """Пример использования LoRA для быстрой адаптации"""
    print("\n" + "="*60)
    print("Пример: донастройка одной модели на разные задачи")
    print("="*60)

    base_model = SimpleModel()

    # Создаём две независимые LoRA адаптации
    model_task_a = SimpleModel()
    model_task_b = SimpleModel()
    apply_lora(model_task_a, rank=4)
    apply_lora(model_task_b, rank=4)

    # Сохраняем только LoRA веса (A и B)
    def save_lora_weights(model: nn.Module, path: str):
        lora_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                lora_weights[name] = module.A.data
                lora_weights[name] = module.B.data
        torch.save(lora_weights, path)
        print(f"Сохранено LoRA: {path} ({len(lora_weights)} матриц)")

    # Имитируем обучение на двух разных задачах
    # Задача А: XOR-like данные
    X_a = torch.randn(100, 10)
    y_a = (X_a[:, 0] * X_a[:, 1] > 0).long()

    # Задача B: сумма > 0
    X_b = torch.randn(100, 10)
    y_b = (X_b.sum(dim=1) > 0).long()

    # Обучаем обе версии
    print("\nАдаптация к задаче А...")
    opt_a = torch.optim.Adam(model_task_a.parameters(), lr=0.01)
    for _ in range(30):
        opt_a.zero_grad()
        loss_a = F.cross_entropy(model_task_a(X_a), y_a)
        loss_a.backward()
        opt_a.step()

    print("Адаптация к задаче Б...")
    opt_b = torch.optim.Adam(model_task_b.parameters(), lr=0.01)
    for _ in range(30):
        opt_b.zero_grad()
        loss_b = F.cross_entropy(model_task_b(X_b), y_b)
        loss_b.backward()
        opt_b.step()

    # Сохраняем обе адаптации
    save_lora_weights(model_task_a, "lora_task_a.pt")
    save_lora_weights(model_task_b, "lora_task_b.pt")

    print("\nОбе адаптации сохранены (каждая ~2KB)")
    print("Можно переключать задачи, подгружая разные LoRA веса!")

# ========== 6. Запуск ==========
if __name__ == "__main__":
    # Основная демонстрация
    lora_model = demo_lora_training()

    # Пример переключения задач
    lora_for_new_task()

    print("\n" + "=" * 60)
    print("Ключевые концепции LoRA:")
    print("""
    1. Замораживаем основную модель (99% параметров)
    2. Добавляем маленькие матрицы A и B (1% параметров)
    3. Обучаем только их
    4. Результат: точность близка к full fine-tuning, но с 20× экономией
    """)
    print("Где используется LoRA?")
    print("  • Llama 2/3 fine-tuning (QLoRA)")
    print("  • Stable Diffusion (DreamBooth, Textual Inversion)")
    print("  • GPT-3.5/4 (параметрические адаптации)")