import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. Подготовка данных ==========
# Наши исходные данные
X_numpy = np.array([[1, 2, 3], 
                    [4, 5, 6]], dtype=np.float32)  # (2, 3)
y_numpy = np.array([[100., 150., 200.]], dtype=np.float32)  # (1, 3)

# Преобразуем в тензоры PyTorch
X_torch = torch.from_numpy(X_numpy.T)  # Транспонируем: (3, 2)
y_torch = torch.from_numpy(y_numpy.T)  # Транспонируем: (3, 1)

print("Исходные данные:")
print("X shape:", X_torch.shape)  # (3, 2) - 3 объекта, 2 признака
print("y shape:", y_torch.shape)  # (3, 1) - 3 объекта, 1 таргет
print()

# Нормализация
X_mean = X_torch.mean(dim=0, keepdim=True)
X_std = X_torch.std(dim=0, keepdim=True)
X_normalized = (X_torch - X_mean) / (X_std + 1e-8)

y_mean = y_torch.mean()
y_std = y_torch.std()
y_normalized = (y_torch - y_mean) / y_std

print("После нормализации:")
print("X_normalized shape:", X_normalized.shape)
print("y_normalized shape:", y_normalized.shape)
print()

# ========== 2. Создание модели ==========
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Первый слой: 2 входа -> 3 выхода
        self.fc1 = nn.Linear(2, 3)
        # Второй слой: 3 входа -> 1 выход
        self.fc2 = nn.Linear(3, 1)
        
        # Инициализация весов как в нашей numpy-реализации
        self._init_weights()
    
    def _init_weights(self):
        """Инициализируем веса как в нашем примере"""
        with torch.no_grad():
            # fc1.weight: размер (3, 2) в PyTorch
            # Наши веса W1: [[-11, 22], [33, -44], [55, 66]]
            self.fc1.weight.copy_(torch.tensor([
                [-11., 22.],   # первый нейрон
                [33., -44.],   # второй нейрон
                [55., 66.]     # третий нейрон
            ]))
            
            # fc1.bias: размер (3,)
            self.fc1.bias.copy_(torch.tensor([10., -20., 30.]))
            
            # fc2.weight: размер (1, 3) в PyTorch
            self.fc2.weight.copy_(torch.tensor([[0.5, -1.0, 0.2]]))
            
            # fc2.bias: размер (1,)
            self.fc2.bias.copy_(torch.tensor([0.1]))
    
    def forward(self, x):
        # x: (batch_size, 2)
        x = torch.relu(self.fc1(x))  # -> (batch_size, 3)
        x = self.fc2(x)              # -> (batch_size, 1)
        return x

# Создаем модель
model = TwoLayerNet()
print("Модель создана:")
print(model)
print()

# Проверка forward pass
with torch.no_grad():
    test_output = model(X_normalized)
    print(f"Тест forward pass:")
    print(f"Вход: {X_normalized.shape} -> Выход: {test_output.shape}")
    print(f"Предсказания (первые 3): {test_output[:3].flatten()}")
print()

# ========== 3. Обучение модели ==========
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

losses_torch = []
print("Начинаем обучение...")

for epoch in range(1000):
    # Forward pass
    predictions = model(X_normalized)
    
    # Вычисление loss
    loss = criterion(predictions, y_normalized)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Обновление весов
    optimizer.step()
    
    losses_torch.append(loss.item())
    
    if epoch % 20 == 0:
        print(f"Эпоха {epoch:3d}: loss = {loss.item():.6f}")

print(f"Эпоха {99:3d}: loss = {losses_torch[-1]:.6f}")
print()

# ========== 4. Предсказания после обучения ==========
with torch.no_grad():
    pred_normalized = model(X_normalized)
    pred_original = pred_normalized * y_std + y_mean

print("Результаты после обучения:")
print("Предсказания PyTorch:", pred_original.T)
print("Истинные значения:", y_torch.T)
print()

# ========== 5. Сравнение с нашей numpy-реализацией ==========
# Загружаем результаты из нашей реализации
from io import StringIO
import sys

# Ваши результаты из предыдущего кода
our_predictions = np.array([[113.4, 143.1, 172.8]])  # Пример из вашего кода
our_final_loss = 320.45  # Пример из вашего кода

print("Сравнение с нашей реализацией:")
print("PyTorch предсказания:", pred_original.T.numpy())
print("Наши предсказания:", our_predictions)
print("Истинные значения:", y_numpy)
print()
print("Потери:")
print("PyTorch финальный loss:", losses_torch[-1])
print("Наш финальный loss:", our_final_loss)

# ========== 6. Визуализация ==========
plt.figure(figsize=(12, 4))

# График обучения
plt.subplot(1, 2, 1)
plt.plot(losses_torch)
plt.yscale('log')
plt.xlabel('Эпоха')
plt.ylabel('Loss (log scale)')
plt.title('Обучение PyTorch модели')
plt.grid(True)

# График предсказаний
plt.subplot(1, 2, 2)
plt.scatter(y_torch.numpy(), pred_original.numpy(), alpha=0.6, label='PyTorch')
if 'our_predictions' in locals():
    plt.scatter(y_numpy.flatten(), our_predictions.flatten(), alpha=0.6, label='Наша реализация')
plt.plot([y_torch.min(), y_torch.max()], [y_torch.min(), y_torch.max()], 'r--', label='Идеально')
plt.xlabel('Истинные значения')
plt.ylabel('Предсказания')
plt.title('Предсказания vs Истина')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ========== 7. Проверка градиентов ==========
print("\nПроверка градиентов:")
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"{name}: grad mean = {param.grad.mean().item():.6f}, grad std = {param.grad.std().item():.6f}")