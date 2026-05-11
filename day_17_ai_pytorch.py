import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ========== 1. CNN модель для MNIST (ИСПРАВЛЕННАЯ) ==========
class CNNMnist(nn.Module):
    def __init__(self):
        super().__init__()
        # Свёрточные слои
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # BatchNorm слои
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Пулинг слои
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Полносвязные слои
        # ИСПРАВЛЕНИЕ: после 3 пулингов: 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Сохраняем оригинальный размер для отладки
        # print(f"Input shape: {x.shape}")

        # Первый свёрточный блок
        x = self.conv1(x)  # [batch, 32, 28, 28]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # [batch, 32, 14, 14]
        # print(f"After conv1+pool: {x.shape}")

        # Второй свёрточный блок
        x = self.conv2(x)  # [batch, 64, 14, 14]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # [batch, 64, 7, 7]
        # print(f"After conv2+pool: {x.shape}")

        # Третий свёрточный блок
        x = self.conv3(x)  # [batch, 128, 7, 7]
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # [batch, 128, 3, 3] (7/2=3.5 -> округляется ВНИЗ до 3)
        # print(f"After conv3+pool: {x.shape}")

        # Выпрямляем для полносвязных слоёв
        x = x.view(x.size(0), -1)  # Используем x.size(0) для динамического batch_size
        # print(f"After flatten: {x.shape}")

        # Полносвязные слои
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


# ========== 2. Упрощённая CNN для тестирования ==========
class SimpleCNN(nn.Module):
    """Простая CNN для быстрого тестирования"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # После 1 пулинга: 28->14->7
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, 7, 7]
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ========== 3. Функция для проверки размерностей ==========
def test_model_shapes():
    """Тестируем, что размерности сходятся"""
    print("Тестирование размерностей модели...")

    # Создаём тестовый батч
    test_batch = torch.randn(64, 1, 28, 28)  # batch_size=64, channels=1, height=28, width=28
    print(f"Входной тензор: {test_batch.shape}")

    # Тестируем обе модели
    models = {
        "CNNMnist": CNNMnist(),
        "SimpleCNN": SimpleCNN()
    }

    for name, model in models.items():
        print(f"\n{name}:")
        model.eval()
        with torch.no_grad():
            output = model(test_batch)
            print(f"  Вход: {test_batch.shape}")
            print(f"  Выход: {output.shape}")

            # Считаем параметры
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Всего параметров: {total_params:,}")


# ========== 4. Обучение CNN (ИСПРАВЛЕННАЯ) ==========
def train_cnn_simple():
    """Упрощённое обучение CNN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуется устройство: {device}")

    # Загрузка данных
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2
    )

    # Используем SimpleCNN (она точно работает)
    model = SimpleCNN().to(device)

    print(f"\nМодель: SimpleCNN")
    print(model)

    # Loss и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Функции обучения
    def train_epoch():
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Проверяем размерности
            # print(f"Batch - images: {images.shape}, labels: {labels.shape}")

            # Forward
            outputs = model(images)
            # print(f"Outputs shape: {outputs.shape}")

            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Статистика
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(train_loader), 100 * correct / total

    def evaluate(loader):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(loader), 100 * correct / total

    # Обучение
    num_epochs = 5  # Меньше эпох для тестирования
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    print("\nНачинаем обучение CNN...")
    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch + 1}/{num_epochs}")

        # Обучение
        train_loss, train_acc = train_epoch()
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Тестирование
        test_loss, test_acc = evaluate(test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Визуализация фильтров первого слоя
    visualize_filters(model)

    # Графики обучения
    plot_training_curves(train_losses, test_losses, train_accs, test_accs)

    print(f"\nИтоговая точность CNN: {test_accs[-1]:.2f}%")

    return model, test_accs[-1]


# ========== 5. Визуализация фильтров ==========
def visualize_filters(model):
    """Визуализация фильтров первого свёрточного слоя"""
    print("\nВизуализация фильтров первого свёрточного слоя...")

    weights = model.conv1.weight.detach().cpu()

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(min(16, weights.shape[0])):  # Первые 16 фильтров
        filter_img = weights[i].squeeze()  # [3, 3]
        axes[i].imshow(filter_img, cmap='gray')
        axes[i].set_title(f'Фильтр {i + 1}')
        axes[i].axis('off')

    plt.suptitle('Фильтры первого свёрточного слоя', fontsize=14)
    plt.tight_layout()
    plt.show()


# ========== 6. Графики обучения ==========
def plot_training_curves(train_losses, test_losses, train_accs, test_accs):
    """Построение графиков обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(train_losses, label='Train Loss', linewidth=2, marker='o')
    axes[0].plot(test_losses, label='Test Loss', linewidth=2, marker='s')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Кривая обучения (Loss)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(train_accs, label='Train Accuracy', linewidth=2, marker='o')
    axes[1].plot(test_accs, label='Test Accuracy', linewidth=2, marker='s')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Точность (%)')
    axes[1].set_title('Кривая обучения (Accuracy)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ========== 7. Тестирование на реальных изображениях ==========
def test_on_examples(model, device):
    """Тестирование на конкретных примерах"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=True
    )

    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Предсказания
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    # Визуализация
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(10):
        img = images[i].cpu().squeeze()
        img = img * 0.3081 + 0.1307  # Денормализация
        img = np.clip(img, 0, 1)

        axes[i].imshow(img, cmap='gray')

        pred = predictions[i].item()
        true = labels[i].item()
        color = 'green' if pred == true else 'red'
        mark = '✓' if pred == true else '✗'

        axes[i].set_title(f'Pred: {pred} {mark}\nTrue: {true}', color=color)
        axes[i].axis('off')

    plt.suptitle('Примеры предсказаний CNN', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Статистика
    correct = (predictions.cpu() == labels).sum().item()
    accuracy = 100 * correct / 10
    print(f"\nТочность на 10 случайных примерах: {accuracy:.1f}% ({correct}/10)")


# ========== 8. Основной скрипт ==========
if __name__ == "__main__":
    print("=" * 60)
    print("День 17: Свёрточные нейронные сети (CNN)")
    print("=" * 60)

    # 1. Тестируем размерности моделей
    test_model_shapes()

    # 2. Обучаем упрощённую CNN
    print("\n" + "=" * 60)
    print("Обучение CNN на MNIST")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, final_accuracy = train_cnn_simple()

    # 3. Тестируем на примерах
    print("\n" + "=" * 60)
    print("Тестирование обученной модели")
    print("=" * 60)
    test_on_examples(model, device)

    # 4. Сохранение модели
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_accuracy': final_accuracy,
        'model_type': 'SimpleCNN'
    }, 'mnist_cnn_simple.pth')

    print(f"\nМодель сохранена в 'mnist_cnn_simple.pth'")
    print(f"Итоговая точность: {final_accuracy:.2f}%")

    print("\n" + "=" * 60)
    print("CNN успешно обучена!")
    print("Ключевые выводы:")
    print("1. CNN извлекают иерархические признаки (грани -> текстуры -> объекты)")
    print("2. Пулинг уменьшает размерность и повышает инвариантность к сдвигам")
    print("3. Параметры фильтров делятся по всему изображению")
    print("=" * 60)