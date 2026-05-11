import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


# ========== 1. Загрузка и подготовка CIFAR-10 ==========
def load_cifar10():
    print("Загрузка CIFAR-10 датасета...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Размер тренировочного датасета: {len(train_dataset)}")
    print(f"Размер тестового датасета: {len(test_dataset)}")
    print(f"Количество классов: {len(classes)}")

    return train_dataset, test_dataset, classes

# ========== 2. Визуализация CIFAR-10 ==========
def visualize_cifar10(dataset, classes):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i in range(10):
        for img, label in dataset:
            if label == i:
                img_np = img.numpy().transpose(1, 2, 0)
                mean = np.array([0.4914, 0.4822, 0.4465])
                std = np.array([0.2470, 0.2435, 0.2616])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)

                axes[i].imshow(img_np)
                axes[i].set_title(f"{classes[i]}")
                axes[i].axis('off')
                break

    plt.suptitle('Образцы CIFAR-10 (10 классов)', fontsize=14)
    plt.tight_layout()
    plt.show()

# ========== 3. ResNet блок (остаточный блок) ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        # Первый свёрточный слой
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Второй сверточный слой
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

# ========== 4. Простая ResNet ==========
class SimpleResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()

        # Начальный слой
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet blocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, block, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

# ========== 5. Трансферное обучение с предобученной моделью ==========
def transfer_learning_resnet18(num_classes=10, freeze_layers=True):
    model = torchvision.models.resnet18(weights='DEFAULT')

    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    print(f"Модель ResNet18 загружена")
    print(f"Заморожены слои: {freeze_layers}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model

# ========== 6. Обучение с прогресс-баром ==========
def train_model(model, train_loader, test_loader, device, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'learning_rate': []
    }

    print(f"\nНачинаем обучение на {device}...")
    print(f"Количество эпох: {num_epochs}")
    print(f"Learning rate: {lr}")

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix({
                'train_loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })

        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            test_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for images, labels in test_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                test_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * test_correct / test_total:.2f}%'
                })

        epoch_train_loss = train_loss / len(train_loader)
        epoch_test_loss = test_loss / len(test_loader)
        epoch_test_acc = test_correct / test_total * 100
        epoch_train_acc = train_correct / train_total * 100

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(epoch_test_loss)

        epoch_time = time.time() - start_time
        print(f"\nЭпоха {epoch+1} завершена за {epoch_time:.1f} сек")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

    return history


# ========== 7. Визуализация результатов ==========
def plot_results(history, model_name):
    """Визуализация кривых обучения"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['test_loss'], label='Test Loss', linewidth=2)
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'{model_name} - Кривые Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['test_acc'], label='Test Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Точность (%)')
    axes[0, 1].set_title(f'{model_name} - Кривые Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 0].plot(history['learning_rates'], color='red', linewidth=2)
    axes[1, 0].set_xlabel('Эпоха')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title(f'{model_name} - Динамика Learning Rate')
    axes[1, 0].grid(True, alpha=0.3)

    # Accuracy comparison
    axes[1, 1].bar(['Train', 'Test'],
                   [history['train_acc'][-1], history['test_acc'][-1]],
                   color=['blue', 'orange'])
    axes[1, 1].set_ylabel('Точность (%)')
    axes[1, 1].set_title(f'{model_name} - Финальная точность')
    axes[1, 1].grid(True, alpha=0.3)

    # Добавляем значения на столбцы
    for i, v in enumerate([history['train_acc'][-1], history['test_acc'][-1]]):
        axes[1, 1].text(i, v + 1, f'{v:.2f}%', ha='center')

    plt.suptitle(f'Результаты обучения: {model_name}', fontsize=16)
    plt.tight_layout()
    plt.show()


# ========== 8. Сравнение предсказаний ==========
def visualize_predictions_comparison(model1, model2, test_loader, classes, device, num_samples=10):
    """Сравнение предсказаний двух моделей"""
    model1.eval()
    model2.eval()

    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    with torch.no_grad():
        images = images.to(device)

        # Предсказания модели 1
        outputs1 = model1(images)
        _, predictions1 = torch.max(outputs1, 1)

        # Предсказания модели 2
        outputs2 = model2(images)
        _, predictions2 = torch.max(outputs2, 1)

    # Визуализация
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))

    for i in range(num_samples):
        # Изображение
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'True: {classes[labels[i]]}')
        axes[i, 0].axis('off')

        # Предсказания модели 1
        pred1 = predictions1[i].item()
        color1 = 'green' if pred1 == labels[i] else 'red'
        axes[i, 1].text(0.5, 0.5, f'{classes[pred1]}',
                        ha='center', va='center', fontsize=12, color=color1)
        axes[i, 1].set_title('Model 1')
        axes[i, 1].axis('off')

        # Предсказания модели 2
        pred2 = predictions2[i].item()
        color2 = 'green' if pred2 == labels[i] else 'red'
        axes[i, 2].text(0.5, 0.5, f'{classes[pred2]}',
                        ha='center', va='center', fontsize=12, color=color2)
        axes[i, 2].set_title('Model 2')
        axes[i, 2].axis('off')

    plt.suptitle('Сравнение предсказаний моделей', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    train_dataset, test_dataset, classes = load_cifar10()
    visualize_cifar10(train_dataset, classes)

    batch_size = 120
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    simple_resnet = SimpleResNet(num_classes=10).to(device)
    print(simple_resnet)

    history_simple = train_model(simple_resnet, train_loader, test_loader, device, num_epochs=3, lr=0.001)

    # Вариант A: Замороженные слои (быстрое обучение)
    resnet18_frozen = transfer_learning_resnet18(num_classes=10, freeze_layers=False).to(device)
    history_frozen = train_model(resnet18_frozen, train_loader, test_loader, device, num_epochs=3, lr=0.001)

    # Вариант B: Fine-tuning всех слоёв (медленнее, но точнее)
    resnet18_finetune = transfer_learning_resnet18(num_classes=10, freeze_layers=False).to(device)
    history_finetune = train_model(resnet18_finetune, train_loader, test_loader, device, num_epochs=3, lr=0.001)

    plot_results(history_simple, "SimpleResNet")
    plot_results(history_frozen, "ResNet18 (Frozen)")
    plot_results(history_finetune, "ResNet18 (Fine-tuned)")

    visualize_predictions_comparison(simple_resnet, resnet18_finetune, test_loader, classes, device)

    results = {
        "SimpleResNet": history_simple['test_acc'][-1],
        "ResNet18 (Frozen)": history_frozen['test_acc'][-1],
        "ResNet18 (Fine-tuned)": history_finetune['test_acc'][-1]
    }

    for model_name, accuracy in results.items():
        print(f"{model_name:25s}: {accuracy:.2f}%")

    best_model = max(results, key=results.get)
    print(f"\n🏆 Лучшая модель: {best_model} ({results[best_model]:.2f}%)")

    # 7. Сохранение лучшей модели
    if best_model == "SimpleResNet":
        torch.save(simple_resnet.state_dict(), 'simple_resnet_cifar10.pth')
    elif best_model == "ResNet18 (Frozen)":
        torch.save(resnet18_frozen.state_dict(), 'resnet18_frozen_cifar10.pth')
    else:
        torch.save(resnet18_finetune.state_dict(), 'resnet18_finetune_cifar10.pth')

    print(f"\nЛучшая модель сохранена")

    # 8. Демонстрация работы skip connections
    print("\n" + "=" * 60)
    print("Демонстрация skip connections в ResNet")
    print("=" * 60)

    test_block = ResidualBlock(64, 64).to(device)

    test_input = torch.randn(1, 3, 32, 32).to(device)

    print("Тестируем ResidualBlock:")
    print(f"Вход: {test_input.shape}")

    with torch.no_grad():
        test_output = test_block(test_input)
        print(f"Выход: {test_output.shape}")

        if test_input.shape == test_output.shape:
            print("✓ Skip connection работает: вход и выход имеют одинаковую размерность")
        else:
            print("✗ Проблема с размерностями")

        if torch.all(test_output != 0):
            print("✓ Выход не нулевой")
        else:
            print("✗ Выход нулевой")

    return results

if __name__ == "__main__":
    try:
        results = main()

        print("\n" + "=" * 60)
        print("Ключевые выводы дня 18:")
        print("=" * 60)
        print("1. ResNet решает проблему исчезающих градиентов через skip connections")
        print("2. Трансферное обучение позволяет использовать предобученные модели")
        print("3. Заморозка слоёв ускоряет обучение, fine-tuning повышает точность")
        print("4. Предобученные модели часто лучше собственных реализаций")
        print("=" * 60)

    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        print("Возможные причины:")
        print("1. Проблемы с загрузкой данных (проверьте интернет)")
        print("2. Нехватка памяти GPU")
        print("3. Попробуйте уменьшить batch_size")