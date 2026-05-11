import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
print(f"\nИспользуется устройство: {device}")            

# ========== 1. Загрузка и подготовка данных MNIST ==========
print("\nЗагрузка MNIST датасета...")

transform = transforms.Compose([
    transforms.ToTensor(), # Преобразует PIL Image в тензор [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # Среднее и std MNIST
])

# Load dataset 
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"Размер тренировочного датасета: {len(train_dataset)}")
print(f"Размер тестового датасета: {len(test_dataset)}")

# DataLoader
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# ========== 2. Визуализация данных ==========
def visualize_mnist_samples(loader, num_samples=10):
    """Визуализация образцов MNIST"""
    dataiter = iter(loader)
    images, labels = next(dataiter)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i in range(num_samples):
        # Извлекаем изображение и метку
        img = images[i].squeeze().numpy()
        label = labels[i].item()

        # Денормализуем для отображения
        img = img * 0.3081 + 0.1307
        img = np.clip(img, 0, 1)

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Цифра: {label}')
        axes[i].axis('off')

    plt.suptitle('Образцы рукописных цифр MNIST', fontsize=14)    
    plt.tight_layout()
    plt.show()

    # Статистика по размерам
    print(f"Размерность изображений: {images[0].shape}")
    print(f"Диапазон значений пикселей: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Количество классов: {len(torch.unique(labels))}")    

# Визуализируем примеры
visualize_mnist_samples(train_loader)

# ========== 3. Построение модели для классификации ==========
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10) # 10 классов (цифры 0-9)
        )

    def forward(self, x):
        return self.fc_net(x)

model = MNISTClassifier().to(device)        
print(f"\nАрхитектура модели:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Всего параметров: {total_params:,}")
print(f"Обучаемых параметров: {trainable_params:,}")

# ========== 4. Обучение с функциями для классификации ==========
# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

def train_epoch_classification(model, loader, optimizer, criterion, device):
    """Одна эпоха обучения для классификации"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(loader)}: '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100 * correct / total:.2f}%', end='\r')

    epoch_loss = running_loss / len(loader)              
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

def evaluate_classification():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            iaages, labels = images.to(device), labels.to(device)    

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc, all_predictions, all_labels

num_epochs = 10
train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_val_acc = 0.0
best_model_state = None

print("\nНачинаем обучение классификатора...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 50)

    train_loss, train_acc, _, _ = evaluate_classification(
        model, test_loader, criterion, device
    )

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"  🎯 Новая лучшая модель! Точность: {val_acc:.2f}%")

    print(f"\n  Итоги эпохи:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

# Загружаем лучшую модель
model.load_state_dict(best_model_state)            
print(f"\nЛучшая точность на валидации: {best_val_acc:.2f}%")

# ========== 6. Оценка на тестовых данных ==========
print("\n" + "="*60)
print("Финальная оценка на тестовых данных")
print("="*60)

test_loss, test_acc, all_preds, all_labels = evaluate_classification(
    model, test_loader, criterion, device
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Количество правильно классифицированных: {int(test_acc/100 * len(test_dataset))}/{len(test_dataset)}")

# ========== 7. Матрица ошибок и отчёт классификации ==========
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)

    plt.title('Матрица ошибок (Confusion Matrix)')
    plt.label('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.show()

    return cm

cm = plot_confusion_matrix(all_labels, all_preds, classes=[str(i) for i in range(10)])    

print("\nОтчёт классификации:")
print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))

# ========== 8. Визуализация правильных и неправильных предсказаний ==========
def visualize_predictions(model, loader, device, num_samples=10, correct=True):
    images_list, labels_list, preds_list = [], [], []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            for i in range(len(images)):
                if (predicted[i] == labels[i]) == correct:
                    images_list.append(images[i].cpu())
                    labels_list.append(labels[i].cpu().item())
                    preds_list.append(predicted[i].cpu().item())

                if len(images_list) >= num_samples:
                    break

            if len(images_list) >= num_samples:
                break           

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))            
    exes = axes.flatten()

    title = "Correct preds" if correct else "Incorrect preds"
    plt.suptitle(title, fontsize=14)

    for i in range(min(num_samples, len(images_list))):
        img = images_list[i].squeeze().numpy()
        img = img * 0.3081 + 0.1307
        img = np.clip(img, 0, 1)

        axes[i].imshow(img, cmap='gray')

        if correct:
            axes[i].set_title(f'Цифра: {labels_list[i]} ✓', color='green')
        else:
            axes[i].set_title(f'True: {labels_list[i]}, Pred: {preds_list[i]} ✗', color='red')

        axes[i].axis('off')            

    plt.tight_layout()    
    plt.show()

# Визуализируем правильные и неправильные предсказания
print("\nВизуализация правильных предсказаний:")
visualize_predictions(model, test_loader, device, correct=True)

print("\nВизуализация неправильных предсказаний:")
visualize_predictions(model, test_loader, device, correct=False, num_samples=10)

def analyze_confidence(model, loader, device):
    confidences_corrrect = []
    confidences_incorrect = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Softmax 
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

            for i in range(len(images)):
                confidence = confidences[i].item()
                if predicted[i] == labels[i]:
                    confidences_correct.append(confidence)
                else:
                    confidences_incorrect.append(confidence)    

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))                

    # Гистограмма уверенности
    axes[0].hist(confidences_correct, bins=30, alpha=0.7, label='Правильные', color='green')
    axes[0].hist(confidences_incorrect, bins=30, alpha=0.7, label='Неправильные', color='red')
    axes[0].set_xlabel('Уверенность модедли')
    axes[0].set_ylabel('Количество')
    axes[0].set_title('Распределение уверенности предсказаний')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    data = [confidences_correct, confidences_incorrect]
    axes[1].boxplot(data, labels=['Правильные', 'Неправильные'])
    axes[1].set_ylabel('Уверенность модели')
    axes[1].set_title('Уверенность по категориям')
    axes[1].grid(True, alpha=0.3)

    # Статистика
    print(f"\nСтатистика уверенности:")
    print(f"Правильные предсказания: среднее = {np.mean(confidences_correct):.3f}, "
          f"медиана = {np.median(confidences_correct):.3f}")
    print(f"Неправильные предсказания: среднее = {np.mean(confidences_incorrect):.3f}, "
          f"медиана = {np.median(confidences_incorrect):.3f}")

    plt.tight_layout()
    plt.show()

analyze_confidence(model, test_loader, device)              

torch.save({
    'model_state_dict': model.state_dict(),
    'test_accuracy': test_acc, 
    'total_params': total_params,
    'best_val_acc': best_val_acc
}, 'mnist_classifier.pth')

print(f"\nМодель сохранена в 'mnist_classifier.pth'")

# ========== 11. Графики обучения ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(val_losses, label='Val Loss', linewidth=2)
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Loss')
axes[0].set_title('Кривая обучения (Loss)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
axes[1].plot(val_accs, label='Val Accuracy', linewidth=2)
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('Точность (%)')
axes[1].set_title('Кривая обучения (Accuracy)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Обучение классификатора MNIST завершено!")
print(f"Итоговая точность: {test_acc:.2f}%")
print("="*60)