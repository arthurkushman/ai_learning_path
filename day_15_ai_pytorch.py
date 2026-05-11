import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Загрузка California Housing Dataset...")
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)
feature_names = data.feature_names

print(f"\nИнформация о датасете:")
print(f"Количество объектов: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Признаки: {', '.join(feature_names)}")
print(f"Целевая переменная: Медианная цена дома (в $100,000)")
print(f"Диапазон цен: ${y.min()*100000:.0f} - ${y.max()*100000:.0f}")

X = X.astype(np.float32)
y = y.astype(np.float32)

# ========== 2. Анализ данных ==========
print("\nПервые 5 строк данных:")
df = pd.DataFrame(X, columns=feature_names)
df['Price'] = y * 100000
print(df.head())

print('Статистика')
print(df.describe())

# Визуализация распределения признаков
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, feature in enumerate(feature_names):
    axes[i].hist(X[:, i], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(feature)
    axes[i].set_xlabel('Значение')
    axes[i].set_ylabel('Частота')
    axes[i].grid(True, alpha=0.3)

# Распределение цен
axes[0].hist(y * 100000, bins=30, edgecolor='black', alpha=0.7, color='green')
axes[0].set_title('Распеределение цен домов')
axes[0].set_xlabel('Price ($)')
axes[0].set_ylabel('Frequency')
axes[8].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========== 3. Корреляционный анализ ==========
# Матрица корреляций
df_features = pd.DateFrame(X, columns=feature_names)
df_features['Price'] = y.flatten()

correletion_matrix = df_features.corr()
print("\nКорреляция признаков с ценой:")
price_corr = correletion_matrix['Price'].sort_values(ascending=False)
for feature, corr in price_corr.items():
    print(f"{feature:20s}: {corr:+.3f}")

# Визуализация матрицы корреляций
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Корреляция')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Матрица корреляций')
plt.tight_layout()
plt.show()    

# ========== 4. Предобработка данных ==========
# Разделение на train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"\nРазделение данных:")
print(f"Train: {X_train.shape[0]} объектов")
print(f"Val: {X_val.shape[0]} объектов")
print(f"Test: {X_test.shape[0]} объектов")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.fit_transform(X_val)
X_test_scaled = scaler_X.fit_transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.fit_transform(y_val)
y_test_scaled = scaler_y.fit_transform(y_test)

# ========== 5. PyTorch Dataset ==========
class CaliforniaHousingDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)    

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]    

train_dataset = CaliforniaHousingDataset(X_train_scaled, y_train_scaled)
val_dataset = CaliforniaHousingDataset(X_val_scaled, y_val_scaled)
test_dataset = CaliforniaHousingDataset(X_test_scaled, y_test_scaled)

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# ========== 6. Модель ==========
class CalifornianHousingModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=[64, 32, 16], dropout_rate=0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))    

        self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

# Проверяем доступность GPU
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
print(f"\nИспользуется устройство: {device}")            

# Создаём модель
model = CalifornianHousingModel(
    input_size=8,
    hidden_size=[128, 64, 32],
    dropout_rate=0.2
).to(device)

print(f"\nАрхитектура модели:")
print(model)
print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

# ========== 7. Обучение с расширенными функциями ==========
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)    

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)    
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

    return total_loss / len(loader)        

# ========== 8. Цикл обучения ==========
num_epochs = 200
train_losses = []
val_losses = []
learning_rates = []
best_val_loss = float('inf')    
best_model_state = None

print("\nНачинаем обучение...")
for epoch in range(num_epochs):
    # Learning
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)

    # Validation
    val_loss = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)

    # Save learning rate
    learning_rates.append(optimizer.param_groups[0]['lr'])

    # Update scheduler
    scheduler.step()

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    # early stop
    if patience_counter >= 20:
        print(f"\nРанняя остановка на эпохе {epoch+1}")
        break 

# upload the best model
model.load_state_dict(best_model_state)        

# ========== 9. Оценка модели ==========
# Оценка на тестовых данных
test_loss = evaluate(model, test_loader, criterion, device)
print(f"\nТестовая потеря (MSE на нормализованных данных): {test_loss:.6f}")

# interpret real prices
model.eval()
with torch.no_grad():
    all_predictions = []
    all_true = []

    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        predictions = model(batch_X).cpu().numpy()
        all_predictions.extend(predictions.flatten())
        all_true.extend(batch_y.numpy().flatten())

    all_predictions = scaler_y.inverse_transform(
        np.array(all_predictions).reshape(-1, 1)
    ).flatten()
    all_true = scaler_y.inverse_transform(
        np.array(all_true).reshape(-1, 1)
    )    

    # Вычисляем метрики
    mae = np.mean(np.abs(all_predictions - all_true))
    rmse = np.sqrt(np.mean(all_predictions - all_true) ** 2)
    r2 = 1 - np.sum((all_predictions - all_true) ** 2) / np.sum(all_true - np.mean(all_true) ** 2)

    print(f"\nМетрики на тестовых данных (в единицах датасета):")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    print(f"\nВ реальных деньгах ($):")
    print(f"Средняя абсолютная ошибка: ${mae * 100000:.2f}")
    print(f"Среднеквадратичная ошибка: ${rmse * 100000:.2f}")

        # Примеры предсказаний
    print(f"\nПримеры предсказаний (первые 5 объектов):")
    for i in range(5):
        pred_price = all_predictions[i] * 100000
        true_price = all_true[i] * 100000
        error = abs(pred_price - true_price)
        error_percent = (error / true_price) * 100
        
        print(f"Объект {i+1}:")
        print(f"  Предсказано: ${pred_price:,.0f}")
        print(f"  Истина: ${true_price:,.0f}")
        print(f"  Ошибка: ${error:,.0f} ({error_percent:.1f}%)")
        print()

# ========== 10. Визуализация результатов ==========
fig, exes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Кривая обучения
axes[0, 0].plot(train_losses, label='Train loss', linewidth=2)
axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Кривая обучения')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 2. Learning rate schedule
axes[0, 1].plot(learning_rates, color='red', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Learning rate')
axes[0, 1].set_title('Динамика Learning Rate')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# 3. Предсказания vs Истина
axes[0, 2].scatter(all_true * 100000, all_predictions * 100000, alpha=0.5, s=20)
axes[0, 2].plot([all_true.min() * 100000, all_true.max() * 100000], [all_true.min() * 100000, all_true.max() * 100000],    
    'r--', linewidth=2, label='Ideal'
)
axes[0, 2].set_xlabel('Истинная цена ($)')
axes[0, 2].set_ylabel('Предсказанная цена ($)')
axes[0, 2].set_title('Предсказания vs Истина')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Распределение ошибок
errors = (all_predictions - all_true) * 100000  # В долларах
axes[1, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Ошибка предсказания ($)')
axes[1, 0].set_ylabel('Количество')
axes[1, 0].set_title(f'Распределение ошибок\nСредняя ошибка: ${np.mean(np.abs(errors)):,.0f}')
axes[1, 0].grid(True, alpha=0.3)

# 5. Важность признаков
def compute_feature_importance_gradient(model, X_sample, device):
     """Вычисление важности признаков через градиенты"""
     X_tensor = torch.from_numpy(X_sample).to(device).requires_grad_(True)

     model.eval()
     prediction = model(X_tensor).mean()

     model.zero_grad()
     prediction.backward()

     importance = torch.abs(X_tensor.grad).mean(dim=0).cpu().detach().numpy()

X_sample = X_test_scaled[:100]
feature_importance = compute_feature_importance_gradient(model, X_sample, device)

sorted_idx = np.argsort(feature_importance)
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importance = feature_importance[sorted_idx]

axes[1, 1].barh(range(len(sorted_features)), sorted_importance)
axes[1, 1].set_yticks(range(len(sorted_features)))
axes[1, 1].set_yticklabels(sorted_features)
axes[1, 1].set_xlabel('Важность признака (средний абсолютный градиент)')
axes[1, 1].set_title('Важность признаков')

# 6. Остатки (residuals)
residuals = all_predictions - all_true
axes[1, 2].scatter(all_predictions * 100000, residuals * 100000, 
                   alpha=0.5, s=20)
axes[1, 2].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 2].set_xlabel('Предсказанная цена ($)')
axes[1, 2].set_ylabel('Остатки ($)')
axes[1, 2].set_title('Остатки предсказаний')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========== 11. Сохранение модели ==========
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_names': feature_names,
    'test_metrics': {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
}, 'california_housing_model.pth')

print("\nМодель сохранена в 'california_housing_model.pth'")
print("\nПайплайн завершён успешно!")