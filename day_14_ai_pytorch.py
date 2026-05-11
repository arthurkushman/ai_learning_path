import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_housing_data(n_samples=500):
    np.random.seed(42)

    area = np.random.uniform(30, 300, n_samples)
    rooms = np.random.randint(1, 7, n_samples)
    floor = np.random.randint(1, 25, n_samples)
    distance_to_center = np.random.uniform(8.5, 30, n_samples)

    price = (
        1000 * area + 
        50000 * rooms + 
        20000 * floor + 
        -10000 * distance_to_center +
        np.random.normal(0, 50000, n_samples)
    )

    X = np.column_stack([area, rooms, floor, distance_to_center]).astype(np.float32)
    y = price.reshape(-1, 1).astype(np.float32)

    return X, y

# Генерируем данные
X, y = generate_housing_data(1000)
print(f"Данные сгенерированы: {X.shape[0]} объектов, {X.shape[1]} признаков") 

# ========== 2. Предобработка данных ==========
# Разделение на train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nРазделение данных:")
print(f"Train: {X_train.shape[0]} объектов")
print(f"Val: {X_val.shape[0]} объектов")
print(f"Test: {X_test.shape[0]} объектов")

# Нормализация (на основе train данных!)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# ========== 3. PyTorch Dataset и DataLoader ==========
class HousingDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):    
        return self.X[idx], self.y[idx]

# Создаём датасеты
train_dataset = HousingDataset(X_train_scaled, y_train_scaled)        
val_dataset = HousingDataset(X_val_scaled, y_val_scaled)
test_dataset = HousingDataset(X_test_scaled, y_test_scaled)

# DataLoader с батчами
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ========== 4. Модель ==========
class HousePricePredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
             nn.Linear(input_size, hidden_size),
             nn.ReLU(),
             nn.Dropout(0.2),
             nn.Linear(hidden_size, hidden_size // 2),
             nn.ReLU(),
             nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.network(x)    

# Создаём модель
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
print(f"\nИспользуется устройство: {device}")

model = HousePricePredictor(input_size=4, hidden_size=64).to(device)

# ========== 5. Функции потерь и оптимизатор ==========
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ========== 6. Обучение с валидацией ==========
def train_epoch(model, loader, optimizer, criterion, device):
    """Одна эпоха обучения"""
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

"""Оценка модели"""
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

# ========== 7. Цикл обучения ==========
num_epochs = 100
train_losses = []
val_losses = []
best_val_loss = float('inf')    
best_model_state = None

print("\nНачинаем обучение...")
for epoch in range(num_epochs): 
    # Обучение
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)

    # Валидация 
    val_loss = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)

    # Планировщик learning rate
    scheduler.step(val_loss)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()

    # output progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs}: "
              f"Train Loss = {train_loss:.6f}, "
              f"Val Loss = {val_loss:.6f}, "
              f"LR = {optimizer.param_groups[0]['lr']:.6f}")

model.load_state_dict(best_model_state)          

# ========== 8. Оценка на тестовых данных ==========
test_loss = evaluate(model, test_loader, criterion, device)
print(f"\nТестовая потеря (MSE на нормализованных данных): {test_loss:.6f}")

# Преобразуем обратно в исходный масштаб для интерпретации
model.eval()
with torch.no_grad():
    # Берём несколько примеров из тестовой выборки
    X_test_tensor = torch.from_numpy(X_test_scaled[:5]).to(device)
    predictions_scaled = model(X_test_tensor).cpu().numpy()

    # Денормализуем
    predictions = scaler_y.inverse_transform(predictions_scaled)
    true_prices = y_test[:5]

    print("\nПримеры предсказаний:")
    for i in range(5):
        error = abs(predictions[i][0] - true_prices[i][0])
        error_percent = (error / true_prices[i][0]) * 100
        print(f"Объект {i+1}: Предсказано = {predictions[i][0]:.0f} руб., "
              f"Истина = {true_prices[i][0]:.0f} руб., "
              f"Ошибка = {error_percent:.1f}%")    

# ========== 9. Визуализация ==========      
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# График обучения
axes[0, 0].plot(train_losses, label='Train Loss')
axes[0, 0].plot(val_losses, label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Кривая обучения')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Предсказания vs Истинные значения
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
    ).flatten() 

axes[0, 1].scatter(all_true, all_predictions, alpha=0.5)        
axes[0, 1].plot([all_true.min(), all_true.max()], [all_true.min(), all_true.max()], 'r--', linewidth=2)
axes[0, 1].set_xlabel('Истинная цена (руб.)')
axes[0, 1].set_ylabel('Предсказанная цена (руб.)')
axes[0, 1].set_title('Предсказания vs Истина')
axes[0, 1].grid(True)

# Erorrs diff
errors = all_predictions - all_true
axes[1, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Ошибка предсказания (руб.)')
axes[1, 0].set_ylabel('Количество')
axes[1, 0].set_title('Распределение ошибок')
axes[1, 0].grid(True)

# feature importance computation
def compute_feature_importance(model, X_sample, device):
    X_tensor = torch.from_numpy(X_sample).to(device).requires_grad_(True)

    model.eval()
    prediction = model(X_tensor).mean()

    model.zero_grad()
    prediction.backward()

    importance = torch.abs(X_tensor.grad).mean(dim=0).cpu().numpy()
    return importance

X_sample = X_test_scaled[:100]    
feature_importance = compute_feature_importance(model, X_sample, device)

feature_names = ['Площадь', 'Комнаты', 'Этаж', 'Расст. до центра']
axes[1, 1].barh(feature_names, feature_importance)
axes[1, 1].set_xlabel('Важность признака (средний градиент)')
axes[1, 1].set_title('Важномть признаков')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# ========== 10. Сохранение модели ==========
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X_state': scaler_X,
    'scaler_y_state': scaler_y,
    'input_size': 4,
    'hidden_size': 64
}, 'house_price_model.pth')

print("\nМодель сохранена в 'house_price_model.pth'")