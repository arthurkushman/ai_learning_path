import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

# Нормализация
X_mean = X_torch.mean(dim=0, keepdim=True)
X_std = X_torch.std(dim=0, keepdim=True)
X_normalized = (X_torch - X_mean) / (X_std + 1e-8)

y_mean = y_torch.mean()
y_std = y_torch.std()
y_normalized = (y_torch - y_mean) / y_std

class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Первый слой: 2 входа -> 3 выхода
        self.fc1 = nn.Linear(2, 3)
        # Второй слой: 3 входа -> 1 выход
        self.fc2 = nn.Linear(3, 1)
        
        # Инициализация весов как в нашей numpy-реализации
        # self._init_weights()
    
    # def _init_weights(self):
    #     """Инициализируем веса как в нашем примере"""
    #     with torch.no_grad():
    #         # fc1.weight: размер (3, 2) в PyTorch
    #         # Наши веса W1: [[-11, 22], [33, -44], [55, 66]]
    #         self.fc1.weight.copy_(torch.tensor([
    #             [-11., 22.],   # первый нейрон
    #             [33., -44.],   # второй нейрон
    #             [55., 66.]     # третий нейрон
    #         ]))
            
    #         # fc1.bias: размер (3,)
    #         self.fc1.bias.copy_(torch.tensor([10., -20., 30.]))
            
    #         # fc2.weight: размер (1, 3) в PyTorch
    #         self.fc2.weight.copy_(torch.tensor([[0.5, -1.0, 0.2]]))
            
    #         # fc2.bias: размер (1,)
    #         self.fc2.bias.copy_(torch.tensor([0.1]))
    
    def forward(self, x):
        # x: (batch_size, 2)
        x = torch.relu(self.fc1(x))  # -> (batch_size, 3)
        x = self.fc2(x)              # -> (batch_size, 1)
        return x

def generate_regression_data(n_samples=100):
    np.random.seed(42)

    area = np.random.uniform(50, 200, n_samples)
    rooms = np.random.randint(1, 6, n_samples)

    price = 1000 * area + 50000 + rooms + np.random.normal(0, 10000, n_samples)

    X = np.column_stack([area, rooms]).astype(np.float32)
    y = price.reshape(-1, 1).astype(np.float32)

    return X, y

X_real, y_real = generate_regression_data(100)
print(f"Сгенерировано {len(X_real)} объектов")
print(f"X shape: {X_real.shape}, y shape: {y_real.shape}")
print(f"Пример: дом {X_real[0][0]:.1f} м², {int(X_real[0][1])} комнат -> цена {y_real[0][0]:.2f} руб.")

class HousingDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]    

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42
)

train_dataset = HousingDataset(X_train, y_train)
test_dataset = HousingDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Попробуйте разные комбинации:
learning_rates = [0.1, 0.01, 0.001, 0.0001]
num_epochs_list = [100, 500, 1000]

best_loss = float('inf')
best_params = {}

criterion = nn.MSELoss()

for lr in learning_rates:
    for num_epochs in num_epochs_list:
        print(f"\n==== LR: {lr}, Epochs: {num_epochs} ===")

        model = TwoLayerNet()

        optimizer = optim.SGD(model.parameters(), lr=lr)

        for epoch in range (num_epochs):
            predictions = model(X_normalized)
            loss = criterion(predictions, y_normalized)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss.item():.4f}")


        with torch.no_grad():
            final_predictions = model(X_normalized)
            final_loss = criterion(final_predictions, y_normalized)

        print(f"Final loss: {final_loss.item():.4f}")    

        if final_loss.item() < best_loss:
            best_loss = final_loss.item()
            best_params = {'lr': lr, 'epochs': num_epochs}
            best_model = model.state_dict().copy()


print(f"\nЛучшие параметры: {best_params}")
print(f"Лучший loss: {best_loss}")

class HousePriceModel(nn.Module): 
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x): 
        return self.net(x)

model = HousePriceModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_losses = []

for epoch in range(100):
    model.train()
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    train_losses.append(epoch_train_loss / len(train_loader))

    model.eval()
    epoch_test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            epoch_test_loss += loss.item()

    test_losses.append(epoch_test_loss / len(test_loader))                

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train loss = {train_losses[-1]:.2f}, Test loss = {test_losses[-1]:.2f}")