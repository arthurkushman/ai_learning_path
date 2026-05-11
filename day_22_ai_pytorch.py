import torch
import torch.nn as nn
import torch.quantization as qat
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub


# ========== 1. Подготовка модели для квантизации ==========
class QuantizableCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Квантизация входа
        x = self.quant(x)

        # Свёрточные слои
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Полносвязные слои
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        # Деквантизация выхода
        x = self.dequant(x)
        return x

    def fuse_modules(self):
        """Слияние слоёв для оптимизации"""
        torch.quantization.fuse_modules(self, [
            ['conv1', 'relu1'],
            ['conv2', 'relu2'],
            ['fc1', 'relu3']
        ], inplace=True)


# ========== 2. Процесс квантизации ==========
def quantize_model(model, calibration_data):
    """Квантизация модели с калибровкой"""

    # 1. Подготовка модели
    model.eval()
    model.fuse_modules()

    # 2. Конфигурация квантизации
    model.qconfig = torch.quantization.get_default_qconfig('onednn')  # Для CPU

    torch.quantization.prepare(model, inplace=True)

    # 4. Калибровка (определение диапазонов)
    with torch.no_grad():
        for batch in calibration_data:
            _ = model(batch)

    # 5. Конвертация в квантизованный формат
    quantize_model = torch.quantization.convert(model, inplace=False)

    return quantize_model


# ========== 3. Сравнение производительности ==========
def compare_performance(original_model, quantized_model, test_data, device):
    """Сравнение точности и скорости"""
    import time

    # Сравнение размера
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.pth")
        size = os.path.getsize("temp.pth") / (1024 * 1024)  # MB
        os.remove("temp.pth")
        return size

    print("\n" + "=" * 60)
    print("Сравнение производительности")
    print("=" * 60)

    # Размер моделей
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)

    print(f"Размер оригинальной модели: {original_size:.2f} MB")
    print(f"Размер квантизованной модели: {quantized_size:.2f} MB")
    print(f"Сжатие: {original_size / quantized_size:.2f}x")

    # Скорость инференса
    def benchmark_inference(model, data, num_runs=100):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(data)

        end_time = time.time()
        avg_time = (end_time - start_time) * 1000 / num_runs  # ms
        return avg_time

    test_input = torch.randn(1, 1, 28, 28).to(device)

    # Прогрев
    _ = original_model(test_input)
    _ = quantized_model(test_input)

    # Бенчмарк
    original_time = benchmark_inference(original_model, test_input)
    quantized_time = benchmark_inference(quantized_model, test_input)

    print(f"\nСкорость инференса:")
    print(f"Оригинальная: {original_time:.2f} ms")
    print(f"Квантизованная: {quantized_time:.2f} ms")
    print(f"Ускорение: {original_time / quantized_time:.2f}x")

    # Точность
    def calculate_accuracy(model, test_loader):
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return 100 * correct / total

    # Загрузка MNIST для тестирования
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    original_accuracy = calculate_accuracy(original_model, test_loader)
    quantized_accuracy = calculate_accuracy(quantized_model, test_loader)

    print(f"\nТочность:")
    print(f"Оригинальная: {original_accuracy:.2f}%")
    print(f"Квантизованная: {quantized_accuracy:.2f}%")
    print(f"Потеря точности: {original_accuracy - quantized_accuracy:.2f}%")

    return {
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': original_size / quantized_size,
        'original_time_ms': original_time,
        'quantized_time_ms': quantized_time,
        'speedup': original_time / quantized_time,
        'original_accuracy': original_accuracy,
        'quantized_accuracy': quantized_accuracy,
        'accuracy_drop': original_accuracy - quantized_accuracy
    }


# ========== 4. Dynamic Quantization (для RNN/Transformers) ==========
def dynamic_quantization_example():
    class QuantizableLSTM(nn.Module):
        def __init__(self, input_size=128, hidden_size=256, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, 10)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])
            return output

    model = QuantizableLSTM()
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)

    print("Динамическая квантизация завершена")
    print(f"Типы слоёв в квантизованной модели:")
    for name, module in quantized_model.named_modules():
        print(f"  {name}: {type(module)}")

    return quantized_model


# ========== 5. Post-Training Quantization (PTQ) ==========
def post_training_quantization_example():
    from transformers import BertModel

    model = BertModel.from_pretrained('bert-base-uncased')

    print(f"Размер оригинального BERT: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M параметров")

    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    return model


# ========== 6. Quantization Aware Training (QAT) ==========
def quantization_aware_training_example():
    class QATModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.quant = QuantStub()
            self.dequant = DeQuantStub()

            self.conv1 = nn.Conv2d(3, 16, 3)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu1 = nn.ReLU()

            self.conv2 = nn.Conv2d(16, 32, 3)
            self.bn2 = nn.BatchNorm2d(32)
            self.relu2 = nn.ReLU()

            self.fc = nn.Linear(32 * 6 * 6, 10)

        def forward(self, x):
            x = self.quant(x)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)

            x = torch.flatten(x, 1)
            x = self.fc(x)

            x = self.dequant(x)

            return x

        def fuse_model(self):
            torch.quantization.fuse_modules(self,
                                            [['conv1', 'bn1', 'relu1'],
                                             ['conv2', 'bn2', 'relu2']],
                                            inplace=True)

    model = QATModel()

    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    torch.quantization.prepare_qat(model, inplace=True)

    return model


# ========== 7. Практический пример с MobileNet ==========
def mobilenet_quantization_example():
    import torchvision.models as models

    model = models.mobilenet_v2(pretrained=True)
    model.eval()

    original_params = sum(p.numel() for p in model.parameters())

    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.Conn2d}, dtype=torch.qint8)

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        original_output = model(dummy_input)
        quantized_output = quantized_model(dummy_input)

    # Сравнение выходов
    output_diff = torch.mean(torch.abs(original_output - quantized_output))
    print(f"Средняя разница выходов: {output_diff.item():.6f}")

    # Сравнение производительности
    import time

    def run_inference(model, input_tensor, num_iterations=100):
        model.eval()
        start = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)

        end = time.time()
        return (end - start) * 1000 / num_iterations  # ms per inference

    original_time = run_inference(model, dummy_input, 50)
    quantized_time = run_inference(quantized_model, dummy_input, 50)

    print(f"\nMobileNetV2 Performance:")
    print(f"Original: {original_time:.2f} ms per inference")
    print(f"Quantized: {quantized_time:.2f} ms per inference")
    print(f"Speedup: {original_time / quantized_time:.2f}x")

    return quantized_model


# ========== 8. ONNX экспорт с квантизацией ==========
def export_quantized_onnx(model, dummy_input, filename='quantized_model.onnx'):
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    import os
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"Размер ONNX файла: {size_mb:.2f} MB")

    return filename


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = QuantizableCNN().to(device)

    calibration_data = []
    for _ in range(100):
        calibration_data.append(torch.randn(1, 1, 28, 28).to(device))

    quantized_model = quantize_model(model, calibration_data)

    results = compare_performance(model, quantized_model, calibration_data, device)

    quantized_lstm = dynamic_quantization_example()

    get_model = quantization_aware_training_example()

    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    onnx_file = export_quantized_onnx(quantized_model, dummy_input)

    import matplotlib.pyplot as plt

    fig, exes = plt.subplots(2, 2, figsize=(12, 10))
    # График сравнения размеров
    models = ['Оригинальная', 'Квантизованная']
    sizes = [results['original_size_mb'], results['quantized_size_mb']]

    axes[0, 0].bar(models, sizes, color=['blue', 'green'])
    axes[0, 0].set_ylabel('Размер (MB)')
    axes[0, 0].set_title('Сравнение размеров моделей')
    for i, v in enumerate(sizes):
        axes[0, 0].text(i, v + 0.1, f'{v:.1f} MB', ha='center')

    # График сравнения скорости
    times = [results['original_time_ms'], results['quantized_time_ms']]

    axes[0, 1].bar(models, times, color=['red', 'orange'])
    axes[0, 1].set_ylabel('Время (ms)')
    axes[0, 1].set_title('Сравнение скорости инференса')
    for i, v in enumerate(times):
        axes[0, 1].text(i, v + 0.01, f'{v:.2f} ms', ha='center')

    # График сравнения точности
    accuracies = [results['original_accuracy'], results['quantized_accuracy']]

    axes[1, 0].bar(models, accuracies, color=['purple', 'cyan'])
    axes[1, 0].set_ylabel('Точность (%)')
    axes[1, 0].set_title('Сравнение точности')
    axes[1, 0].set_ylim([0, 100])
    for i, v in enumerate(accuracies):
        axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center')

    # График сжатия и ускорения
    metrics = ['Сжатие', 'Ускорение']
    values = [results['compression_ratio'], results['speedup']]

    axes[1, 1].bar(metrics, values, color=['magenta', 'yellow'])
    axes[1, 1].set_ylabel('Коэффициент')
    axes[1, 1].set_title('Выигрыш от квантизации')
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + 0.05, f'{v:.2f}x', ha='center')

    plt.suptitle('Результаты квантизации нейронной сети', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 9. Практические рекомендации
    print("\n" + "=" * 60)
    print("Практические рекомендации по квантизации:")
    print("=" * 60)

    recommendations = [
        ("Когда использовать:", "Production deployment, мобильные устройства, edge AI"),
        ("Static Quantization:", "CNN, когда есть калибровочные данные"),
        ("Dynamic Quantization:", "RNN, LSTM, Transformers"),
        ("QAT (Quantization Aware Training):", "Когда важна точность, можно дообучать"),
        ("Торговля точность/скорость:", "Обычно 1-3% потери точности, 2-4x ускорение"),
        ("Поддерживаемые операции:", "Conv, Linear, LSTM, Embedding, некоторые активации"),
        ("Неподдерживаемые:", "Сложные операции, кастомные слои"),
        ("Инструменты:", "PyTorch Quantization, ONNX Runtime, TensorRT")
    ]

    for title, description in recommendations:
        print(f"{title:30s} {description}")

    # 10. Домашнее задание
    print("\n" + "=" * 60)
    print("Домашнее задание:")
    print("=" * 60)

    homework_tasks = [
        "1. Квантизируйте ResNet18 и сравните с оригиналом",
        "2. Реализуйте QAT для простой CNN на CIFAR-10",
        "3. Экспортируйте квантизованную модель в ONNX и запустите через ONNX Runtime",
        "4. Измерьте энергопотребление (если возможно) до и после квантизации",
        "5. Реализуйте mixed-precision training (FP16) и сравните с квантизацией"
    ]

    for task in homework_tasks:
        print(task)

    return {
        'original_model': model,
        'quantized_model': quantized_model,
        'results': results,
        'onnx_file': onnx_file
    }


# ========== 10. Дополнительные техники оптимизации ==========
def advanced_optimization_techniques():
    class PrunableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)

                return x

    model = PrunableModel()

    def apply_pruning(model, amount=0.3):
        parameters_to_prune = (
            (model.fc1, 'weight'),
            (model.fc2, 'weight'),
            (model.fc3, 'weight'),
        )

        torch.nn.utils.prune.global_unstructured(
            parameters_to_prune,
            pruning_method=torch.nn.utils.prune.L1Unstructured,
            amount=amount
        )

        total_weights = 0
        zero_weights = 0

        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                if module.weight is not None:
                    total_weights += module.weight.nelement()
                    zero_weights += torch.sum(module.weight == 0).item()

        sparsity = 100 * zero_weights / total_weights
        print(f"  Sparsity после pruning: {sparsity:.1f}%")

        return model

    pruned_model = apply_pruning(model, amount=0.3)

    # 2. Knowledge Distillation
    print("\n2. Knowledge Distillation:")

    class TeacherModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

            return x

    class StudentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    def knowledge_distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):

        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=-1)

        distillation_loss = F.kl_div(sfot_prob, soft_targets, reduction='batchmean')
        student_loss = F.cross_entropy(student_logits, labels)

        loss = alpha * distillation_loss * (temperature ** 2) + (1 - alpha) * student_loss

        return loss

     # 3. Mixed Precision Training
    print("\n3. Mixed Precision Training (FP16/FP32):")

    from torch.cuda.amp import autocast, GradScaler

    def mixed_precision_training_example():
        model = nn.Linear(10, 10).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scalar = GradScaler()

        for epoch in range(10):
            with autocast():
                output = model(torch.randn(1, 10).cuda())
                loss = F.mse_loss(output, torch.randn(1, 10).cuda())

            scaler.scale(loss).backward()
            scalar.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        print("Mixed precision training завершён")

    mixed_precision_training_example()

    # 4. Gradient Checkpointing
    print("\n4. Gradient Checkpointing (экономия памяти):")

    class MemoryEfficientModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(256, 256) for _ in range(10)
            ])

        def forward(self, x):
            # Используем gradient checkpointing
            from torch.utils.checkpoint import checkpoint

            for i, layer in enumerate(self.layers):
                # Чекпоинтим каждые 2 слоя
                if i % 2 == 0:
                    x = checkpoint(layer, x)
                else:
                    x = layer(x)
                x = F.relu(x)

            return x

    memory_efficient_model = MemoryEfficientModel()
    print(" Модель с gradient checkpointing создана")

    # 5. Model Parallelism
    print("\n5. Model Parallelism (распределение по GPU):")

    class ModelParallelCNN(nn.Module):
        def __init__(self):
            super().__init__()

        # Часть 1 на GPU 0
        self.part1 = nn.Sequential(
            nn.Conv2d(3, 64, 3).to('cuda:0'),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Часть 2 на GPU 1
        self.part2 = nn.Sequential(
            nn.Conv2d(64, 128, 3).to('cuda:1'),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5, 10).to('cuda:0'),
            nn.ReLU(),
            nn.Linear(256, 10).to('cuda:0'),
        )

        def forward(self, x):
            # Перемещаем данные между GPU
            x = self.part1(x).to('cuda:0')
            x = self.part2(x).to('cuda:1')
            x = torch.flatten(x, 1)
            x = self.classifier(x.to('cuda:0'))
            return x

    if torch.cuda.device_count() > 1:
        model_parallel = ModelParallelCNN()
        print("  Model Parallelism инициализирован на 2 GPU")
    else:
        print("  Недостаточно GPU для Model Parallelism")

    return {
        'pruning': "L1 Unstructured Pruning",
        'distillation': "Teacher-Student Framework",
        'mixed_precision': "FP16/FP32 Training",
        'checkpointing': "Gradient Checkpointing",
        'model_parallel': "Multi-GPU Distribution"
    }


# ========== 11. Запуск ==========
if __name__ == "__main__":
    try:
        # Импортируем необходимые библиотеки
        import os
        import torchvision

        # Запускаем основной скрипт
        results = main()

        # Запускаем продвинутые техники
        advanced_techniques = advanced_optimization_techniques()

        print("\n" + "=" * 60)
        print("Итоги Дня 22:")
        print("=" * 60)
        print("✅ Освоили квантизацию моделей (Static/Dynamic/QAT)")
        print("✅ Научились сравнивать производительность")
        print("✅ Узнали про ONNX экспорт квантизованных моделей")
        print("✅ Познакомились с продвинутыми техниками оптимизации")
        print("✅ Построили графики сравнения производительности")
        print("=" * 60)

    except ImportError as e:
        print(f"Не хватает библиотеки: {e}")
        print("Установите: pip install torch torchvision matplotlib")
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        import traceback

        traceback.print_exc()