import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
import math
import time
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt


# ========== 1. Data Parallelism с DDP ==========

def setup_ddp(rank: int, world_size: int):
    """Инициализация Distributed Data Parallel"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Очистка DDP"""
    dist.destroy_process_group()


class DistributedTrainer:
    """Тренировка с Distributed Data Parallel"""

    def __init__(self, model: nn.Module, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        # Модель на устройство
        self.model = model.to(self.device)

        # Оборачиваем в DDP
        self.model = DDP(self.model, device_ids=[rank])

        # Оптимизатор
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def train_epoch(self, dataloader) -> float:
        """Одна эпоха обучения с DDP"""
        self.model.train()
        total_loss = 0.0

        # Важно: устанавливаем seed для каждой эпохи разный на разных GPU
        torch.manual_seed(self.rank + 42)

        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward
            output = self.model(data)
            loss = F.cross_entropy(output, target)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (важно для стабильности)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Собираем loss со всех GPU
            loss_tensor = loss.clone().detach()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= self.world_size

            total_loss += loss_tensor.item()

            if batch_idx % 10 == 0 and self.rank == 0:
                print(f"Batch {batch_idx}: Loss = {loss_tensor.item():.4f}")

        self.scheduler.step()

        return total_loss / len(dataloader)


def train_ddp_example():
    """Пример запуска DDP обучения"""
    print("=" * 60)
    print("Distributed Data Parallel (DDP) Training")
    print("=" * 60)

    world_size = torch.cuda.device_count()
    print(f"Доступно GPU: {world_size}")

    if world_size < 2:
        print("Нужно минимум 2 GPU для демонстрации")
        return

    # Создаём простую модель
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    print("\nКонфигурация DDP:")
    print(f"  • World size: {world_size}")
    print(f"  • Master addr: localhost")
    print(f"  • Master port: 12355")
    print(f"  • Backend: nccl")

    print("\nПреимущества DDP:")
    advantages = [
        "✓ Линейное ускорение с добавлением GPU",
        "✓ Автоматическая синхронизация градиентов",
        "✓ Поддержка mixed precision",
        "✓ Эффективное использование памяти"
    ]

    for adv in advantages:
        print(f"  {adv}")

    print("\nКоманда для запуска с torchrun:")
    print("  torchrun --nproc_per_node=2 train.py")


# ========== 2. Fully Sharded Data Parallel (FSDP) ==========

class FSDPTrainer:
    """
    Fully Sharded Data Parallel
    Каждый GPU хранит только свою часть параметров
    """

    def __init__(self, model: nn.Module, rank: int, world_size: int):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            CPUOffload,
            BackwardPrefetch,
            ShardingStrategy
        )
        from torch.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy,
            size_based_auto_wrap_policy
        )

        self.rank = rank
        self.world_size = world_size

        # FSDP конфигурация
        self.fsdp_config = {
            "cpu_offload": CPUOffload(offload_params=True),
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
            "use_orig_params": True,
        }

        # Оборачиваем модель в FSDP
        self.model = FSDP(
            model,
            **self.fsdp_config
        ).cuda(rank)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def get_model_size(self) -> Dict[str, float]:
        """Получение информации о размере модели"""
        total_params = sum(p.numel() for p in self.model.parameters())
        local_params = sum(p.numel() for p in self.model.parameters() if p.is_cuda)

        return {
            "total_params_m": total_params / 1e6,
            "local_params_m": local_params / 1e6,
            "memory_saved": (total_params - local_params) / total_params * 100
        }


def demonstrate_fsdp():
    """Демонстрация FSDP"""
    print("\n" + "=" * 60)
    print("Fully Sharded Data Parallel (FSDP)")
    print("=" * 60)

    # Создаём большую модель
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(4096, 4096) for _ in range(12)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))
            return x

    model = LargeModel()
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Размер модели: {total_params / 1e9:.2f}B параметров")
    print(f"Память для параметров: {total_params * 4 / 1e9:.2f}GB (FP32)")

    print("\nСравнение подходов:")
    comparison = [
        "| Подход | Память на GPU | Коммуникация |",
        "|--------|---------------|--------------|",
        "| DDP    | 100% модели   | Большая      |",
        "| FSDP   | 1/N модели    | Средняя      |",
        "| TP     | 1/N каждого слоя | Малая     |"
    ]

    for line in comparison:
        print(line)

    print("\nПреимущества FSDP:")
    advantages = [
        "✓ Можно обучать модели до 13B на 8x A100",
        "✓ CPU offload для ещё большей экономии",
        "✓ Автоматическое шардирование",
        "✓ Поддержка checkpointing"
    ]

    for adv in advantages:
        print(f"  {adv}")


# ========== 3. Mixed Precision Training (AMP) ==========

class MixedPrecisionTrainer:
    """
    Mixed Precision Training
    FP16/BF16 для скорости, FP32 для точности
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.device = device
        self.model = model.to(device)

        # Gradient scaler для FP16
        self.scaler = torch.cuda.amp.GradScaler()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

    def train_step_amp(self, data: torch.Tensor, target: torch.Tensor) -> float:
        """Один шаг обучения с mixed precision"""
        self.model.train()

        # Автоматический кастинг в FP16
        with torch.cuda.amp.autocast():
            output = self.model(data)
            loss = F.cross_entropy(output, target)

        # Backward со scaler
        self.scaler.scale(loss).backward()

        # Unscale и градиентный clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Обновление весов
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return loss.item()

    def compare_precision(self, data: torch.Tensor):
        """Сравнение FP32 и FP16"""
        print("\nСравнение точности:")

        with torch.no_grad():
            # FP32
            output_fp32 = self.model(data.float())

            # FP16
            with torch.cuda.amp.autocast():
                output_fp16 = self.model(data.half())

            diff = torch.mean(torch.abs(output_fp32.float() - output_fp16.float()))
            print(f"  Разница FP32 vs FP16: {diff.item():.6f}")

            # Размер в памяти
            print(f"  Память FP32: {output_fp32.element_size() * output_fp32.nelement() / 1024:.2f} KB")
            print(f"  Память FP16: {output_fp16.element_size() * output_fp16.nelement() / 1024:.2f} KB")


def demonstrate_amp():
    """Демонстрация Mixed Precision Training"""
    print("\n" + "=" * 60)
    print("Mixed Precision Training (AMP)")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1000)
    )

    trainer = MixedPrecisionTrainer(model)

    # Тестовые данные
    data = torch.randn(32, 1024).cuda()
    target = torch.randint(0, 1000, (32,)).cuda()

    # Замер скорости
    import time

    # FP32
    start = time.time()
    for _ in range(10):
        trainer.optimizer.zero_grad()
        output = model(data.float())
        loss = F.cross_entropy(output, target)
        loss.backward()
        trainer.optimizer.step()
    fp32_time = (time.time() - start) * 1000 / 10

    # FP16
    start = time.time()
    for _ in range(10):
        loss = trainer.train_step_amp(data, target)
    fp16_time = (time.time() - start) * 1000 / 10

    print(f"\nСкорость обучения:")
    print(f"  FP32: {fp32_time:.2f} ms/step")
    print(f"  FP16: {fp16_time:.2f} ms/step")
    print(f"  Ускорение: {fp32_time / fp16_time:.2f}x")

    trainer.compare_precision(data[:1])


# ========== 4. Gradient Accumulation ==========

class GradientAccumulationTrainer:
    """
    Gradient Accumulation
    Накопление градиентов для больших effective batch sizes
    """

    def __init__(self, model: nn.Module, accumulation_steps: int = 4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step_accumulated(self, dataloader, epoch: int):
        """Обучение с накоплением градиентов"""
        self.model.train()

        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()

            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = F.cross_entropy(output, target)

                # Нормализуем loss для накопления
                loss = loss / self.accumulation_steps

            # Накопление градиентов
            self.scaler.scale(loss).backward()

            # Обновляем веса после accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() * self.accumulation_steps:.4f}")


def demonstrate_gradient_accumulation():
    """Демонстрация gradient accumulation"""
    print("\n" + "=" * 60)
    print("Gradient Accumulation")
    print("=" * 60)

    print("Проблема: Не хватает памяти для большого batch size")
    print("Решение: Накопление градиентов из нескольких микро-батчей")

    example = """
    Batch size = 32, Accumulation steps = 4
    ┌─────────────────────────────────────────┐
    │ Micro-batch 1 (BS=8) → gradient 1       │
    │ Micro-batch 2 (BS=8) → gradient 2       │
    │ Micro-batch 3 (BS=8) → gradient 3       │
    │ Micro-batch 4 (BS=8) → gradient 4       │
    │                                         │
    │ gradients 1+2+3+4 → update weights      │
    └─────────────────────────────────────────┘
    Effective batch size = 32
    """
    print(example)

    print("\nПреимущества:")
    advantages = [
        "✓ Можно использовать большие effective batch sizes",
        "✓ Стабильнее обучение",
        "✓ Экономия памяти GPU",
        "✓ Гибкость в настройке"
    ]

    for adv in advantages:
        print(f"  {adv}")


# ========== 5. Advanced Optimizers ==========

class AdvancedOptimizers:
    """Сравнение современных оптимизаторов"""

    @staticmethod
    def get_optimizer(name: str, params, lr: float = 1e-4):
        """Получение оптимизатора по имени"""
        optimizers = {
            "AdamW": torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1),
            "Lion": lambda: __import__('lion_pytorch').Lion(params, lr=lr, weight_decay=0.1),
            "Sophia": lambda: __import__('sophia').Sophia(params, lr=lr),
            "Adafactor": torch.optim.Adafactor(params, lr=lr, scale_parameter=False, relative_step=False),
            "Adam8bit": lambda: __import__('bitsandbytes').optim.Adam8bit(params, lr=lr),
        }

        if name in optimizers:
            if callable(optimizers[name]):
                return optimizers[name]()
            return optimizers[name]
        else:
            return torch.optim.Adam(params, lr=lr)

    @staticmethod
    def compare_optimizers():
        """Сравнение оптимизаторов"""
        print("\n" + "=" * 60)
        print("Сравнение современных оптимизаторов")
        print("=" * 60)

        comparison = """
┌───────────┬──────────────┬──────────┬─────────────┬────────────┐
│ Оптимизатор│ Память       │ Скорость │ Сходимость  │ Применение │
├───────────┼──────────────┼──────────┼─────────────┼────────────┤
│ AdamW     │ 2x параметров │ Средняя  │ Отличная    │ Стандарт   │
│ Lion      │ 1x параметров │ Высокая  │ Хорошая     │ Google     │
│ Sophia    │ 2x параметров │ Высокая  │ Отличная    │ Новинка    │
│ Adafactor │ 1x параметров │ Средняя  │ Хорошая     │ T5, PaLM   │
│ Adam8bit  │ 0.5x параметров│ Высокая │ Хорошая     │ BNB        │
└───────────┴──────────────┴──────────┴─────────────┴────────────┘
        """
        print(comparison)

        print("\nРекомендации:")
        recommendations = [
            "• AdamW: Для большинства задач, стабильный выбор",
            "• Lion: Для очень больших моделей, экономия памяти",
            "• Sophia: Для быстрой сходимости, особенно в NLP",
            "• Adafactor: Для T5-подобных моделей, трансформеры",
            "• Adam8bit: Когда критична память GPU"
        ]

        for rec in recommendations:
            print(f"  {rec}")


# ========== 6. Learning Rate Scheduling ==========

class LRSchedulers:
    """Продвинутые стратегии изменения learning rate"""

    @staticmethod
    def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
        """Cosine annealing with warmup"""

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def linear_with_warmup(optimizer, warmup_steps: int, total_steps: int):
        """Linear decay with warmup"""

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def inverse_sqrt(optimizer, warmup_steps: int):
        """Inverse square root decay (используется в T5)"""

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return float(warmup_steps) ** 0.5 / float(current_step) ** 0.5

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def visualize_lr_schedules():
    """Визуализация learning rate schedules"""
    print("\n" + "=" * 60)
    print("Learning Rate Schedules")
    print("=" * 60)

    steps = 1000
    warmup = 100

    schedulers = {
        "Cosine": lambda step: LRSchedulers.cosine_with_warmup(None, warmup, steps).get_lr()[
            0] if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup))),
        "Linear": lambda step: step / warmup if step < warmup else (steps - step) / (steps - warmup),
        "Inverse Sqrt": lambda step: step / warmup if step < warmup else math.sqrt(warmup / step),
        "Constant": lambda step: 1.0
    }

    # Строим графики
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (name, func) in enumerate(schedulers.items()):
        lrs = [func(step) for step in range(steps)]
        axes[idx].plot(lrs, linewidth=2)
        axes[idx].set_xlabel('Step')
        axes[idx].set_ylabel('Learning Rate')
        axes[idx].set_title(name)
        axes[idx].axvline(x=warmup, color='r', linestyle='--', alpha=0.5, label='Warmup end')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Learning Rate Schedules Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\nКогда использовать:")
    usage = [
        "• Cosine: Стандарт для большинства задач",
        "• Linear: Простой, хорошо работает",
        "• Inverse Sqrt: Для очень долгого обучения (T5)",
        "• Constant: Когда не хотите затухания"
    ]

    for line in usage:
        print(f"  {line}")


# ========== 7. Activation Checkpointing ==========

class ActivationCheckpointing:
    """
    Activation Checkpointing (Gradient Checkpointing)
    Экономия памяти за счёт пересчёта активаций
    """

    @staticmethod
    def checkpointed_layer(module, x):
        """Слой с checkpointing"""
        from torch.utils.checkpoint import checkpoint

        return checkpoint(module, x)

    @staticmethod
    def compare_memory_usage():
        """Сравнение использования памяти"""
        print("\n" + "=" * 60)
        print("Activation Checkpointing")
        print("=" * 60)

        # Создаём глубокую модель
        layers = []
        for _ in range(24):
            layers.extend([
                nn.Linear(1024, 1024),
                nn.ReLU()
            ])
        model = nn.Sequential(*layers)

        def estimate_memory(model, batch_size=32, seq_len=512, use_checkpointing=False):
            """Оценка памяти для активаций"""
            hidden_size = 1024
            bytes_per_param = 2  # FP16

            # Память для параметров
            param_memory = sum(p.numel() for p in model.parameters()) * bytes_per_param

            # Память для активаций (приблизительно)
            if use_checkpointing:
                # Только для checkpointed слоёв
                activation_memory = batch_size * seq_len * hidden_size * bytes_per_param * 2
            else:
                # Для всех слоёв
                activation_memory = batch_size * seq_len * hidden_size * bytes_per_param * len(layers)

            total_memory = (param_memory + activation_memory) / (1024 ** 3)  # GB

            return param_memory / (1024 ** 3), activation_memory / (1024 ** 3), total_memory

        params_gb, acts_gb, total_gb = estimate_memory(model, use_checkpointing=False)
        params_gb_c, acts_gb_c, total_gb_c = estimate_memory(model, use_checkpointing=True)

        print(f"\nБез checkpointing:")
        print(f"  Параметры: {params_gb:.2f} GB")
        print(f"  Активации: {acts_gb:.2f} GB")
        print(f"  Всего: {total_gb:.2f} GB")

        print(f"\nС checkpointing:")
        print(f"  Параметры: {params_gb_c:.2f} GB")
        print(f"  Активации: {acts_gb_c:.2f} GB")
        print(f"  Всего: {total_gb_c:.2f} GB")
        print(f"  Экономия: {(1 - total_gb_c / total_gb) * 100:.1f}%")


# ========== 8. Практическое задание ==========

def advanced_training_assignment():
    """Практическое задание по продвинутому обучению"""

    print("\n" + "=" * 60)
    print("🏋️ Практическое задание: Оптимизация обучения LLM")
    print("=" * 60)

    tasks = [
        "1. Реализуйте полный пайплайн обучения с DDP на 4 GPU",
        "2. Добавьте mixed precision (FP16/BF16) с GradScaler",
        "3. Реализуйте gradient accumulation с configurable steps",
        "4. Сравните AdamW, Lion и Sophia на небольшой модели",
        "5. Добавьте cosine schedule с warmup",
        "6. Внедрите activation checkpointing для экономии памяти",
        "7. Измерьте throughput (tokens/sec) для разных конфигураций",
        "8. Найдите оптимальный баланс скорости/памяти/точности"
    ]

    for i, task in enumerate(tasks, 1):
        print(f"  {task}")

    print("\n📊 Метрики для отслеживания:")
    metrics = {
        "Training speed": "tokens/sec",
        "Memory usage": "GB per GPU",
        "Loss convergence": "final loss",
        "Gradient norm": "stability",
        "GPU utilization": "%"
    }

    for metric, description in metrics.items():
        print(f"  • {metric}: {description}")

    print("\n🔧 Продвинутые техники:")
    techniques = [
        "• ZeRO-3 для огромных моделей",
        "• Pipeline parallelism для 100B+ моделей",
        "• Tensor parallelism для отдельных слоёв",
        "• Sequence parallelism для длинных контекстов",
        "• Compile model с torch.compile()"
    ]

    for technique in techniques:
        print(f"  {technique}")


# ========== 9. Основной скрипт ==========

def main():
    print("=" * 60)
    print("День 25: Продвинутые техники обучения и оптимизации LLM")
    print("=" * 60)

    # 1. Distributed Training
    train_ddp_example()
    demonstrate_fsdp()

    # 2. Mixed Precision
    demonstrate_amp()

    # 3. Gradient Accumulation
    demonstrate_gradient_accumulation()

    # 4. Advanced Optimizers
    AdvancedOptimizers.compare_optimizers()

    # 5. Learning Rate Schedules
    visualize_lr_schedules()

    # 6. Activation Checkpointing
    ActivationCheckpointing.compare_memory_usage()

    # 7. Практическое задание
    advanced_training_assignment()

    # 8. Сводка по оптимизации
    print("\n" + "=" * 60)
    print("📈 Сводка: Как оптимизировать обучение LLM")
    print("=" * 60)

    optimization_summary = [
        ("1. Data", "• DDP/FSDP для нескольких GPU\n"
                    "• Gradient accumulation для больших batch\n"
                    "• Dynamic batching для разной длины"),

        ("2. Precision", "• Mixed precision (FP16/BF16)\n"
                         "• Gradient scaler для FP16\n"
                         "• Master weights в FP32"),

        ("3. Memory", "• Activation checkpointing\n"
                      "• Gradient checkpointing\n"
                      "• CPU offload для оптимизатора"),

        ("4. Speed", "• torch.compile() для ускорения\n"
                     "• Flash Attention для длинных seq\n"
                     "• Fused kernels для операций"),

        ("5. Stability", "• Gradient clipping\n"
                         "• Warmup + decay schedule\n"
                         "• Weight decay и dropout")
    ]

    for title, content in optimization_summary:
        print(f"\n{title}:")
        for line in content.split('\n'):
            print(f"  {line}")

    # 9. Практический пример конфигурации
    print("\n" + "=" * 60)
    print("🚀 Пример production конфигурации для обучения 7B модели")
    print("=" * 60)

    config = """
{
    "model": "7B parameters",
    "gpus": 8,
    "per_gpu_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "effective_batch_size": 256,

    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "lr_schedule": "cosine",
    "warmup_steps": 2000,
    "weight_decay": 0.1,

    "precision": "bf16",
    "gradient_clipping": 1.0,
    "activation_checkpointing": true,

    "distributed_strategy": "FSDP",
    "sharding_strategy": "FULL_SHARD",
    "cpu_offload": false,

    "throughput_target": "150k tokens/sec",
    "memory_target": "40GB per GPU"
}
    """
    print(config)

    print("\n" + "=" * 60)
    print("🎉 Теперь вы знаете все ключевые техники обучения больших моделей!")
    print("Завтра: Инференс и оптимизация для production!")
    print("=" * 60)


if __name__ == "__main__":
    main()