import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class SparseMoELayer(nn.Module):
    """
    Разреженный слой Mixture of Experts
    Используется в Mixtral 8x7B, Switch Transformers
    """
    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Эксперты — независимые FFN сети
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            ) for _ in range(num_experts)
        ])

        # Gating network — определяет, к каким экспертам идти
        self.gate = nn.Linear(d_model, num_experts)

        # Для стабильности
        self.noise_epsilon = 1e-6

    def _gating(self, x: torch.Tensor, use_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Вычисление gating probabilities с шумом для балансировки"""
        # x shape: [batch_size, seq_len, d_model]
        logits = self.gate(x)

        if use_noise and self.training:
            # Добавляем шум для лучшей балансировки (как в Switch Transformer)
            noise = torch.randn_like(logits) * F.softplus(logits)
            logits += noise

        # Softmax по экспертам
        probs = F.softmax(logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Zero out non-top k
        mask = torch.zeros_like(probs).scatter(-1, top_k_indices, 1.0)
        probs *= mask

        probs /= (probs.sum(dim=-1, keepdim=True) + self.noise_epsilon)

        return probs, top_k_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Получаем gating probabilities
        probs, expert_indices = self._gating(x)

        # Reshape для обработки
        flat_x = x.view(-1, d_model)
        flat_probs = probs.view(-1, self.num_experts)
        flat_indices = expert_indices.view(-1, self.top_k)

        # Result
        final_output = torch.zeros_like(flat_x)

        # Для каждого эксперта обрабатываем его токены
        for expert_idx, expert in enumerate(self.experts):
            # Находим все позиции, где этот эксперт в top-k
            mask = (flat_indices == expert_idx).any(dim=-1)

            if mask.any():
                expert_input = flat_x[mask]
                expert_output = expert(expert_input)

                # Добавляем с весом от gating
                expert_weight = flat_probs[mask][:, expert_idx:expert_idx+1]
                final_output[mask] += expert_output * expert_weight

        return final_output.view(batch_size, seq_len, d_model)

    class MoETransformerBlock(nn.Module):
        def __init__(self, d_model: int, num_heads: int, num_experts: int = 8, top_k: int  = 2):
            super().__init__()
            self.attention = nn.MultiHeadAttention(d_model, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            # MoE слой вместо обычного FFN
            self.moe = SparseMoELayer(d_model, num_experts, top_k)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Self-attention с residual
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x, attn_out)

            # MoE с residual
            moe_out = self.moe(x)
            x = self.norm2(x + moe_out)

            return x

# ========== Демонстрация работы MoE ==========
def demonstrate_moe():
    """Показываем как работает Mixture of Experts"""

    print("=" * 60)
    print("Mixture of Experts (MoE) Demonstration")
    print("=" * 60)

    # Создаём MoE слой
    d_model = 64
    num_experts = 8
    top_k = 2

    moe = SparseMoELayer(d_model, num_experts, top_k)

    # Тестовые данные
    batch_size = 4
    seq_len = 10

    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = moe(x)

    print(f"Вход: {x.shape}")
    print(f"Выход: {output.shape}")
    print(f"Количество экспертов: {num_experts}")
    print(f"Top-k экспертов на токен: {top_k}")

    # Анализ загрузки экспертов
    with torch.no_grad():
        probs, indices = moe._gating(x, use_noise=False)

    # Статистика использования экспертов
    expert_usage = torch.zeros(num_experts)
    for i in range(num_experts):
        expert_usage[i] = (indices == i).float().sum().item()

    expert_usage /= expert_usage.sum()

    print(f"\nРаспределение нагрузки по экспертам:")
    for i, usage in enumerate(expert_usage):
        print(f"  Эксперт {i}: {usage:.2%}")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gating probabilities для примера
    sample_probs = probs[0, 0].numpy()  # Первый токен первого батча
    axes[0].bar(range(num_experts), sample_probs)
    axes[0].set_xlabel('Эксперт')
    axes[0].set_ylabel('Вероятность')
    axes[0].set_title('Gating probabilities для одного токена')
    axes[0].axhline(y=1 / num_experts, color='r', linestyle='--', label='Равномерное')
    axes[0].legend()

    # Распределение нагрузки
    axes[1].bar(range(num_experts), expert_usage.numpy())
    axes[1].set_xlabel('Эксперт')
    axes[1].set_ylabel('Доля использования')
    axes[1].set_title('Распределение нагрузки по экспертам')
    axes[1].axhline(y=1 / num_experts, color='r', linestyle='--', label='Равномерное')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return moe

# ========== 2. Multi-Query Attention (MQA) и Grouped-Query Attention (GQA) ==========
class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention — используется в PaLM, Falcon
    Один ключ и значение для всех голов, разные запросы
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Запросы для каждой головы
        self.W_q = nn.Linear(d_model, d_model)

        self.W_k = nn.Linear(d_model, self.head_dim)
        self.W_v = nn.Linear(d_model, self.head_dim)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Запросы: [batch, seq_len, num_heads * head_dim] -> [batch, seq_len, num_heads, head_dim]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        K = self.W_k(x).unsqueeze(1)
        V = self.W_v(x).unsqueeze(1)

        # Вычисляем attention
        scores = torch.matmul(Q.unsqueeze(1), K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn_weights = F.softmax(scores, dim=-1)

        # Применяем attention
        context = torch.matmul(attn_weights, V.unsqueeze(1))

        # Объединяем головы
        context = context.premute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        return self.W_o(context)

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention — используется в Llama 2, 3
    Компромисс между MHA и MQA
    """
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads

        assert num_heads % num_kv_heads == 0
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Запросы: [batch, seq_len, num_heads * head_dim]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Ключи и значения для KV групп
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Повторяем ключи для каждой группы запросов
        K = K.repeat_interleave(self.num_queries_per_kv, dim=2)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Вычисляем attention
        scores = torch.matmul(Q.transpose(1, 2), K.transpose(1, 2).transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, V.transpose(1, 2))

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.W_o(context)

# ========== 3. Rotary Position Embedding (RoPE) — используется в Llama, GPT-NeoX ==========
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding
    Вращает эмбеддинги на основе позиции
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Предвычисляем частоты
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Предвычисляем позиции
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Строим кэш для быстрого вычисления"""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Объединяем синус и косинус
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        return (
            self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0),
            self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_position_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Применяет rotary position embedding к Q и K"""
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed

class RoPEAttention(nn.Module):
    """Self-attention с Rotary Position Embedding"""
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Применяем RoPE
        cos, sin = self.rotary_emb(x, seq_len)
        Q, K = apply_rotary_position_emb(Q, K, cos, sin)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, V)

        # Объединяем головы
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.W_o(context)

# ========== 4. Flash Attention (память-эффективная реализация) ==========
def flash_attention_implementation():
    """
    Концептуальная реализация Flash Attention
    Без сохранения большой матрицы attention
    """
    print("\n" + "="*60)
    print("Flash Attention — память-эффективное внимание")
    print("="*60)

    def naive_attention(Q, K, V):
        """Наивная реализация — O(n²) памяти"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output

    def flash_attention_concept(Q, K, V, block_size=256):
        """
        Flash attention — обрабатывает блоками
        Память: O(n * block_size)
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape

        output = torch.zeros_like(Q)

        for i in range(0, seq_len, block_size):
            Q_block = Q[:, :, i:i + block_size, :]

            # Скалярные произведения для блока
            scores_block = torch.matmul(Q_block, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Softmax на блоке
            attn_block = F.softmax(scores_block, dim=-1)

            # Выход для блока
            outputp[:, i:i + block_size, :] = torch.matmul(attn_block, V)

        return output

    print("\nПреимущества Flash Attention:")
    advantages = [
        "✅ Память: O(n) вместо O(n²)",
        "✅ Скорость: до 2-4x быстрее",
        "✅ Можно обрабатывать очень длинные последовательности",
        "✅ Встроена в PyTorch 2.0+ (torch.nn.functional.scaled_dot_product_attention)"
    ]

    for adv in advantages:
        print(adv)

# ========== 5. Практический пример: Сборка современного LLM ==========
class ModuleLLMBlock(nn.Module):
    """Блок современной LLM (Llama-стиль)"""
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int = 4, max_seq_len: int = 2048, use_swiglu: bool = True):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Grouped-Query Attention с RoPE
        self.attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads)

        # Добавляем RoPE
        self.rotary_emb = RotaryEmbedding(d_model // num_heads, max_seq_len)

        # SwiGLU activation (как в современных LLM)
        if use_swiglu:
            hidden_dim = 4 * d_model
            self.ffn = nn.Sequential(
                nn.Linear(d_model, hidden_dim *2),
                SwiGLU(),
                nn.Linear(hidden_dim, d_model)
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)

        batch_size, seq_len, _ = x.shape
        Q = self.attention.W_q(x)
        K = self.attention.W_k(x)

        # Здесь нужно применить RoPE к Q и K перед attention
        # (упрощённо для демонстрации)
        x = self.attention(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x

class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


# ========== 6. Сравнение архитектур ==========

def compare_architectures():
    """Сравнение различных архитектур внимания"""

    print("\n" + "=" * 60)
    print("Сравнение архитектур внимания")
    print("=" * 60)

    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_len = 1024

    # Создаём разные виды внимания
    architectures = {
        "Multi-Head Attention": nn.MultiheadAttention(d_model, num_heads, batch_first=True),
        "Multi-Query Attention": MultiQueryAttention(d_model, num_heads),
        "Grouped-Query Attention": GroupedQueryAttention(d_model, num_heads, num_kv_heads=4),
        "RoPE Attention": RoPEAttention(d_model, num_heads)
    }

    # Тестовые данные
    x = torch.randn(batch_size, seq_len, d_model)

    # Сравниваем
    results = {}

    for name, layer in architectures.items():
        # Forward pass
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if torch.cuda.is_available():
            x = x.cuda()
            layer = layer.cuda()
            start.record()

        if "Multi-Head Attention" in name:
            output = layer(x, x, x)[0]
        else:
            output = layer(x)

        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end)
        else:
            time_ms = 0

        # Считаем параметры
        params = sum(p.numel() for p in layer.parameters())

        results[name] = {
            "params": params,
            "time_ms": time_ms,
            "output_shape": output.shape
        }

    # Вывод результатов
    print(f"\{'Архитектура':<30} {'Параметры':<15} {'Время (ms)':<15} {'Выход':<20}")
    print("-" * 80)

    for name, metrics in results.items():
        print(f"{name:<30} {metrics['params']:<15,} {metrics['time_ms']:<15.2f} {str(metrics['output_shape']):<20}")


# ========== 7. Практическое задание ==========

def advanced_assignment():
    """Практическое задание по современным архитектурам"""

    print("\n" + "=" * 60)
    print("🏋️ Практическое задание: Сборка современной LLM")
    print("=" * 60)

    tasks = [
        "1. Реализуйте полный трансформер с MoE (8 экспертов, top_k=2)",
        "2. Добавьте Rotary Position Embedding в вашу реализацию",
        "3. Реализуйте Grouped-Query Attention с 4 KV heads",
        "4. Сравните производительность MHA, MQA и GQA",
        "5. Реализуйте SwiGLU активацию и сравните с GELU",
        "6. Добавьте Flash Attention (используйте PyTorch 2.0+)",
        "7. Соберите полную модель с 12 слоями для генерации текста"
    ]

    for task in tasks:
        print(task)

    print("\n📊 Бенчмарки для проверки:")
    benchmarks = {
        "Model size": "< 1B параметров",
        "Training speed": "> 10k tokens/sec",
        "Inference speed": "> 100 tokens/sec",
        "Memory usage": "< 16GB GPU RAM",
        "Perplexity": "< 15 on WikiText-2"
    }

    for metric, target in benchmarks.items():
        print(f"  {metric}: {target}")

    print("\n🔧 Подсказки:")
    hints = [
        "• Используйте torch.compile() для ускорения",
        "• Flash Attention доступен в F.scaled_dot_product_attention",
        "• Для MoE добавьте auxiliary loss для балансировки экспертов",
        "• Rotary embeddings можно предвычислить для скорости",
        "• Используйте mixed precision training (AMP)"
    ]

    for hint in hints:
        print(hint)


# ========== 8. Дополнительные темы для изучения ==========

def advanced_topics():
    """Продвинутые темы для изучения"""

    print("\n" + "=" * 60)
    print("📚 Продвинутые темы для самостоятельного изучения")
    print("=" * 60)

    topics = [
        {
            "name": "1. Speculative Decoding",
            "description": "Ускорение инференса в 2-3x за счёт предсказания нескольких токенов сразу",
            "papers": ["Fast Inference from Transformers via Speculative Decoding (2022)"]
        },
        {
            "name": "2. PagedAttention",
            "description": "Эффективное управление KV cache для длинных контекстов",
            "papers": ["Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)"]
        },
        {
            "name": "3. Quantization-Aware Training",
            "description": "Обучение с учётом последующей квантизации",
            "papers": ["LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (2022)"]
        },
        {
            "name": "4. Long Context Extension",
            "description": "Методы для расширения контекста до миллионов токенов",
            "papers": ["YaRN: Efficient Context Window Extension of Large Language Models (2023)"]
        },
        {
            "name": "5. Mixture of Experts Optimization",
            "description": "Балансировка нагрузки и коммуникация между экспертами",
            "papers": ["GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (2020)"]
        },
        {
            "name": "6. Reinforcement Learning from Human Feedback",
            "description": "Настройка LLM под предпочтения человека",
            "papers": ["Training language models to follow instructions with human feedback (2022)"]
        }
    ]

    for topic in topics:
        print(f"\n{topic['name']}")
        print(f"   {topic['description']}")
        for paper in topic['papers']:
            print(f"   📄 {paper}")


# ========== 9. Основной скрипт ==========

def main():
    print("=" * 60)
    print("День 24: Продвинутые архитектуры и SOTA модели")
    print("=" * 60)

    # 1. Демонстрация MoE
    print("\n1. Mixture of Experts (MoE)")
    moe = demonstrate_moe()

    # 2. Сравнение архитектур внимания
    print("\n2. Сравнение Multi-Query и Grouped-Query Attention")
    compare_architectures()

    # 3. Flash Attention
    flash_attention_implementation()

    # 4. Практическое задание
    advanced_assignment()

    # 5. Продвинутые темы
    advanced_topics()

    # 6. Рекомендации по дальнейшему изучению
    print("\n" + "=" * 60)
    print("🚀 Что дальше?")
    print("=" * 60)

    next_steps = [
        "1. Реализуйте полную LLM с нуля (12 слоёв, 8 голов, RoPE, GQA, SwiGLU)",
        "2. Обучите её на OpenWebText или The Pile",
        "3. Внедрите speculative decoding для ускорения",
        "4. Добавьте поддержку длинного контекста (>32k токенов)",
        "5. Оптимизируйте с помощью quantization (GPTQ, AWQ)",
        "6. Разверните модель с помощью vLLM или TensorRT-LLM"
    ]

    for step in next_steps:
        print(step)

    # 7. Ресурсы
    print("\n📚 Полезные ресурсы:")
    resources = [
        "• The Annotated Transformer (Harvard NLP)",
        "• Illustrated Transformer (Jay Alammar)",
        "• GPT in 60 Lines of NumPy (Jay Mody)",
        "• LLM Visualization (bbycroft.net)",
        "• flash-attention repository (Dao-AILab)",
        "• vLLM: Fast and easy LLM serving"
    ]

    for resource in resources:
        print(resource)

    print("\n" + "=" * 60)
    print("🎉 Поздравляю! Вы изучили современные архитектуры!")
    print("Теперь вы готовы к работе с SOTA моделями!")
    print("=" * 60)


if __name__ == "__main__":
    main()