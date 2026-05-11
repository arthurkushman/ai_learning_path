import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt


# ========== 1. Self-Attention механизм ==========
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Линейные слои для Query, Key, Value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Финальный линейный слой
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        output = self.fc_out(attention_output)

        return output, attention_weights


# ========== 2. Transformer Block ==========
class TransformerBlock(nn.Module):
    def __init__(selfself, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output, attention_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attention_output))

        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention_weights


# ========== 3. Positional Encoding ==========
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=100):
        super().__init__()

        pe = torch.zeros(max_seq_len, embed_dim)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ========== 4. Мини-Трансформер для классификации ==========
class MiniTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2, max_seq_len=50, num_classes=2,
                 dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)

        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Эмбеддинги слов
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # 2. Добавляем позиционную информацию
        self.positional_encoding(embedded)

        # 3. Проходим через transformer блоки
        attention_weights_list = []
        for transformer_block in self.transformer_blocks:
            encoded, attention_weights = transformer_block(encoded)
            attention_weights_list.append(attention_weights)

        # 4. Глобальный средний пулинг
        # Транспонируем: [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim, seq_len]
        pooled = self.pooling(encoded.transpose(1, 2)).squeeze(-1)

        # 5. Классификатор
        output = self.classifier(pooled)

        return output, attention_weights_list


# ========== 5. Демонстрация работы Self-Attention ==========
def visualize_attention():
    # Создаём тестовые данные
    batch_size = 1
    seq_len = 5
    embed_dim = 8
    num_heads = 2

    # Тестовый вход
    test_input = torch.randn(batch_size, seq_len, embed_dim)

    # Создаём механизм внимания
    attention = SelfAttention(embed_dim, num_heads)

    output, attention_weights = attention(test_input)
    print(f"Выход: shape={output.shape}")
    print(f"Attention weights: shape={attention_weights.shape}")

    attn_weights_np = attention_weights[0, 0].detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Attention matrix
    im = axes[0].imshow(attn_weights_np, cmap='hot', aspect='auto')
    axes[0].set_xlabel('Key positions')
    axes[0].set_ylabel('Query positions')
    axes[0].set_title('Attention Weights (Head 1)')
    plt.colorbar(im, ax=axes[0])

    # Пример интерпретации
    tokens = ['The', 'cat', 'sat', 'on', 'mat']
    for i in range(seq_len):
        # Находим наиболее связанные токены
        most_attended = np.argsort(attn_weights_np[i])[-3:][::-1]
        axes[1].text(0.1, 0.9 - i * 0.15,
                     f"'{tokens[i]}' attends to: {[tokens[j] for j in most_attended]}",
                     fontsize=10)

    axes[1].axis('off')
    axes[1].set_title('Интерпретация внимания')

    plt.suptitle('Механизм Self-Attention', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Объяснение
    print("\nКак работает Self-Attention:")
    print("1. Каждый токен создаёт Query, Key, Value векторы")
    print("2. Query одного токена сравнивается со Keys всех токенов")
    print("3. Получаем веса внимания (насколько каждый токен важен)")
    print("4. Взвешенная сумма Values даёт выход")
    print("\nПример выше: 'cat' уделяет внимание 'sat' и 'mat'")


# ========== 6. Демонстрация Positional Encoding ==========
def visualize_positional_encoding():
    embed_dim = 16
    max_seq_len = 20

    pe = PositionalEncoding(embed_dim, max_seq_len)

    dummy_input = torch.zeros(1, max_seq_len, embed_dim)
    positional_encoding = pe(dummy_input)[0].detach().numpy()

    # Визуализация
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Heatmap
    im = axes[0].imshow(positional_encoding.T, cmap='RdBu', aspect='auto')
    axes[0].set_xlabel('Позиция в последовательности')
    axes[0].set_ylabel('Размерность эмбеддинга')
    axes[0].set_title('Positional Encoding (Heatmap)')
    plt.colorbar(im, ax=axes[0])

    # Графики для нескольких размерностей
    for dim in [0, 1, 2, 3]:
        axes[1].plot(positional_encoding[:, dim], label=f'Dimension {dim}')

    axes[1].set_xlabel('Позиция в последовательности')
    axes[1].set_ylabel('Значение кодирования')
    axes[1].set_title('Positional Encoding по размерностям')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Positional Encoding добавляет информацию о позиции', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\nЗачем нужно Positional Encoding:")
    print("1. Трансформеры не имеют встроенного понимания порядка")
    print("2. Attention работает одинаково для всех позиций")
    print("3. Positional Encoding добавляет информацию о позиции токена")
    print("4. Синусоидальные функции: могут обобщать на длинные последовательности")


# ========== 7. Сравнение RNN и Transformer ==========
def compare_rnn_transformer():
    print("\n" + "=" * 60)
    print("Сравнение RNN и Transformer")
    print("=" * 60)

    comparison_data = {
        'Аспект': [
            'Обработка последовательности',
            'Параллелизм',
            'Длинные зависимости',
            'Память',
            'Вычислительная сложность',
            'Обучение',
            'Интерпретируемость'
        ],
        'RNN/LSTM': [
            'Последовательно (токен за токеном)',
            'Низкий (последовательная природа)',
            'Проблема исчезающих градиентов',
            'Внутреннее состояние (hidden state)',
            'O(n) для последовательности длины n',
            'Медленное (нельзя параллелизовать)',
            'Сложно интерпретировать'
        ],
        'Transformer': [
            'Параллельно (все токены сразу)',
            'Высокий (матричные операции)',
            'Отличная (attention на всё)',
            'Нет внутреннего состояния',
            'O(n²) (attention matrix)',
            'Быстрое (полная параллелизация)',
            'Attention weights показывают связи'
        ]
    }

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Архитектурная схема RNN
    axes[0].text(0.5, 0.9, 'RNN/LSTM Architecture', ha='center', fontsize=12, fontweight='bold')
    axes[0].text(0.1, 0.8, 'x₁ → RNN → h₁', fontsize=10)
    axes[0].text(0.1, 0.7, 'x₂ → RNN → h₂ (использует h₁)', fontsize=10)
    axes[0].text(0.1, 0.6, 'x₃ → RNN → h₃ (использует h₂)', fontsize=10)
    axes[0].text(0.1, 0.5, '...', fontsize=10)
    axes[0].text(0.1, 0.4, 'xₙ → RNN → hₙ (использует hₙ₋₁)', fontsize=10)
    axes[0].text(0.1, 0.2, '→ Последовательная обработка\n→ Проблема длинных зависимостей', fontsize=9)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].axis('off')
    axes[0].set_title('RNN: Последовательная обработка')

    # Архитектурная схема Transformer
    axes[1].text(0.5, 0.9, 'Transformer Architecture', ha='center', fontsize=12, fontweight='bold')
    axes[1].text(0.1, 0.8, 'x₁  x₂  x₃  ...  xₙ', fontsize=10)
    axes[1].text(0.1, 0.7, '│    │    │        │', fontsize=10)
    axes[1].text(0.1, 0.6, 'Self-Attention (все со всеми)', fontsize=10)
    axes[1].text(0.1, 0.5, '│    │    │        │', fontsize=10)
    axes[1].text(0.1, 0.4, 'h₁  h₂  h₃  ...  hₙ', fontsize=10)
    axes[1].text(0.1, 0.2, '→ Параллельная обработка\n→ Attention между всеми токенами', fontsize=9)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    axes[1].set_title('Transformer: Параллельная обработка')

    plt.suptitle('Сравнение архитектур: RNN vs Transformer', fontsize=14)
    plt.tight_layout()
    plt.show()


# ========== 8. Простой пример обучения ==========
def simple_transformer_example():
    print("\n" + "=" * 60)

    vocab_size = 1000
    batch_size = 4
    seq_len = 10

    model = MiniTransformerClassifier(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_layers=2,
        max_seq_len=seq_len,
        num_classes=2,
    )

    print("Архитектура модели:")
    print(model)

    inputs = torch.randin(0, vocab_size, (batch_size, seq_len))
    print(f"\nВходные данные: shape={inputs.shape}")
    print(f"Пример: {inputs[0]}")

    # Forward pass
    outputs, attention_weights = model(inputs)
    print(f"\nВыход модели: shape={outputs.shape}")
    print(f"Выходные значения: {outputs}")

    # Преобразуем в вероятности
    probabilities = F.softmax(outputs, dim=1)
    print(f"\nВероятности классов: {probabilities}")

    # Получаем предсказания
    predictions = torch.argmax(outputs, dim=1)
    print(f"Предсказания: {predictions}")

    # Показываем attention weights
    print(f"\nAttention weights из {len(attention_weights)} слоёв:")
    for i, attn_weights in enumerate(attention_weights):
        print(f"  Слой {i + 1}: shape={attn_weights.shape}")
        first_head_attention = attn_weights[0, 0].detach().numpy()

    return model


# ========== 9. Практическое задание ==========
def transformer_practical_exercise():
    """Практическое задание по трансформерам"""

    print("\n" + "=" * 60)
    print("Практическое задание: Собери свой Transformer!")
    print("=" * 60)

    # Задание 1: Реализуйте Multi-Head Attention
    print("\nЗадание 1: Понимание Multi-Head Attention")
    print("-" * 50)

    questions = [
        "1. Зачем нужны multiple heads в attention?",
        "2. Что происходит с размерностью при разделении на головы?",
        "3. Почему attention scores делятся на sqrt(d_k)?",
        "4. Что такое residual connections и зачем они нужны?",
        "5. Чем LayerNorm отличается от BatchNorm?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"{question}")
        input("  Ваш ответ (нажмите Enter для продолжения)...")

        # Ответы
        answers = [
            "Multiple heads позволяют модели обращать внимание на разные аспекты (синтаксис, семантика, и т.д.)",
            "Embedding dimension делится на num_heads (например, 512 → 8 heads × 64 dim each)",
            "Для стабилизации градиентов (предотвращение слишком больших/малых значений softmax)",
            "Residual connections помогают градиентам проходить напрямую, решая проблему исчезающих градиентов",
            "LayerNorm нормализует по features (для каждого примера отдельно), BatchNorm — по батчу"
        ]

        print(f"  Ответ: {answers[i - 1]}")
        print()

    # Задание 2: Модифицируйте код
    print("\nЗадание 2: Модификации кода")
    print("-" * 50)

    modifications = [
        "1. Добавьте bidirectional attention",
        "2. Реализуйте causal attention (для GPT-стиля моделей)",
        "3. Добавьте dropout к attention weights",
        "4. Реализуйте different attention mechanisms (dot-product, additive)",
        "5. Добавьте возможность разного размера heads"
    ]

    print("Предложите как можно модифицировать наш MiniTransformer:")
    for i, mod in enumerate(modifications, 1):
        print(f"{mod}")

    print("\nПопробуйте реализовать одну из модификаций!")


# ========== 10. Основной скрипт ==========
def main():
    print("=" * 60)
    print("День 20: Трансформеры и механизм внимания (Transformers)")
    print("=" * 60)

    # 1. Демонстрация Self-Attention
    print("\n1. Изучаем механизм Self-Attention")
    visualize_attention()

    # 2. Демонстрация Positional Encoding
    print("\n2. Изучаем Positional Encoding")
    visualize_positional_encoding()

    # 3. Сравнение RNN и Transformer
    print("\n3. Сравниваем RNN и Transformer архитектуры")
    compare_rnn_transformer()

    # 4. Простой пример с Transformer
    print("\n4. Работаем с мини-трансформером")
    model = simple_transformer_example()

    # 5. Практическое задание
    transformer_practical_exercise()

    # 6. Современные применения
    print("\n" + "=" * 60)
    print("Современные модели на основе Transformers")
    print("=" * 60)

    modern_models = {
        'GPT (OpenAI)': {
            'Тип': 'Decoder-only',
            'Применение': 'Генерация текста',
            'Особенность': 'Causal attention (только на предыдущие токены)',
            'Пример': 'ChatGPT, GPT-4'
        },
        'BERT (Google)': {
            'Тип': 'Encoder-only',
            'Применение': 'Понимание текста',
            'Особенность': 'Bidirectional attention',
            'Пример': 'Поиск, классификация'
        },
        'T5 (Google)': {
            'Тип': 'Encoder-Decoder',
            'Применение': 'Трансформация текста',
            'Особенность': 'Универсальная архитектура',
            'Пример': 'Перевод, суммаризация'
        },
        'Vision Transformer': {
            'Тип': 'Transformer для изображений',
            'Применение': 'Computer Vision',
            'Особенность': 'Разбивает изображение на патчи',
            'Пример': 'Классификация изображений'
        }
    }

    print("\nКлючевые модели:")
    for model_name, info in modern_models.items():
        print(f"\n{model_name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    # 7. Ключевые выводы
    print("\n" + "=" * 60)
    print("Ключевые выводы дня 20:")
    print("=" * 60)
    print("1. Self-Attention: каждый токен 'видит' все токены одновременно")
    print("2. Multi-Head: несколько 'взглядов' на данные")
    print("3. Positional Encoding: добавляет информацию о порядке")
    print("4. Residual + LayerNorm: стабилизирует обучение")
    print("5. Параллелизм: главное преимущество над RNN")
    print("=" * 60)

    print("\n🎉 Поздравляю! Вы изучили основы трансформеров!")
    print("Это основа современных LLM (GPT, BERT, Llama, etc.)")


# ========== 11. Запуск ==========
if __name__ == "__main__":
    try:
        import pandas as pd
        main()
    except ImportError as e:
        print(f"Не хватает библиотеки: {e}")
        print("Установите: pip install pandas")
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        import traceback
        traceback.print_exc()
