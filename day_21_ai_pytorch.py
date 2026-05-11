import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from collections import defaultdict

# ========== 1. Causal Transformer (GPT-style) ==========
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x, use_causal_mask=True):
        batch_size, seq_len, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Разделяем на головы
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if use_causal_mask:
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention
        attempt_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim
        )

        output = self.fc_out(attention_output)

        return output, attention_weights

class CausalTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, max_seq_len=100, dropout=0.1):
        super().__init__()

        self.attention = CausalSelfAttention(embed_dim, num_heads, max_seq_len)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed Forward Network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_causal_mask=True):
        attn_output, attn_weight = self.attention(x, use_causal_mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)

        return x, attn_weight

class CausalTransformerGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_dim=256, num_layers=3, max_seq_len=100, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Эмбеддинги
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer блоки
        self.layers = nn.ModuleList(
            [
                CausalTransformerBlock(embed_dim, num_heads, ff_dim, max_seq_len, dropout)
                for _ in range(num_layers)
            ]
        )

        # Нормализация и выходной слой
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights (как в GPT)
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)

        # Инициализация
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, use_causal_mask=True):
        batch_size, seq_len = input_ids.shape

        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        position_embeds = self.position_embedding(positions)

        x = self.dropout(token_embeds + position_embeds)

        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, use_causal_mask)
            attention_weights.append(attn)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, attention_weights

    def generate(self, prompt_ids, max_length=50, temperature=1.0, top_k=50):
        self.eval()

        generated = prompt_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Берем последние max_seq_len токенов
                input_ids = generated[:, -self.max_seq_len:] if generated.size(1) > self.max_seq_len else generated

                # Forward pass
                logits, _ = self(input_ids, use_causal_mask=True)

                # Берем логиты для последнего токена
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == 0:
                    break

        return generated


# ========== 2. Подготовка данных для генерации ==========
def create_vocabulary_from_reviews():
    """Создание словаря из отзывов"""
    # Создадим небольшой набор отзывов
    reviews = [
        "This movie was absolutely fantastic I loved every minute",
        "Great acting and wonderful story highly recommend",
        "Terrible waste of time horrible acting",
        "Amazing cinematography and brilliant performances",
        "Boring and predictable would not watch again",
        "Excellent film with superb direction",
        "Worst movie I have ever seen",
        "Beautiful storytelling and emotional journey",
        "Disappointing and overrated",
        "Masterpiece of modern cinema"
    ]

    # Создаем словарь
    all_words = []
    for review in reviews:
        words = review.lower().split()
        all_words.extend(words)

    word_counts = defaultdict(int)
    for word in all_words:
        word_counts[word] += 1

    # Сортируем по частоте
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Создаем словарь
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in sorted_words:
        if word not in vocab:
            vocab[word] = len(vocab)

    # Обратный словарь
    idx_to_word = {idx: word for word, idx in vocab.items()}

    print(f"Словарь создан: {len(vocab)} слов")
    print(f"Примеры: {list(vocab.keys())[:10]}")

    return vocab, idx_to_word


def encode_text(text, vocab, max_length=20):
    """Кодирование текста в последовательность индексов"""
    words = text.lower().split()
    encoded = []

    for word in words[:max_length]:
        encoded.append(vocab.get(word, vocab['<UNK>']))

    # Добавляем padding если нужно
    if len(encoded) < max_length:
        encoded += [vocab['<PAD>']] * (max_length - len(encoded))

    return encoded[:max_length]


def decode_sequence(sequence, idx_to_word):
    """Декодирование последовательности в текст"""
    words = []
    for idx in sequence:
        if idx == 0:  # <PAD>
            continue
        word = idx_to_word.get(idx, '<UNK>')
        words.append(word)

    return ' '.join(words)

# ========== 3. Обучение модели ==========
def train_generator(model, data_loader, optimizer, device, num_epochs=10):

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in  data_loader:
            inputs = batch.to(device)

            # Для language modeling: предсказываем следующий токен
            # inputs: [batch_size, seq_len]
            # targets: [batch_size, seq_len] (сдвинутые на 1)
            targets = inputs[:, 1:].contiguous()

            # Forward pass
            logits, _ = model(inputs[:, :-1], use_causal_mask=True)

            # Вычисляем loss
            loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)

        print(f"Эпоха {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            test_prompt = "this movie was"
            test_encoded = encode_text(test_prompt, vocab, max_length=5)
            test_tensor = torch.tensor([test_encoded]).to(device)

            generated = model.generate(test_tensor, max_length=10, temperature=0.8)
            generated_text = decode_sequence(generated[0].cpu().numpy(), idx_to_word)
            print(f"  '{test_prompt}...' → '{generated_text}'")

    return losses

# ========== 4. Демонстрация генерации ==========
def demonstrate_generations(model, vocab, idx_to_word, device):
    prompts = [
        "this movie was",
        "i loved this",
        "terrible acting",
        "amazing film",
        "boring and"
    ]

    settings = [
        {"temp": 0.5, "top_k": 10, "name": "Консервативная"},
        {"temp": 1.0, "top_k": 50, "name": "Сбалансированная"},
        {"temp": 1.5, "top_k": 100, "name": "Креативная"}
    ]

    for prompt in prompts:
        print(f"\nПромпт: '{prompt}'")
        print("-" * 40)
        for setting in settings:
            encoded = encode_text(prompt, vocab, max_length=5)
            input_tensor = torch.tensor([encoded]).to(device)

            generated = model.generate(input_tensor, max_length=15, temperature=setting["temp"], top_k=setting["top_k"])

            generated_text = decode_sequence(generated[0].cpu().numpy(), idx_to_word)

            print(f"{setting['name']} (temp={setting['temp']}, top_k={setting['top_k']}):")
            print(f"  '{generated_text}'")

def visualize_generation_process():
    print("\n" + "="*60)
    print("Как работает генерация текста")
    print("="*60)

    steps = [
        ("1. Промпт", "this movie was", "Входная последовательность"),
        ("2. Эмбеддинг", "→ Векторы", "Слова → числа → векторы"),
        ("3. Transformer", "→ Self-Attention", "Обработка контекста"),
        ("4. Предсказание", "→ Вероятности", "Softmax над всем словарём"),
        ("5. Sampling", "→ 'great'", "Выбор следующего слова"),
        ("6. Добавление", "this movie was great", "Повтор с новой последовательностью"),
        ("7. Продолжение", "→ ...", "Повтор до max_length или стоп-токена")
    ]

    for step_num, step_name, example, explanation in steps:
        print(f"{step_num}. {step_name:20s} {example:30s} # {explanation}")

# ========== 6. Финальный проект: система рекомендаций ==========
def final_project():
    """Финальный проект: Система генерации и классификации отзывов"""

    print("\n" + "=" * 60)
    print("🎉 ФИНАЛЬНЫЙ ПРОЕКТ: Генератор отзывов с трансформером")
    print("=" * 60)

    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # 1. Подготовка данных
    print("\n1. Подготовка данных...")
    vocab, idx_to_word = create_vocabulary_from_reviews()
    vocab_size = len(vocab)

    # Создаём тренировочные данные
    training_texts = [
        "this movie was great",
        "i really enjoyed it",
        "fantastic acting and story",
        "terrible waste of time",
        "amazing cinematography",
        "boring and predictable",
        "excellent film",
        "worst movie ever",
        "beautiful storytelling",
        "disappointing ending"
    ]

    encoded_data = []
    for text in training_texts:
        encoded = encode_text(text, vocab, max_length=10)
        encoded_data.append(encoded)

    data_tensor = torch.tensor(encoded_data)
    data_loader = torch.utils.data.DataLoader(data_tensor, batch_size=2, shuffle=True)

    model = CausalTransformerGenerator(
        vocab_size=vocab_size,
        embed_dim=64,
        ff_dim=128,
        num_layers=2,
        max_seq_len=20,
        dropout=0.1
    ).to(device)

    print(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = train_generator(model, data_loader, optimizer, device, num_epochs=5)

    demonstrate_generations(model, vocab, idx_to_word, device)

    # 5. Интерактивная демонстрация
    print("\n" + "="*60)
    print("🔄 ИНТЕРАКТИВНАЯ ДЕМОНСТРАЦИЯ")
    print("="*60)
    print("Попробуйте разные промпты! (английские слова)")
    print("Примеры: good, bad, amazing, terrible, i loved")
    print("Введите 'quit' для выхода")

    while True:
        user_input = input("\nВаш промпт: ").strip().lower()

        if user_input == 'quit':
            break

        if not user_input:
            continue

        # Настройки
        print("\nВыберите стиль генерации:")
        print("1. Консервативный (точный)")
        print("2. Сбалансированный")
        print("3. Креативный")

        try:
            choice = int(input("Ваш выбор (1-3): "))
            if choice == 1:
                temp, top_k = 0.5, 10
            elif choice == 2:
                temp, top_k = 1.0, 50
            else:
                temp, top_k = 1.5, 100
        except:
            temp, top_k = 1.0, 50

        encoded = encode_text(user_input, vocab, max_length=5)
        input_tensor = torch.tensor([encoded]).to(device)

        generated = model.generate(input_tensor, max_length=20, temperature=temp, top_k=top_k)

        generated_text = decode_sequence(generated[0].cpu().numpy(), idx_to_word)

        print(f"\nРезультат:")
        print(f"Промпт: '{user_input}'")
        print(f"Настройки: temp={temp}, top_k={top_k}")
        print(f"Сгенерировано: '{generated_text}'")

        # 6. Сохранение модели
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'idx_to_word': idx_to_word,
            'config': {
                'embed_dim': 64,
                'num_layers': 2,
                'ff_dim': 128,
                'max_seq_len': 20,
                'num_heads': 2,
            }
        }, 'review_generator_transformer.pth')

        print("Модель сохранена в 'review_generator_transformer.pth'")

# ========== 7. Запуск финального проекта ==========
if __name__ == "__main__":
    try:
        final_project()
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        import traceback
        traceback.print_exc()