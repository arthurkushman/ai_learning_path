import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from collections import Counter
import re
import warnings

warnings.filterwarnings('ignore')


# ========== 1. Упрощённая предобработка текста (без NLTK) ==========
def simple_preprocess_text(text):
    """Упрощённая очистка текста без NLTK"""
    # Приводим к нижнему регистру
    text = text.lower()

    # Убираем специальные символы, оставляем буквы и пробелы
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    # Разбиваем на слова
    words = text.split()

    # Простой список стоп-слов
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
        'will', 'just', 'should', 'now'
    }

    # Убираем стоп-слова
    words = [word for word in words if word not in stop_words and len(word) > 1]

    return words


def build_vocabulary(texts, max_vocab_size=3000):
    """Построение словаря (vocabulary)"""
    all_words = []

    for text in texts:
        words = simple_preprocess_text(text)
        all_words.extend(words)

    # Считаем частоту слов
    word_counts = Counter(all_words)

    # Берём самые частые слова
    most_common = word_counts.most_common(max_vocab_size - 2)  # -2 для <PAD> и <UNK>

    # Создаём словарь
    vocab = {'<PAD>': 0, '<UNK>': 1}  # PAD для padding, UNK для неизвестных слов
    for word, _ in most_common:
        vocab[word] = len(vocab)

    print(f"Размер словаря: {len(vocab)} слов")
    print(f"10 самых частых слов: {list(word_counts.most_common(10))}")

    return vocab


def text_to_sequence(text, vocab, max_length=100):
    """Преобразование текста в последовательность индексов"""
    words = simple_preprocess_text(text)

    # Ограничиваем длину последовательности
    words = words[:max_length]

    # Преобразуем в индексы
    sequence = []
    for word in words:
        sequence.append(vocab.get(word, vocab['<UNK>']))  # Используем UNK для неизвестных слов

    # Padding до нужной длины
    if len(sequence) < max_length:
        sequence = sequence + [vocab['<PAD>']] * (max_length - len(sequence))

    return sequence[:max_length]  # На всякий случай обрезаем


# ========== 2. Подготовка данных: IMDB Reviews ==========
def load_imdb_data(num_samples=5000):
    """Загрузка и подготовка данных IMDB для классификации тональности"""
    print("Создание синтетических данных IMDB...")

    np.random.seed(42)

    # Позитивные и негативные слова для генерации отзывов
    positive_words = ['excellent', 'amazing', 'wonderful', 'great', 'best',
                      'love', 'fantastic', 'brilliant', 'enjoyed', 'perfect',
                      'awesome', 'outstanding', 'superb', 'marvelous',
                      'good', 'nice', 'pleasant', 'delightful', 'terrific']

    negative_words = ['terrible', 'awful', 'horrible', 'bad', 'worst',
                      'hate', 'boring', 'disappointing', 'poor', 'waste',
                      'disgusting', 'annoying', 'ridiculous', 'stupid',
                      'weak', 'dull', 'predictable', 'slow', 'confusing']

    # Генерация отзывов
    reviews = []
    sentiments = []  # 1 = positive, 0 = negative

    for i in range(num_samples):
        if np.random.random() > 0.5:  # Позитивный отзыв
            sentiment = 1
            num_words = np.random.randint(8, 25)
            words = np.random.choice(positive_words, num_words)
            review = ' '.join(words)
            review = f"I really {np.random.choice(['liked', 'loved', 'enjoyed'])} this movie. {review}. Definitely recommend!"
        else:  # Негативный отзыв
            sentiment = 0
            num_words = np.random.randint(8, 25)
            words = np.random.choice(negative_words, num_words)
            review = ' '.join(words)
            review = f"I was very {np.random.choice(['disappointed', 'upset', 'annoyed'])} with this movie. {review}. Would not recommend."

        reviews.append(review)
        sentiments.append(sentiment)

    # Создаём DataFrame
    df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})

    print(f"Создано {len(df)} отзывов")
    print(f"Позитивных: {df['sentiment'].sum()}")
    print(f"Негативных: {len(df) - df['sentiment'].sum()}")

    # Примеры
    print("\nПримеры отзывов:")
    for i in range(3):
        sentiment = "Позитивный" if df.iloc[i]['sentiment'] == 1 else "Негативный"
        print(f"{i + 1}. [{sentiment}] {df.iloc[i]['review']}")
        print()

    return df


# ========== 3. PyTorch Dataset для текста ==========
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Преобразуем текст в последовательность
        sequence = text_to_sequence(text, self.vocab, self.max_length)

        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32)
        }


# ========== 4. Модели для NLP ==========
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1, num_layers=1, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        last_output = rnn_out[:, -1, :]
        output = self.fc(last_output)
        return output


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1, num_layers=1, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        last_hidden = hidden[-1]
        output = self.fc(last_hidden)
        return output


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1, num_layers=1, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        last_hidden = hidden[-1]
        output = self.fc(last_hidden)
        return output


# ========== 5. Обучение модели ==========
def train_model_nlp(model, train_loader, val_loader, device, num_epochs=5, lr=0.001, model_name="RNN"):
    """Обучение NLP модели"""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nОбучение {model_name}...")

    for epoch in range(num_epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)

                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        # Сохраняем статистику
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Эпоха {epoch + 1}/{num_epochs}: "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    return history


# ========== 6. Основной скрипт ==========
def main():
    print("=" * 60)
    print("День 19: NLP и Рекуррентные нейронные сети (RNN)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # 1. Загрузка данных
    print("\n1. Подготовка данных...")
    df = load_imdb_data(num_samples=1000)  # Используем меньше данных для скорости

    # Разделение
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"\nРазделение данных:")
    print(f"Train: {len(train_df)} отзывов")
    print(f"Val: {len(val_df)} отзывов")
    print(f"Test: {len(test_df)} отзывов")

    # 2. Построение словаря
    print("\n2. Построение словаря...")
    vocab = build_vocabulary(train_df['review'].tolist(), max_vocab_size=2000)

    # 3. Создание датасетов
    print("\n3. Создание датасетов...")
    max_length = 30  # Уменьшаем длину для скорости

    train_dataset = TextDataset(train_df['review'].tolist(), train_df['sentiment'].tolist(), vocab, max_length)
    val_dataset = TextDataset(val_df['review'].tolist(), val_df['sentiment'].tolist(), vocab, max_length)
    test_dataset = TextDataset(test_df['review'].tolist(), test_df['sentiment'].tolist(), vocab, max_length)

    # DataLoader
    batch_size = 16  # Уменьшаем batch для скорости
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Создание моделей
    print("\n4. Создание моделей...")
    vocab_size = len(vocab)

    models = {
        "SimpleRNN": SimpleRNN(vocab_size).to(device),
        "LSTM": LSTMClassifier(vocab_size).to(device),
        "GRU": GRUClassifier(vocab_size).to(device)
    }

    # 5. Обучение и сравнение
    print("\n5. Сравнение архитектур RNN...")
    results = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Модель: {model_name}")
        print(f"{'=' * 50}")

        history = train_model_nlp(
            model, train_loader, val_loader, device,
            num_epochs=3, model_name=model_name  # Всего 3 эпохи для скорости
        )

        results[model_name] = {
            'model': model,
            'history': history,
            'final_val_acc': history['val_acc'][-1]
        }

        # Быстрая визуализация
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Эпоха')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Эпоха')
        axes[1].set_ylabel('Точность (%)')
        axes[1].set_title(f'{model_name} - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'Обучение {model_name}', fontsize=14)
        plt.tight_layout()
        plt.show()

    # 6. Результаты
    print("\n" + "=" * 60)
    print("Сравнение результатов моделей")
    print("=" * 60)

    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Validation Accuracy': [r['final_val_acc'] for r in results.values()]
    }).sort_values('Validation Accuracy', ascending=False)

    print(results_df.to_string(index=False))

    # Визуализация сравнения
    plt.figure(figsize=(8, 5))
    bars = plt.bar(results_df['Model'], results_df['Validation Accuracy'], color=['blue', 'green', 'red'])
    plt.xlabel('Модель')
    plt.ylabel('Точность на валидации (%)')
    plt.title('Сравнение архитектур RNN для анализа тональности')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, results_df['Validation Accuracy']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # 7. Демонстрация работы лучшей модели
    best_model_name = results_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    print(f"\nЛучшая модель: {best_model_name} ({results_df.iloc[0]['Validation Accuracy']:.1f}%)")

    # Тестирование на нескольких примерах
    print("\nТестирование на примерах:")
    best_model.eval()

    test_examples = [
        "I loved this movie it was fantastic and amazing",
        "Terrible waste of time horrible acting",
        "Good movie but could be better",
        "Not bad but not great either",
        "Excellent brilliant superb perfect"
    ]

    for i, text in enumerate(test_examples, 1):
        sequence = text_to_sequence(text, vocab, max_length)
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)

        with torch.no_grad():
            output = best_model(sequence_tensor)
            probability = torch.sigmoid(output).item()
            sentiment = "ПОЗИТИВНЫЙ" if probability > 0.5 else "НЕГАТИВНЫЙ"

            print(f"{i}. '{text[:30]}...' -> {sentiment} (вероятность: {probability:.2%})")

    # 8. Сохранение модели
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'vocab': vocab,
        'max_length': max_length,
        'model_type': best_model_name
    }, f'best_{best_model_name.lower()}_sentiment.pth')

    print(f"\nЛучшая модель сохранена в 'best_{best_model_name.lower()}_sentiment.pth'")

    print("\n" + "=" * 60)
    print("Ключевые выводы дня 19:")
    print("=" * 60)
    print("1. RNN обрабатывают последовательности с памятью")
    print("2. LSTM/GRU имеют gates для управления информацией")
    print("3. Эмбеддинги представляют слова как векторы")
    print("4. Gradient clipping важен для RNN")
    print("=" * 60)

    return results


# ========== 7. Запуск ==========
if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        import traceback

        traceback.print_exc()
