import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt


# ========== 1. PagedAttention концепция ==========
@dataclass
class Page:
    """Страница KV cache как в виртуальной памяти"""
    page_id: int
    keys: torch.Tensor
    values: torch.Tensor
    is_active: bool = True
    last_accessed: int = 0


class PagedAttentionsSimulator:
    """
    Симуляция PagedAttention — как vLLM управляет памятью
    """

    def __init__(self, num_heads: int = 32, head_dim: int = 128, page_size: int = 16, num_physical_pages: int = 1024):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_physical_pages = num_physical_pages

        # Физическая память
        self.physical_memory = {
            'keys': torch.zeros(num_physical_pages, num_heads, page_size, head_dim),
            'values': torch.zeros(num_physical_pages, num_heads, page_size, head_dim),
        }

        # Таблица страниц для каждого запроса
        self.page_tables = {}  # request_id -> list of page_ids

        self.free_pages = set(range(num_physical_pages))

        # Статистика
        self.page_faults = 0
        self.access_count = 0


def allocate_pages(self, request_id: str, num_tokens: int) -> List[int]:
    num_pages_needed = (num_tokens + self.page_size - 1) // self.page_size

    if len(self.free_pages) < num_pages_needed:
        # Нужно вытеснить страницы (как в ОС)
        self._evict_pages(num_pages_needed)

    allocated = []
    for _ in range(num_pages_needed):
        page_id = self.free_pages.pop()
        allocated.append(page_id)

        self.physical_memory['keys'][page_id].zero_()
        self.physical_memory['values'][page_id].zero_()

    self.page_tables[request_id] = allocated
    return allocated


def _evict_pages(self, num_needed: int):
    """Вытеснение страниц (LRU)"""
    # Собираем все страницы с временем доступа
    all_pages = []
    for req_id, pages in self.page_tables.items():
        for page_id in pages:
            all_pages.append({page_id, self.access_count})

    all_pages.sort(key=lambda x: x[1])

    # Вытесняем самые старые
    for page_id, _ in all_pages[:num_needed]:
        self.free_pages.add(page_id)

        for req_id, pages in list(self.page_tables.items()):
            if page_id in pages:
                pages.remove(page_id)

    self.page_faults += num_needed


def read_tokens(self, request_id: str, token_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
    self.access_count += 1
    page_idx = token_pos // self.page_size
    offset = token_pos % self.page_size

    if request_id not in self.page_tables:
        raise ValueError(f"Request {request_id} not found")

    if page_idx >= len(self.page_tables[request_id]):
        self.page_faults += 1
        return None

    page_id = self.page_tables[request_id][page_idx]

    key = self.physical_memory['keys'][page_id, :, offset, :]
    values = self.physical_memory['values'][page_id, :, offset, :]

    return key, values


def write_token(self, request_id: str, token_pos: int, keys: torch.Tensor, values: torch.Tensor):
    page_idx = token_pos // self.page_size
    offset = token_pos % self.page_size

    page_id = self.page_tables[request_id][page_idx]
    sllf.physical_memory['keys'][page_id, :, offset, :] = keys
    sllf.physical_memory['values'][page_id, :, offset, :] = values


def get_stats(self) -> Dict:
    used_pages = self.num_physical_pages - len(self.free_pages)
    return {
        'used_pages': used_pages,
        'free_pages': self.free_pages,
        'utilization': used_pages / self.num_physical_pages * 100,
        "page_faults": self.page_faults,
        'active_requests': len(self.page_tables),
    }


def demonstrate_paged_attention():
    """Демонстрация PagedAttention"""
    print("=" * 60)
    print("PagedAttention — как vLLM управляет памятью")
    print("=" * 60)

    simulator = PagedAttentionsSimulator(
        num_heads=32,
        head_dim=128,
        page_size=16,
        num_physical_pages=1024
    )

    print(f"\nКонфигурация:")
    print(f"  • Голов: {simulator.num_heads}")
    print(f"  • Размер страницы: {simulator.page_size} токенов")
    print(f"  • Физических страниц: {simulator.num_physical_pages}")

    requests = [
        {"req_1", 50},
        {"req_2", 30},
        {"req_3", 40},
        {"req_4", 60},
        {"req_5", 45},
    ]

    print(f"\nВыделение памяти для запросов:")
    for req_id, num_tokens in requests:
        pages = simulator.allocate_pages(req_id, num_tokens)
        print(f"  • {req_id}: {num_tokens} токенов → {len(pages)} страниц")
        stats = simulator.get_stats()
        print(f"Использовано: {stats['used_pages']}/{simulator.num_physical_pages} страниц")

    print(f"\nСтатистика после выделения:")
    stats = simulator.get_stats()
    print(f"  • Использовано памяти: {stats['utilization']:.1f}%")
    print(f"  • Page faults: {stats['page_faults']}")

    # Симулируем доступ к токенам
    print(f"\nДоступ к токенам (имитация генерации):")
    for req_id, num_tokens in requests[:3]:
        for pos in range(0, num_tokens, 10):
            reset = simulator.reqd_token(req_id, pos)
            if result:
                print(f"  • {req_id}, позиция {pos}: ✓")
            else:
                print(f"  • {req_id}, позиция {pos}: ✗ (page fault)")

    print(f"\nИтоговая статистика:")
    stats = simulator.get_stats()
    print(f"  • Всего page faults: {stats['page_faults']}")
    print(f"  • Утилизация памяти: {stats['utilization']:.1f}%")

    print("\nПреимущества PagedAttention:")
    advantages = [
        "✓ Почти 0 фрагментации памяти",
        "✓ Эффективное sharing KV cache (например, для parallel sampling)",
        "✓ До 4x больше запросов на том же оборудовании",
        "✓ Поддержка длинных контекстов (до 1M токенов)"
    ]

    for adv in advantages:
        print(f"  {adv}")


# ========== 2. Continuous Batching ==========
class ContinuousBatchingScheduler:
    """
    Continuous batching — динамическое формирование батчей
    Вместо ожидания заполнения батча, добавляем запросы "на лету"
    """

    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.running_requests = []
        self.waiting_queue = []
        self.completed = []

    def add_request(self, request_id: str, num_tokens: int):
        self.weiting_queue.append({
            "req_id": request_id,
            "tokens_generated": 0,
            "total_tokens": num_tokens,
            "arrival_time": time.time(),
        })

    def step(self) -> List[str]:
        """Один шаг инференса (генерация одного токена)"""
        # Добавляем новые запросы в running, если есть место
        while len(self.running_requests) < self.max_batch_size and self.waiting_queue:
            new_req = self.waiting_queue.pop(0)
            self.running_requests.append(new_req)
            print(f"  ➕ Запрос {new_req['id']} добавлен в батч")

        if not self.running_requests:
            return []

        # Генерируем один токен для всех в батче
        finished = []
        for req in self.running_requests[:]:
            req['tokens_generated'] += 1

            if req['tokens_generated'] >= req['total_tokens']:
                finished.append(req['req_id'])
                self.running_requests.remove(req)
                self.completed.append(req)

        return finished

    def get_stats(self) -> Dict:
        return {
            'running': len(self.running_requests),
            'waiting': len(self.waiting_queue),
            'completed': len(self.completed),
            'utilization': len(self.completed) / len(self.waiting_queue) * 100,
        }

def demonstrate_continuous_batching():
    """Демонстрация continuous batching"""
    print("\n" + "="*60)
    print("Continuous Batching — динамическое формирование батчей")
    print("="*60)

    scheduler = ContinuousBatchingScheduler(max_batch_size=4)

    requests = [
        ("A", 5), ("B", 8), ("C", 3), ("D", 6), ("E", 11), ("F", 7)
    ]

    print(f"\nИмитация инференса с continuous batching:")
    print(f"Максимальный размер батча: {scheduler.max_batch_size}")

    time_step = 0
    for req_id, num_tokens in requests:
        scheduler.add_request(req_id, num_tokens)
        print(f"\nВремя {time_step}: добавлен запрос {req_id} ({num_tokens} токенов)")

        # Симулируем несколько шагов генерации
        for _ in range(2):
            finished = scheduler.step()
            stats = scheduler.get_stats()
            print(f"  Шаг генерации: в батче {stats['running']}, "
                  f"в очереди {stats['waiting']}, завершено {stats['completed']}")
            if finished:
                print(f"  ✅ Завершены: {finished}")

        time_step += 1

    # Догоняем оставшиеся
    print("\nЗавершение оставшихся запросов:")
    while scheduler.running_requests:
        finished = scheduler.step()
        stats = scheduler.get_stats()

        print(f"В бфтче {stats['running']}, завершено: {stats['completed']}")

        if finished:
            print(f"Завершены: {finished}")

    print("\nПреимущества continuous batching:")
    advantages = [
        "✓ Нет простоя GPU",
        "✓ Меньшая latency для новых запросов",
        "✓ До 2-3x higher throughput",
        "✓ Используется в vLLM, TensorRT-LLM"
    ]

    for adv in advantages:
        print(f"  {adv}")

# ========== 3. Speculative Decoding ==========
class SpeculativeDecoder:
      """
     Speculative Decoding — ускорение инференса через "гадание"
     Маленькая модель предсказывает несколько токенов, большая проверяет
     """
      def __init__(self, draft_model, target_model, gamma: int = 5):
         self.draft_model = draft_model
         self.target_model = target_model
         self.gamma = gamma

      def speculative_step(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
          """
          Один шаг speculative decoding
          Возвращает: новые токены и сколько принято
          """
          # 1. Draft model предсказывает gamma токенов
          draft_tokens = []
          draft_probs = []

          current = input_ids
          for _ in range(self.gamma):
              with torch.no_grad():
                  logits = self.draft_model(current)
                  probs = F.softmax(logits[:, -1, :], dim=-1)
                  next_token = torch.multinomial(probs, num_samples=1)

              draft_tokens.append(next_token)
              draft_probs.append(probs)
              current = torch.cat([current, next_token], dim=1)

          # 2. Target model проверяет все сразу
          with torch.no_grad():
            target_logits = self.target_model(
                torch.cat([input_ids, torch.cat(draft_tokens, dim=1)], dim=1)
            )

          accepted = 0
          final_tokens = []

          for i in range(self.gamma):
              target_prob = F.softmax(target_logits[:, input_ids.size(1) + i, :], dim=-1)
              draft_prob = draft_probs[i]

              r = torch.rand(1).item()
              acceptance_prob = min(1, target_prob[0, draft_tokens[i]] / (draft_prb[0, draft_tokens[i]] + 1e-10))

              if r < acceptance_prob:
                  final_tokens.append(draft_tokens[i])
                  accepted += 1
              else:
                  final_tokens.append(torch.multinomial(target_prob, num_samples=1))

          return torch.cat(final_tokens, dim=1), accepted

    def estimate_speedup(self, num_steps: int = 100) -> float:
        """Оценка ускорения"""
        # draft_time = время инференса draft модели * gamma
        # target_time = время инференса target модели
        # accepted_ratio = среднее количество принятых токенов

        draft_speed = 10
        target_speed = 1

        return (self.gamma * draft_speed + target_speed) / (draft_speed + target_speed)


def demonstrate_speculative_decoding():
    """Демонстрация speculative decoding"""
    print("\n" + "=" * 60)
    print("Speculative Decoding — ускорение через 'гадание'")
    print("=" * 60)

    print("""
    Концепция:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. Draft model (маленькая, быстрая)                     │
    │    предсказывает следующие K токенов                     │
    │                                                         │
    │ 2. Target model (большая, точная)                       │
    │    проверяет все предсказания сразу (параллельно)       │
    │                                                         │
    │ 3. Принимаем совпадающие токены, отвергаем несовпадающие│
    │    и пересчитываем первый отвергнутый                    │
    └─────────────────────────────────────────────────────────┘
    """)

    gamma_values = [3, 5, 7, 10]
    acceptance_rates = [0.7, 0.8, 0.75, 0.6]  # примерные значения

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # График ускорения
    speedups = []
    for gamma, rate in zip(gamma_values, acceptance_rates):
        # Упрощённая формула ускорения
        speedup = (gamma * 10 + 1) / (10 + 1) * rate
        speedups.append(speedup)

    ax1.plot(gamma_values, speedups, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Gamma (число предсказываемых токенов)')
    ax1.set_ylabel('Ускорение')
    ax1.set_title('Зависимость ускорения от gamma')
    ax1.grid(True, alpha=0.3)

    # Принятые токены
    ax2.bar(range(len(gamma_values)), acceptance_rates, tick_label=gamma_values)
    ax2.set_xlabel('Gamma')
    ax2.set_ylabel('Доля принятых токенов')
    ax2.set_title('Acceptance rate')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nРезультаты:")
    for gamma, rate, speedup in zip(gamma_values, acceptance_rates, speedups):
        print(f"  Gamma={gamma}: acceptance rate={rate:.0%}, speedup={speedup:.2f}x")

    print("\nПреимущества speculative decoding:")
    advantages = [
        "✓ 2-3x ускорение без потери качества",
        "✓ Работает с любыми моделями",
        "✓ Особенно эффективно для больших батчей",
        "✓ Используется в DeepMind, Google, Anthropic"
    ]

    for adv in advantages:
        print(f"  {adv}")

# ========== 4. TensorRT Optimization ==========
