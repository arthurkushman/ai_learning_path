import torch
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Set, Optional


# ========== 1. Симуляция страниц памяти ==========
@dataclass
class Page:
    id: int
    data: torch.Tensor
    last_access: int = 0


class PagedMemoryManager:
    def __init__(self, num_pages: int, page_size: int, head_dim: int):
        self.num_pages = num_pages
        self.page_size = page_size
        self.head_dim = head_dim
        # Инициализируем все страницы как свободные
        self.free_pages: Set[int] = set(range(num_pages))
        self.page_pool: Dict[int, Page] = {i: Page(i, torch.randn(page_size, head_dim)) for i in range(num_pages)}
        self.page_tables: Dict[str, List[int]] = {}
        self.access_counter = 0

    def allocate(self, request_id: str, num_tokens: int) -> List[int]:
        num_pages_needed = (num_tokens + self.page_size - 1) // self.page_size
        # Если недостаточно свободных страниц, вытесняем
        if len(self.free_pages) < num_pages_needed:
            self._evict_lru(num_pages_needed)
        allocated = []
        for _ in range(num_pages_needed):
            if not self.free_pages:
                raise RuntimeError("Нет свободных страниц, хотя должна быть эвакция")
            page_id = self.free_pages.pop()
            allocated.append(page_id)
        self.page_tables[request_id] = allocated
        return allocated

    def _evict_lru(self, num_needed: int):
        # Собираем все используемые страницы с их временем доступа
        used_pages = []
        for req_id, pages in self.page_tables.items():
            for pid in pages:
                used_pages.append((pid, self.page_pool[pid].last_access))
        # Сортируем по времени доступа (LRU)
        used_pages.sort(key=lambda x: x[1])
        # Вытесняем самые старые
        to_evict = used_pages[:num_needed]
        for pid, _ in to_evict:
            # Удаляем страницу из всех таблиц, где она есть
            for req_id, pages in list(self.page_tables.items()):
                if pid in pages:
                    pages.remove(pid)
            self.free_pages.add(pid)
            print(f"  Вытеснена страница {pid}")

    def read(self, request_id: str, token_pos: int) -> Optional[torch.Tensor]:
        self.access_counter += 1
        pages = self.page_tables.get(request_id, [])
        if not pages:
            return None
        page_idx = token_pos // self.page_size
        if page_idx >= len(pages):
            return None
        page_id = pages[page_idx]
        self.page_pool[page_id].last_access = self.access_counter
        offset = token_pos % self.page_size
        return self.page_pool[page_id].data[offset]

    def write(self, request_id: str, token_pos: int, value: torch.Tensor):
        pages = self.page_tables.get(request_id, [])
        if not pages:
            return
        page_idx = token_pos // self.page_size
        if page_idx >= len(pages):
            return
        page_id = pages[page_idx]
        self.page_pool[page_id].last_access = self.access_counter
        offset = token_pos % self.page_size
        self.page_pool[page_id].data[offset] = value

    def stats(self):
        used = self.num_pages - len(self.free_pages)
        return {
            'used_pages': used,
            'free_pages': len(self.free_pages),
            'utilization': used / self.num_pages * 100,
            'active_requests': len(self.page_tables)
        }


# ========== 2. Continuous Batching Scheduler ==========
class ContinuousBatchingScheduler:
    def __init__(self, max_batch: int = 4):
        self.max_batch = max_batch
        self.running = []
        self.waiting = []

    def add_request(self, req_id: str, num_tokens: int):
        self.waiting.append({'id': req_id, 'remaining': num_tokens})

    def step(self):
        while len(self.running) < self.max_batch and self.waiting:
            req = self.waiting.pop(0)
            self.running.append(req)
            print(f"  ➕ Добавлен запрос {req['id']} (осталось {req['remaining']} токенов)")
        finished = []
        for req in self.running[:]:
            req['remaining'] -= 1
            if req['remaining'] <= 0:
                finished.append(req['id'])
                self.running.remove(req)
        return finished


# ========== 3. Демонстрация ==========
def demo_vllm():
    print("=" * 60)
    print("vLLM: PagedAttention и Continuous Batching")
    print("=" * 60)

    memory = PagedMemoryManager(num_pages=12, page_size=8, head_dim=64)
    requests = [("A", 12), ("B", 20), ("C", 8), ("D", 15)]

    for req_id, tokens in requests:
        try:
            pages = memory.allocate(req_id, tokens)
            print(f"Запрос {req_id}: выделено {len(pages)} страниц")
        except RuntimeError as e:
            print(f"Ошибка для {req_id}: {e}")

    stats = memory.stats()
    print(f"\nСтатистика памяти:")
    print(f"  Использовано: {stats['used_pages']}/{stats['num_pages']} страниц")
    print(f"  Утилизация: {stats['utilization']:.1f}%")
    print(f"  Активных запросов: {stats['active_requests']}")

    print("\nContinuous Batching (динамическое добавление):")
    scheduler = ContinuousBatchingScheduler(max_batch=2)
    for req_id, tokens in requests:
        scheduler.add_request(req_id, tokens)

    step = 0
    while scheduler.running or scheduler.waiting:
        finished = scheduler.step()
        if finished:
            print(f"  ✅ Завершены: {finished}")
        step += 1
        if step > 30:
            break
        print(f"  Шаг {step}: в батче {len(scheduler.running)}, в очереди {len(scheduler.waiting)}")

    print("\nПреимущества PagedAttention:")
    advantages = [
        "• Нет фрагментации памяти (как в виртуальной памяти ОС)",
        "• Можно эффективно делить память между запросами",
        "• Поддерживает длинные контексты (до 1M токенов)",
        "• Увеличение пропускной способности в 2–4 раза"
    ]
    for adv in advantages:
        print(f"  {adv}")


if __name__ == "__main__":
    demo_vllm()