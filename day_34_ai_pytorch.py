import sched

import torch
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict

# ========== 1. Симуляция страниц памяти ==========
@dataclass
class Page:
    id: int
    data: torch.Tensor

class PagedMemoryManager:
    """Управление страницами KV cache"""
    def __init__(self, num_pages: int, page_size: int, head_dim: int):
        self.num_pages = num_pages
        self.page_size = page_size
        self.head_dim = head_dim

        self.pages = [Page(i, torch.randn(page_size, head_dim)) for i in range(num_pages)]
        self.free_pages = set(range(num_pages))
        self.page_tables = {}

    def allocate(self, request_id: str, num_tokens: int) -> List[int]:
        num_pages_needed = (num_tokens + self.num_pages - 1)
        if len(self.free_pages) < num_pages_needed:
            self._evict_lru(num_pages_needed)
        allocated = []
        for i in range(num_pages_needed):
            page_id = self.free_pages.pop()
            allocated.append(page_id)
        self.page_tables[request_id] = allocated
        return allocated

    def _evict_lru(self, num_needed: int) -> None:
        evict = list(self.free_pages)[:num_needed]
        for pid in evict:
            self.free_pages.discard(pid)
        print(f"Evicted pages: {len(evict)}")

    def read(self, request_id: str, token_pos: int) -> torch.Tensor:
        page_idx = token_pos // self.page_size
        pages = self.page_tables.get(request_id, [])
        if page_idx >= len(pages):
            return None
        page_id = pages[page_idx]
        return self.pages[page_id].data[token_pos % self.page_size]

    def write(self, request_id: str, token_pos: int, value: torch.Tensor):
        page_idx = token_pos // self.page_size
        pages = self.page_tables.get(request_id, [])
        if page_idx >= len(pages):
            return None
        page_id = pages[page_idx]
        self.pages[page_id].data[token_pos % self.page_size] = value

    def stats(self):
        used = self.num_pages - len(self.free_pages)
        return {
            "used_pages": used,
            "free_pages": len(self.free_pages),
            "utilization": used / len(self.free_pages) * 100,
            "active_reqs": len(self.free_pages),
        }

# ========== 2. Эмуляция непрерывного батчинга ==========
class ContinuousBatchingScheduler:
    """Упрощённый scheduler для continuous batching"""
    def __init__(self, max_batch: int = 4):
        self.max_batch = max_batch
        self.running = [] # активные запросы
        self.waiting = [] # очередь

    def add_request(self, req_id: str, num_tokens: int):
        self.waiting.append({'id': req_id, 'num_tokens': num_tokens})

    def step(self):
        # Добавляем новые запросы в батч если есть место
        while len(self.running) < self.max_batch and self.waiting:
            req = self.waiting.pop(0)
            self.running.append(req)
            print(f"  ➕ Добавлен запрос {req['id']} (осталось {req['remaining']} токенов)")

        # Симулируем генерацию одного токена для всех
        finshied = []
        for req in self.running[:]:
            req['remaining'] -= 1
            if req['remaining'] <= 0:
                finshied.append(req['id'])
                self.running.remove(req)
        return finshied

# ========== 3. Демонстрация ==========
def demo_vllm():
    print("="*60)
    print("vLLM: PagedAttention и Continuous Batching")
    print("="*60)

    memory = PagedMemoryManager(num_pages=20, page_size=8, head_dim=64)

    requests = [("A", 12), ("B", 20), ("C", 8), ("D", 15)]
    for req_id, tokens in requests:
        memory.allocate(req_id, tokens)

    stats = memory.stats()
    print(f"\nВыделение памяти:")
    print(f"  Использовано страниц: {stats['used_pages']}/{stats['num_pages']}")
    print(f"  Утилизация: {stats['utilization']:.1f}%")
    print(f"  Активных запросов: {stats['active_requests']}")

    # Демонстрация continuous batching
    print("\nContinuous Batching (запросы обрабатываются динамически):")
    scheduler = ContinuousBatchingScheduler(max_batch=2)
    for req_id, tokens in requests:
        scheduler.add_request(req_id, tokens)

    step = 0
    while scheduler.running or scheduler.waiting:
        finished = scheduler.step()
        if finished:
            print(f"  ✅ Завершены: {finished}")
        step += 1
        if step > 20:
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