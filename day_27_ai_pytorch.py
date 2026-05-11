import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

# ========== 1. CLIP архитектура (упрощённая) ==========
class CLIPTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 49408, d_model: int = 512, max_seq_len: int = 77, num_layers: int = 6) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048, batch_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projection = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(tokens) + self.positional[:, :tokens.size(1), :]

        x = self.transformer(x, src_key_padding_mask=mask)

        x = x[:, -1, :]

        x = self.projection(x)

        return x

class CLIPImageEncoder(nn.Module):
    def __init__(self, d_model: int = 512) -> None:
        super().__init__()

        self.patch_size = 16
        self.image_size = 256
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=self.patch_size, stride=self.patch_size)

        # Positional and CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.projection = nn.Linear(d_model, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [batch, 3, 224, 224]
        batch_size = images.shape[0]

        patches = self.patch_embed(images)
        patches = patches.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x = x + self.pos_embed

        x = self.transformer(x)

        x = x[:, 0, :]

        x = self.projection(x)

        return x

class SimpleCLIP(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.text_encoder = CLIPTextEncoder()
        self.image_encoder = CLIPImageEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features * text_features.t()
        logits_per_text = image_features.t()

        return logits_per_image, logits_per_text

class ClIPDemo:
    def __init__(self):
        try:
            import clip
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.real_clip = True
            print("Real ClIP model loaded")
        except:
            print("Simple ClIP model loaded")
            self.real_clip = False
            self.real_clip = False
            self.model = SimpleCLIP()

    def zero_shot_classification(self, image_path: str, class_names: List[str]) -> Dict:
        if self.real_clip:
            import clip
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

            text = clip.tokenize([f"a photo of a {c}" for c in class_names]).to(self.device)

            with torch.no_grad():
                logits_per_image, _ = self.model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        else:
            probs = np.random.dirichlet(np.ones(len(class_names)))

        return {class_names[i]: probs[0][i] for i in range(len(class_names))}

    def find_best_image(self, images: List[str], query: str) -> str:
        """Находит изображение, наиболее подходящее под текст"""
        print(f"\nПоиск изображения для: '{query}'")

        if self.real_clip:
            import clip
            text = clip.tokenize([query]).to(self.device)

            similarities = []
            for img_path in images:
                image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image)
                    text_features = self.model.encode_text(text)

                    similarity = (image_features @ text_features.T).item()

                    similarities.append(similarity)

            best_idx = np.argmax(similarities)
            return images[best_idx]
        else:
            return images[0] if images else ""

    def visualize_embeddings(self, texts: List[str], images: List[str]):
        """Визуализация CLIP эмбеддингов в 2D"""
        print("\nВизуализация эмбеддингов в 2D (t-SNE)")

        if not self.real_clip:
            print("Для полной демонстрации требуется установка CLIP")
            return

        import clip
        from sklearn.manifold import TSNE

        all_embeddings = []
        labels = []

        for text in texts:
            text_tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                all_embeddings.append(text_features.cpu().numpy())
                labels.append(f"T: {text[:20]}")

        for img_path in images:
            image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                all_embeddings.append(image_features.cpu().numpy())
                labels.append(f"T: {img_path[:15]}")

        all_embeddings = np.array(all_embeddings).squeeze()

        # t-SNE для визуализации
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))

        embeddings_2d = tsne.fit_transform(all_embeddings)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100)

        for i, label in enumerate(labels):
            ax.annotate(label, embeddings_2d[i, 0], embeddings_2d[i, 1], fontsize=8)

        ax.set_title('CLIP Embeddings Space (Text + Images)')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def demonstrate_clip():
        """Демонстрация CLIP"""
        print("=" * 60)
        print("CLIP: Contrastive Language-Image Pre-training")
        print("=" * 60)

        clip_demo = CLIPDemo()

        # Zero-shot классификация
        print("\n1. Zero-shot классификация:")
        classes = ["cat", "dog", "bird", "car", "house"]

    print("Zero-shot классификация работает без обучения!")

    # Как CLIP обучается
    print("\n2. Обучение CLIP:")
    print("""
    Contrastive Learning Objective:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Текстовые эмбеддинги                      │
    │         T1    T2    T3    T4    T5    T6                    │
    │    ┌─────────────────────────────────────┐                 │
    │ I1 │  ✓    ✗    ✗    ✗    ✗    ✗         │                 │
    │ I2 │  ✗    ✓    ✗    ✗    ✗    ✗         │                 │
    │ I3 │  ✗    ✗    ✓    ✗    ✗    ✗         │  ✓ = match      │
    │ I4 │  ✗    ✗    ✗    ✓    ✗    ✗         │  ✗ = mismatch   │
    │ I5 │  ✗    ✗    ✗    ✗    ✓    ✗         │                 │
    │ I6 │  ✗    ✗    ✗    ✗    ✗    ✓         │                 │
    │    └─────────────────────────────────────┘                 │
    │                                                             │
    │ Loss = максимизировать diagonal, минимизировать off-diagonal│
    └─────────────────────────────────────────────────────────────┘
    """)

    print("\nПрименения CLIP:")
    applications = [
        "• Zero-shot классификация изображений",
        "• Поиск изображений по тексту",
        "• Генерация изображений (DALL-E, Stable Diffusion)",
        "• Image captioning",
        "• VQA (Visual Question Answering)"
    ]

    for app in applications:
        print(f"  {app}")

# ========== 2. Flamingo — мультимодальный LLM ==========
class FlamingoLayer(nn.Module):
    """
    Flamingo от DeepMind
    Сочетает vision encoder, text LLM и cross-attention между ними
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()

        # Cross-attention между текстом и изображениями
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        # Self-attention для текста
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)

        #FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, text: torch.Tensor, visual: torch.Tensor, visual_mask: torch.Tensor = None) -> torch.Tensor:
        """
        text: [batch, text_len, d_model]
        visual: [batch, num_images, d_model]
        """
        # Cross-attention: текст внимает к изображениям
        cross_out, _ = self.cross_attention(text, visual, visual, key_padding_mask=visual_mask)
        text = self.cross_norm(text + cross_out)

        # Self-attention внутри текста
        self_out, _ = self.self_attention(text, text, text)
        text = self.self_norm(text + self_out)

        #FFN
        ffn_out = self.ffn(text)
        text = self.ffn_norm(text + ffn_out)

        return text
