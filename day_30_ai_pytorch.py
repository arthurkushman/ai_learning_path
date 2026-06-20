import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, pipeline
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU() # Swish activation, common in modern LLMs

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        batch_size, d_model, seq_len = x.shape

        # 1. Calculate router probabilities
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1)

        # 2. Select top-k experts for each token
        topk_weights, topk_indices = torch.topk(routing_weights, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 3. Compute output by routing tokens to selected experts
        final_output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            # Find which tokens selected this expert
            token_indices = (topk_indices == expert_idx).nonzero(as_tuple=True)
            if len(token_indices[0]) == 0:
                continue

            # Extract those tokens, pass through the expert, and scale by weight
            selected_tokens = x[token_indices]
            expert_output = self.experts[expert_idx](selected_tokens)
            weighted_output = expert_output * topk_weights[token_indices]

            # Scatter back to the final output tensor
            final_output[token_indices] += weighted_output

        return final_output


# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # Normal Float 4-bit (best for LLM weights)
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16 for speed/stability
    bnb_4bit_use_double_quant=True,     # Quantize the quantization constants (saves ~0.4 bits/param)
)

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Required for causal LM training

# 2. Load Quantized Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # Automatically distributes layers across available GPUs
    trust_remote_code=True
)

# 3. Prepare model for k-bit training (casts layernorms to FP32 for stability)
model = prepare_model_for_kbit_training(model)

# 4. Configure LoRA
# For MoE, we often target the router and the expert linear layers for maximum code-adaptation impact
lora_config = LoraConfig(
    r=16,                               # Rank of the update matrices (higher = more capacity, more VRAM)
    lora_alpha=32,                      # Scaling factor
    target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], # Layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. Inject LoRA adapters into the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Should show ~1-3% of parameters are trainable

dataset = load_dataset("HuggingFaceH4/no_robots", split="train")

# Formatting function to turn data into chat/completion format
def format_code_example(example):
    return {
        "text": f"<|im_start|>system\nYou are an expert Python coder.<|im_end|>\n<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['completion']}<|im_end|>"
    }

tokenized_dataset = dataset.map(format_code_example).map(
    lambda examples: tokenizer(examples["text"], truncation=True, max_length=2048),
    batched=True, remove_columns=dataset.column_names
)

training_args = TrainingArguments(
    output_dir="./a2b-coder-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True, # Use BF16 if your GPU supports it (Ampere+ architecture)
    max_steps=500,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    optim="paged_adamw_8bit", # Memory-efficient optimizer
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
)

# Load the model and tokenizer for inference
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

prompt = """Write a Python function that implements a binary search algorithm. 
Include type hints and a docstring."""

messages = [
    {"role": "system", "content": "You are A2B-Coder, an expert AI programming assistant."},
    {"role": "user", "content": prompt}
]

# Apply chat template
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate code
outputs = pipe(
    input_text,
    max_new_tokens=256,
    temperature=0.2,       # Low temperature for deterministic, accurate code
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print(outputs[0]["generated_text"])