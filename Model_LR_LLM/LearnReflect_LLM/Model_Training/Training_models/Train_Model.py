
##############################################
# Training Script (e.g., train.py)
##############################################
import os
import torch
import deepspeed
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset  
import sys
import gc
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Model_Structure.Model_Class import LearnrReflectM, LearnrReflectMConfig

# Cleanup GPU memory
gc.collect()
torch.cuda.empty_cache()

# Step 1: Load Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Step 2: Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Dataset Wrapper (using a small block size to save memory)
class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        for text in dataset["train"]["text"]:
            if text.strip():
                tokenized = tokenizer(
                    text,
                    truncation=True,
                    max_length=block_size,
                    padding="max_length",
                    return_tensors="pt"
                )
                self.examples.append(tokenized["input_ids"].squeeze())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return {"input_ids": input_ids, "labels": input_ids.clone()}

# Step 4: Initialize Model Configuration
config = LearnrReflectMConfig(
    vocab_size=tokenizer.vocab_size,
    n_routed_experts=8,
    n_activated_experts=2
)

# Step 5: DeepSpeed Configuration for maximum memory efficiency
ds_config = {
    "train_batch_size": 4,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 4,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
        "sub_group_size": 1e9
    },
    # Replace fp16 with bf16:
    "bf16": {"enabled": True},
    "moe": {
        "enabled": True,
        "trainer_communicate_every": 1,
        "add_router_loss": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 1000,
            "total_num_steps": 10000
        }
    },
    "gradient_clipping": 1.0,
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": True
    },
    "profiling": {
        "enabled": True,
        "profile_memory": True,
        "profile_step": 1,
        "profile_start": 1,
        "profile_end": 3
    }
}

# Step 6: Model Initialization
model = LearnrReflectM(config)

# Step 7: DeepSpeed Initialization (this wraps the model and optimizer)
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config_params=ds_config,
    model_parameters=model.parameters()
)

# Step 8: DataLoader
train_dataset = TextDataset(dataset, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=ds_config["train_micro_batch_size_per_gpu"],
    shuffle=True
)

# Step 9: Training Loop
for epoch in range(3):
    model_engine.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        inputs = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)
        outputs = model_engine(inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, config.vocab_size),
            labels.view(-1)
        )
        model_engine.backward(loss)
        model_engine.step()
        total_loss += loss.item()
        if step % 10 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader):.4f}")

# Step 10: Save Checkpoint
model_engine.save_checkpoint("learnr_reflectm_final")
print("Training complete. Model saved.")



















































