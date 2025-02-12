import torch
import time
import os
import psutil
from LR_AI_LLM.Model_Structure.Model_Class import LearnrReflectM, LearnrReflectMConfig  # Import your model

# ✅ Ensure PyTorch runs on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training

### 🔹 Function to Check VRAM & CPU RAM Usage ###
def check_memory():
    vram = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    ram = psutil.virtual_memory().used / (1024 ** 3)
    return f"🔹 VRAM Usage: {vram:.2f} GB | 🔹 RAM Usage: {ram:.2f} GB"

# ✅ Print initial memory state
print("🔹 Initial Memory State:", check_memory())

# ✅ Load Model Configuration
config = LearnrReflectMConfig()  # Uses optimized config for single GPU

# ✅ Load Model & Move to GPU
model = LearnrReflectM(config).to(device)
model = torch.compile(model)  # 🔹 Use `torch.compile()` for speed optimization
model = model.to(dtype=torch.float16)  # 🔹 Use mixed precision for lower VRAM

# ✅ Create dummy input for testing (batch=1, seq_len=512)
batch_size = 1
seq_len = 512  # 🔹 Reduce sequence length for single GPU
x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

# ✅ Run Inference & Measure Time
torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    output = model(x)

torch.cuda.synchronize()
end_time = time.time()

# ✅ Print Inference Results
print("🔹 Model Output Shape:", output.shape)
print("🔹 Inference Time:", f"{end_time - start_time:.3f} sec")
print("🔹 Memory After Inference:", check_memory())

# ✅ Save Model Checkpoint (Optional)
torch.save(model.state_dict(), "learnr_reflectm_checkpoint.pth")
print("✅ Model checkpoint saved!")
