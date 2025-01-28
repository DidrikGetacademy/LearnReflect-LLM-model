#Below is an example of a smaller config—we’ll call it LearnrReflectMini—that you can use to test your pipeline quickly. The idea is to:

#Dramatically reduce the number of layers (e.g., 12 vs. 70).
#Use a smaller hidden dimension (e.g., 768 vs. 10240).
#Scale down intermediate sizes, heads, etc.
#This should get you in the 100M–200M parameter range, so you can confirm that your tokenizer, data loading, training loop, and logging all work before you commit to the full ~70‑layer behemoth.
#Notes:
#Hidden size: Set to 768, which is common in smaller GPT‑like models (e.g., GPT‑2 small).
#Number of layers = 12: This is akin to GPT‑2 small or BERT base scale.
#Intermediate size = 3072: Follows the “4× hidden” rule of thumb.
#Heads = 12: Divides 768 evenly (768 / 12 = 64).
#Vocab size = 32000: You could use 32000 or 64000—lower is often enough for a small test.
#Max position embeddings = 2048: Enough to test chunked training or attention over a decent context, while not being too big.
#Use rotary: You can still keep rotary_emb_fraction=1.0 to confirm the code path for rotary embeddings is fine.
#In practice, this small config should land around 100M–150M parameters (depending on exact final shapes) and let you:

#Tokenize a dataset.
#Load batches and verify memory usage / speed.
#Train for a few steps to confirm loss is decreasing.
#Debug any pipeline issues quickly.
#Once you’ve validated that the pipeline is stable, you can scale back up to your “dream” 70‑layer, 10240‑hidden LearnrReflectM+ monster!

from transformers import PretrainedConfig, GPTNeoXForCausalLM

class LearnrReflectMini(PretrainedConfig):
    model_type = "LearnrReflectMini"

    def __init__(
        self,
        vocab_size=32000,
        max_position_embeddings=2048,
        hidden_size=768,         # Much smaller
        num_hidden_layers=12,    # Reduced layer count
        num_attention_heads=12,  # Must divide hidden_size evenly
        intermediate_size=3072,  # 4 * hidden_size
        hidden_act="gelu_new",   # simpler activation for testing
        rms_norm_eps=1e-5,
        rotary_emb_fraction=1.0, # you can still test rotary 
        # optional flags (turn them off or keep them for testing):
        use_flash_attention=False,
        use_multi_query_attention=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rotary_emb_fraction = rotary_emb_fraction
        self.use_flash_attention = use_flash_attention
        self.use_multi_query_attention = use_multi_query_attention

# Instantiate & check parameter count
config = LearnrReflectMini()
model = GPTNeoXForCausalLM(config)
param_count = sum(p.numel() for p in model.parameters())
print(f"LearnrReflectMini Parameter Count: ~{param_count/1e6:.1f}M")
