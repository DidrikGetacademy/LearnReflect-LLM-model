from transformers import PretrainedConfig, GPTNeoXForCausalLM
class LearnrReflectMConfig(PretrainedConfig):
    model_type = "learnr_reflect_m"

    def __init__(
        self,
        vocab_size=64000,
        max_position_embeddings=32768,
        hidden_size=8192,
        num_hidden_layers=60,
        num_attention_heads=64,
        intermediate_size=32768,
        hidden_act="swiGLU",
        rms_norm_eps=1e-5,
        rotary_emb_fraction=1.0,   
        use_flash_attention=True,
        use_multi_query_attention=False,
        hidden_dropout=0.1,
        use_parallel_residual=True,
        layer_norm_eps=1e-6,
        rotary_pct=1.0,
        rotary_emb_base=10000,  
        rope_theta=10000,  
        attention_bias=True,  # Add this line to prevent errors
        _attn_implementation="flash_attention_2",  # Change this if needed
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
        self.hidden_dropout = hidden_dropout
        self.use_parallel_residual = use_parallel_residual
        self.layer_norm_eps = layer_norm_eps
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias  # Assign this value
        self._attn_implementation = _attn_implementation


# Define configuration
config = LearnrReflectMConfig()

# Use a compatible model class and pass the configuration
model = GPTNeoXForCausalLM(config)

save_directory = r"C:\Users\didri\Desktop\LearnReflect Project\LearnReflect-System\LearnReflect_AI_chatbot\LearnReflect_Languge_Model_ChatbotAI\Model_Structure"
model.save_pretrained(save_directory)
# Calculate parameter count
param_count = sum(p.numel() for p in model.parameters())
print(f"GPT-4+ Hypothetical Parameter Count: ~{param_count/1e9:.2f}B")
