##############################################
# Model Definition (Model_Structure/Model_Class.py)
##############################################
import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.distributed as dist
import math  
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func  
from transformers import PretrainedConfig
import deepspeed
from deepspeed.moe.layer import MoE
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

###########################
# 1. Model Configuration
###########################
# This class stores hyperparameters and configurations for the model.
# It defines key parameters such as vocabulary size, hidden size, number of layers/heads, etc.
class LearnrReflectMConfig(PretrainedConfig):
    model_type = "learnr_reflect_m"

    def __init__(
        self,
        vocab_size=50257,  # Number of unique tokens.
        max_position_embeddings=2048,  # Maximum sequence length.
        hidden_size=1024,  # Dimensionality of hidden layers.
        num_hidden_layers=12,  # Number of Transformer blocks.
        num_attention_heads=8,  # Number of attention heads.
        intermediate_size=2048,  # FFN inner dimension.
        expert_dim=512,
        hidden_act="swiGLU",  # Activation function.
        rotary_emb_fraction=1.0,  # Fraction of dimensions to apply RoPE.
        use_flash_attention=True,  # Use FlashAttention.
        layer_norm_eps=1e-6,  # LayerNorm epsilon.
        rope_theta=10000,  # RoPE parameter.
        attention_bias=True,  # Whether attention layers use bias.
        n_routed_experts=8,  # Total number of experts.
        n_activated_experts=2,  # Experts chosen per token.
        moe_inter_dim=2048,  # Intermediate dimension for MoE.
        route_scale=1.0,
        cot_num_thoughts=4,
        cot_attention_heads=8,
        dynamic_learning_ratio=0.3,
        memory_slots=512,  # For the DynamicMemoryBank.
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
        self.expert_dim = expert_dim
        self.rotary_emb_fraction = rotary_emb_fraction
        self.use_flash_attention = use_flash_attention
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.moe_inter_dim = moe_inter_dim
        self.route_scale = route_scale
        self.cot_num_thoughts = cot_num_thoughts
        self.cot_attention_heads = cot_attention_heads
        self.dynamic_learning_ratio = dynamic_learning_ratio
        self.memory_slots = memory_slots

###########################
# 2. Attention & Embedding Modules
###########################
# MetaAttention integrates a memory bank into self-attention.
class MetaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        # Memory bank used to augment key/value vectors.
        self.memory_bank = nn.Parameter(torch.zeros(1, 1024, config.hidden_size))
        nn.init.normal_(self.memory_bank, mean=0.0, std=0.02)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, self.hidden_size, dim=2)
        mem = self.memory_bank.expand(B, -1, -1)
        k = torch.cat([k, mem], dim=1)
        v = torch.cat([v, mem], dim=1)
        qkv_stacked = torch.stack([q, k, v], dim=2)
        x = flash_attn_qkvpacked_func(qkv_stacked, causal=True)
        return self.proj(x)

# FlashAttention wrapper.
class FlashAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
    def forward(self, qkv):
        return flash_attn_qkvpacked_func(qkv, causal=True)

# ThoughtAttention generates multiple thought vectors per token and attends over them.
class ThoughtAttention(nn.Module):
    def __init__(self, hidden_size, num_thoughts=2, num_heads=4):
        super().__init__()
        self.num_thoughts = num_thoughts
        self.hidden_size = hidden_size
        self.generate_thoughts = nn.Linear(hidden_size, num_thoughts * hidden_size)
        self.attend_thoughts = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        thoughts = self.generate_thoughts(x)
        thoughts = thoughts.view(batch_size, seq_len, self.num_thoughts, hidden_size)
        avg_thoughts = thoughts.mean(dim=2)
        attn_output, _ = self.attend_thoughts(avg_thoughts, x, x)
        return attn_output

###########################
# 3. Activation & Expert Modules
###########################
def get_activation(act_name):
    if act_name.lower() == "swiglu":
        return lambda x: x * F.silu(x)
    elif act_name.lower() == "relu":
        return F.relu
    elif act_name.lower() == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Unknown activation: {act_name}")

# Standard two-layer FFN used in MoE.
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_act):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.act = get_activation(hidden_act)
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# Alternative expert for iterative reasoning.
class ExpertFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, act_fn):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.act = get_activation(act_fn)
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

###########################
# 4. Rotary Embedding Utilities
###########################
# Generates frequencies for rotary positional embeddings.
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)))
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_embedding(q, k, theta=10000.0, rotary_emb_fraction=1.0):
    total_dim = q.shape[-1]
    apply_dim = int(total_dim * rotary_emb_fraction)
    if apply_dim % 2 != 0:
        apply_dim -= 1

    q_rot, q_pass = q[..., :apply_dim], q[..., apply_dim:]
    k_rot, k_pass = k[..., :apply_dim], k[..., apply_dim:]

    dim = apply_dim // 2
    freqs = torch.exp(-math.log(theta) * torch.arange(0, dim, 2, device=q.device) / dim)
    pos = torch.arange(q.shape[1], device=q.device).unsqueeze(1)
    freqs = pos * freqs.unsqueeze(0)
    freqs = torch.cat((freqs, freqs), dim=-1)
    
    q_rot = torch.view_as_complex(q_rot.view(*q.shape[:-1], -1, 2))
    k_rot = torch.view_as_complex(k_rot.view(*k.shape[:-1], -1, 2))
    q_rot = torch.view_as_real(q_rot * torch.exp(1j * freqs)).flatten(2)
    k_rot = torch.view_as_real(k_rot * torch.exp(1j * freqs)).flatten(2)

    q_out = torch.cat([q_rot, q_pass], dim=-1)
    k_out = torch.cat([k_rot, k_pass], dim=-1)
    return q_out, k_out

###########################
# 5. Parallel Linear Layers & Efficient Attention
###########################
# These layers support distributed training by synchronizing outputs.
class ColumnParallelLinear(nn.Linear):
    def forward(self, x):
        output = super().forward(x)
        if dist.is_initialized():
            dist.all_reduce(output)
        return output

class RowParallelLinear(nn.Linear):
    def forward(self, x):
        output = super().forward(x)
        if dist.is_initialized():
            dist.all_reduce(output)
        return output

def efficient_attention(qkv):
    return flash_attn_qkvpacked_func(qkv, causal=True)

###########################
# 6. Transformer Blocks & Full Models
###########################
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            bias=config.attention_bias,
            batch_first=True
        )
        self.thought_attn = ThoughtAttention(
            hidden_size=config.hidden_size,
            num_thoughts=4,
            num_heads=8
        )
        self.ffn = MoEWrapper(
            input_dim=config.hidden_size,
            hidden_dim=config.moe_inter_dim,
            num_experts=config.n_routed_experts,
            top_k=config.n_activated_experts,
            hidden_act=config.hidden_act
        )
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        q, k, v = x, x, x
        q, k = apply_rotary_embedding(q, k, theta=self.config.rope_theta, rotary_emb_fraction=self.config.rotary_emb_fraction)
        attn_output, _ = self.attn(q, k, v)
        x = self.norm1(x + attn_output)
        thought_output = self.thought_attn(x)
        x = self.norm3(x + thought_output)
        ffn_output, _ = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class LearnrReflectM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits

###########################
# 7. Mixture of Experts (MoE) Wrapper
###########################
class MoEWrapper(nn.Module):
    """DeepSpeed-MoE Wrapper for Transformer FFN."""
    def __init__(self, input_dim, hidden_dim, num_experts=8, top_k=2, hidden_act="swiGLU"):
        super().__init__()
        experts = nn.ModuleList([Expert(input_dim, hidden_dim, hidden_act) for _ in range(num_experts)])
        self.moe_layer = MoE(
            hidden_size=input_dim,
            expert=experts,
            num_experts=num_experts,
            k=top_k,
            use_residual=True
        )

    def forward(self, x):
        return self.moe_layer(x)

###########################
# 8. Memory & Cognitive Modules
###########################
class DynamicMemoryBank(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory_size = config.memory_slots
        self.hidden_size = config.hidden_size
        self.memory = nn.Parameter(torch.zeros(1, self.memory_size, self.hidden_size))
        self.memory_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 4 * self.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )
        nn.init.normal_(self.memory, mean=0.0, std=0.02)

    def forward(self, x, reset=False):
        B, T, C = x.size()
        if reset:
            self.reset_memory(B)
        expanded_mem = self.memory.expand(B, -1, -1)
        mem_energy = torch.einsum('btc,bmc->btm', x, expanded_mem)
        mem_weights = F.softmax(mem_energy, dim=-1)
        retrieved = torch.einsum('btm,bmc->btc', mem_weights, expanded_mem)
        update_gate = self.memory_gate(torch.cat([x, retrieved], dim=-1))
        new_mem = expanded_mem * (1 - update_gate) + x.mean(1).unsqueeze(1) * update_gate
        self.memory.data = new_mem.detach().mean(0, keepdim=True)
        return torch.cat([x, retrieved], dim=-1)

    def reset_memory(self, batch_size):
        self.memory.data = torch.zeros(batch_size, self.memory_size, self.hidden_size,
                                       device=self.memory.device)

class MetaCognitiveAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.self_aware_proj = nn.Linear(self.hidden_size, self.num_heads)
        self.dynamic_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x, prior_attention=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        awareness = self.self_aware_proj(x.mean(dim=1)).reshape(B, self.num_heads, 1, 1)
        attn = (q @ k.transpose(-2, -1)) * (self.dynamic_scale * awareness)
        attn = attn.softmax(dim=-1)
        if prior_attention is not None:
            attn = attn + prior_attention * 0.3
        output = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return output, attn.detach()

class IterativeReasoningBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_phases = config.cot_num_thoughts
        self.self_attentions = nn.ModuleList([MetaCognitiveAttention(config) for _ in range(self.num_phases)])
        self.thought_merger = nn.Linear(config.hidden_size * self.num_phases, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.moe = MoE(
            hidden_size=config.hidden_size,
            expert=ExpertFFN(config.hidden_size, config.moe_inter_dim, config.hidden_act),
            num_experts=config.n_routed_experts,
            k=config.n_activated_experts
        )

    def forward(self, x):
        residual = x
        phase_outputs = []
        for i in range(self.num_phases):
            x, _ = self.self_attentions[i](x, prior_attention=phase_outputs[-1] if phase_outputs else None)
            x = F.silu(x)
            phase_outputs.append(x)
        merged = self.thought_merger(torch.cat(phase_outputs, dim=-1))
        x = self.norm(merged + residual)
        moe_out, _ = self.moe(x)
        return x + moe_out * self.config.dynamic_learning_ratio

class IntrospectionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quality_estimator = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, 1),
            nn.Sigmoid()
        )
        self.error_correction = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, x):
        quality = self.quality_estimator(x.detach())
        correction = self.error_correction(x * (1 - quality))
        return x + correction * 0.3

class AutonomousLearningGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.learning_gate = nn.Sequential(
            nn.Linear(2 * config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.param_scale = nn.Parameter(torch.ones(config.hidden_size))
        
    def forward(self, primary, secondary):
        gate = self.learning_gate(torch.cat([primary, secondary], dim=-1))
        return primary + gate * self.param_scale * secondary

class ConsciousTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.memory_bank = DynamicMemoryBank(config)
        self.memory_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'reasoning': IterativeReasoningBlock(config),
                'introspection': IntrospectionLayer(config)
            }) for _ in range(config.num_hidden_layers)
        ])
        self.autonomous_gate = AutonomousLearningGate(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x, reset_memory=False):
        x = self.embed(x)
        x = self.memory_bank(x, reset=reset_memory)
        x = self.memory_proj(x)
        for layer in self.layers:
            reason_out = layer['reasoning'](x)
            introspect_out = layer['introspection'](x)
            x = self.autonomous_gate(reason_out, introspect_out)
        return self.head(x)
