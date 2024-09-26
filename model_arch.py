import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class LlamaConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=16,
        hidden_size=6,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=6,
        hidden_act="silu",
        max_position_embeddings=16,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
            attention_mask = attention_mask.to(dtype=attn_weights.dtype)
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        down = self.down_proj(intermediate)
        return down

class LlamaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LlamaModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        return hidden_states

class LlamaForCausalLM(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {"loss": loss, "logits": logits}

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(input_ids, attention_mask=attention_mask, **kwargs)
