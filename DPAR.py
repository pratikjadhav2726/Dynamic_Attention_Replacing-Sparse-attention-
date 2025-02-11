import torch
import torch.nn as nn
import torch.nn.functional as F

def dynamic_token_replacement(embeddings, threshold=0.95):
    """
    Given a tensor of token embeddings (seq_len, dim),
    merge tokens that are similar (cosine similarity > threshold)
    by replacing them with the representative embedding.
    
    Args:
        embeddings (Tensor): shape (seq_len, dim)
        threshold (float): similarity threshold
        
    Returns:
        replaced (Tensor): shape (seq_len, dim) with merged embeddings.
    """
    seq_len, dim = embeddings.shape
    replaced = embeddings.clone()
    used = torch.zeros(seq_len, dtype=torch.bool, device=embeddings.device)
    
    for i in range(seq_len):
        if used[i]:
            continue
        for j in range(i + 1, seq_len):
            if not used[j]:
                sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
                if sim.item() > threshold:
                    replaced[j] = embeddings[i]
                    used[j] = True
    return replaced

class DynamicLlamaAttention(nn.Module):
    def __init__(self, orig_attn, threshold=0.95):
        """
        Wrap the original LlamaAttention module with dynamic token replacement
        for the keys and values.
        
        For Llama, the projections are as follows:
          - q_proj: Linear(3072, 3072)
          - k_proj, v_proj: Linear(3072, 1024)
          - o_proj: Linear(3072, 3072) in the original module
        
        In multi-query attention, keys/values are shared across heads.
        Here, we:
          - Reshape q into 32 heads (3072/32 = 96 dims each)
          - Project each q head from 96 → 1024 (using q_to_k)
          - Compute keys and values once (shape (batch, seq_len, 1024))
          - Apply dynamic token replacement on k and v
          - Compute dot-product attention per head, then average across heads
          - Finally, project the averaged output (of shape (batch, seq_len, 1024))
            to 3072 via a new output projection.
        
        Args:
            orig_attn: The original LlamaAttention module.
            threshold (float): Similarity threshold for merging tokens.
        """
        super().__init__()
        self.threshold = threshold
        
        # Reuse the original projection layers.
        self.q_proj = orig_attn.q_proj  # maps 3072 -> 3072
        self.k_proj = orig_attn.k_proj  # maps 3072 -> 1024
        self.v_proj = orig_attn.v_proj  # maps 3072 -> 1024
        
        # We will not use the original o_proj because its input dimension is 3072.
        # Instead, we create a new output projection mapping from 1024 → 3072.
        self.out_proj_new = nn.Linear(1024, 3072)
        
        # Number of heads (for Llama, typically 32).
        self.num_heads = getattr(orig_attn, "num_heads", 32)
        # Query head dimension: 3072 / 32 = 96.
        self.q_head_dim = 3072 // self.num_heads  # e.g., 96
        # Keys/values dimension (as defined by Llama): 1024.
        self.k_dim = 1024
        
        # To compute dot products, we need to project each q head (96 dims) to 1024 dims.
        self.q_to_k = nn.Linear(self.q_head_dim, self.k_dim)
        # Ensure new parameters use the same dtype later (we'll cast the model to FP16 as needed).
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """
        Args:
            hidden_states (Tensor): shape (batch, seq_len, 3072)
            attention_mask (Tensor, optional): mask to add to attention scores.
        
        Returns:
            output (Tensor): shape (batch, seq_len, 3072)
        """
        batch, seq_len, _ = hidden_states.size()
        
        # Compute projections.
        q = self.q_proj(hidden_states)   # (batch, seq_len, 3072)
        k = self.k_proj(hidden_states)   # (batch, seq_len, 1024)
        v = self.v_proj(hidden_states)   # (batch, seq_len, 1024)
        
        # Reshape q into heads.
        # q: (batch, seq_len, 3072) -> (batch, seq_len, num_heads, q_head_dim)
        # Then transpose to (batch, num_heads, seq_len, q_head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        # Project each q head from 96 -> 1024.
        # q_new: (batch, num_heads, seq_len, 1024)
        q_new = self.q_to_k(q)
        
        # For multi-query attention, keys and values are shared.
        # Instead of expanding to all heads, keep them as (batch, seq_len, 1024)
        # and unsqueeze a head dimension (to allow broadcasting).
        k = k.unsqueeze(1)  # (batch, 1, seq_len, 1024)
        v = v.unsqueeze(1)  # (batch, 1, seq_len, 1024)
        
        # Apply dynamic token replacement on k and v for each batch element.
        # (Since keys and values are shared, we perform clustering once per batch element.)
        new_k_list = []
        new_v_list = []
        for b in range(batch):
            replaced_k = dynamic_token_replacement(k[b, 0], threshold=self.threshold)  # (seq_len, 1024)
            replaced_v = dynamic_token_replacement(v[b, 0], threshold=self.threshold)  # (seq_len, 1024)
            new_k_list.append(replaced_k)
            new_v_list.append(replaced_v)
        # Stack to get tensors of shape (batch, seq_len, 1024) and then unsqueeze for head dimension.
        new_k = torch.stack(new_k_list, dim=0).unsqueeze(1)  # (batch, 1, seq_len, 1024)
        new_v = torch.stack(new_v_list, dim=0).unsqueeze(1)  # (batch, 1, seq_len, 1024)
        
        # Compute attention scores.
        # q_new: (batch, num_heads, seq_len, 1024)
        # new_k: (batch, 1, seq_len, 1024) -> broadcast to (batch, num_heads, seq_len, 1024)
        d_k = self.k_dim  # 1024
        scores = torch.matmul(q_new, new_k.transpose(-2, -1)) / (d_k ** 0.5)  # (batch, num_heads, seq_len, seq_len)
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        # Compute attention output.
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        # new_v: (batch, 1, seq_len, 1024) -> broadcast to (batch, num_heads, seq_len, 1024)
        attn_output = torch.matmul(attn_weights, new_v)  # (batch, num_heads, seq_len, 1024)
        
        # Since keys/values are shared, all heads are effectively computing similar outputs.
        # We average over the head dimension.
        attn_output = attn_output.mean(dim=1)  # (batch, seq_len, 1024)
        
        # Project the averaged output back to the hidden dimension (3072).
        output = self.out_proj_new(attn_output)  # (batch, seq_len, 3072)
        return output
# Replace the self-attention modules in each decoder layer.
for i, layer in enumerate(model.model.layers):
    orig_attn = layer.self_attn
    layer.self_attn = DynamicLlamaAttention(orig_attn, threshold=0.95)
    print(f"Replaced self-attention in layer {i}")
model = model.to("cuda")
# Test the modified model.
prompt = "Once upon a time, in a land far, far away,"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))