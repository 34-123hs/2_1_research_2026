"""
generate.py

Sampling for autoregressive generation (wrapper-side, pure). Extracted from the
inner loop of inference_custom.generate so the sampling step is unit-testable on
its own. Behavior is unchanged (temperature scaling + top-k filtering + multinomial).
"""
import torch


def sample_next_token(logits, temperature, top_k):
    """
    Sample the next token id from last-position logits.

    input : logits [B, V]   last-position logits
            temperature (float)  divides logits if != 1.0 (no temp=0 special case)
            top_k (int)          keep only the top-k logits; <=0 disables filtering
    output: next_token [B, 1]    sampled token ids
    """
    if temperature != 1.0:
        logits = logits / temperature                                  # [B, V]
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # [B, k]
        # mask everything below the k-th largest logit to -inf
        logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
    probs = torch.softmax(logits, dim=-1)                              # [B, V]
    return torch.multinomial(probs, num_samples=1)                     # [B, 1]
