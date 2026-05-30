"""
inference_custom.py
"""

import os
import argparse
import torch
from model import LLM, TiktokenHFWrapper
from generate import sample_next_token


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="custom-llm-out")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    # 학습 시 사용한 아키텍처와 일치해야 함
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)
    p.add_argument("--mlp_dim", type=int, default=2048)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--rope_base", type=int, default=10000)
    p.add_argument("--experts", type=int, default=4)
    return p.parse_args()


def load_model(args, tokenizer, device):
    model = LLM(
        dim=args.dim, depth=args.depth, max_len=args.block_size,
        mlp_dim=args.mlp_dim, heads=args.heads, dim_head=args.dim_head,
        vocab_size=tokenizer.vocab_size, padding_idx=tokenizer.pad_token_id,
        experts=args.experts, base=args.rope_base, dropout=0.0,
    )

    bin_path = os.path.join(args.model_dir, "pytorch_model.bin")
    safe_path = os.path.join(args.model_dir, "model.safetensors")

    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=device)
    elif os.path.exists(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path, device=str(device))
    else:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없음: {args.model_dir}")

    model.load_state_dict(state_dict)
    return model.to(device)


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature, top_k, eos_id, device):
    model.eval()
    generated = input_ids.to(device)  # [1, T]

    for _ in range(max_new_tokens):
        logits = model(generated).logits[:, -1, :]  # [1, V]
        next_token = sample_next_token(logits, temperature, top_k)  # [1, 1]

        if next_token.item() == eos_id:
            break

        generated = torch.cat([generated, next_token], dim=1)

    return generated


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = TiktokenHFWrapper("r50k_base")
    model = load_model(args, tokenizer, device)

    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]
    output_ids = generate(
        model, input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_token_id,
        device=device,
    )

    prompt_len = input_ids.shape[1]
    print(tokenizer.decode(output_ids[0, prompt_len:].tolist()))


def main():
    run_inference(parse_args())


if __name__ == "__main__":
    main()
