"""
chat.py — 학습된 모델로 텍스트 이어쓰기

사용:
    python chat.py "your prompt"

MODEL_DIR 안의 config.json(학습 시 자동 저장)을 읽어 모델 구조를 복원하므로
별도 하이퍼파라미터 동기화가 필요 없다.
"""

import os
import sys
import json
import torch
import tiktoken
from train import LLM


MODEL_DIR = "/workspace/custom-llm-out"


def load_model(device):
    enc = tiktoken.get_encoding("r50k_base")

    cfg_path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"config.json을 찾을 수 없습니다: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    model = LLM(
        dim=cfg["dim"], depth=cfg["depth"], max_len=cfg["block_size"],
        mlp_dim=cfg["dim"]*4, heads=cfg["heads"], dim_head=cfg["dim"]//cfg["heads"],
        vocab_size=enc.n_vocab, base=cfg["rope_base"], dropout=0.0,
    )

    sf = os.path.join(MODEL_DIR, "model.safetensors")
    pt = os.path.join(MODEL_DIR, "pytorch_model.bin")
    if os.path.exists(sf):
        from safetensors.torch import load_file
        sd = load_file(sf)
    elif os.path.exists(pt):
        sd = torch.load(pt, map_location="cpu")
    else:
        sys.exit(f"가중치 파일을 찾을 수 없습니다: {MODEL_DIR}")

    model.load_state_dict(sd)
    return model.to(device).eval(), enc, cfg["block_size"]


@torch.no_grad()
def generate(model, enc, block_size, prompt, max_new=200, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_len = ids.size(1)

    for _ in range(max_new):
        ctx = ids[:, -block_size:]
        logits = model(ctx).logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == enc.eot_token:
            break

    return enc.decode(ids[0, prompt_len:].tolist())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("사용법: python chat.py \"your prompt\"")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, enc, block_size = load_model(device)
    print(generate(model, enc, block_size, sys.argv[1]))
