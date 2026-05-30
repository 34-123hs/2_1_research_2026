"""
tests/make_tiny_bin.py

Write a tiny synthetic uint16 token file for CPU smoke tests, matching the
on-disk format MemmapDataset expects (flat uint16 token ids). This is ONLY for
testing the pipeline on CPU — real training uses the actual r50k_base-tokenized
uint16 data.
"""
import argparse
import numpy as np


def make(path, n_tokens=20000, vocab_size=50257, seed=0):
    """
    input : path (str) output .bin path
            n_tokens (int) number of uint16 tokens to write
            vocab_size (int) token id upper bound (exclusive)
            seed (int)
    output: path (str). Side effect: writes `n_tokens` uint16 values to `path`.
    """
    rng = np.random.default_rng(seed)
    rng.integers(0, vocab_size, n_tokens, dtype=np.uint16).tofile(path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="tiny.bin")
    ap.add_argument("--n_tokens", type=int, default=20000)
    ap.add_argument("--vocab_size", type=int, default=50257)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make(a.path, a.n_tokens, a.vocab_size, a.seed)
    print(f"wrote {a.path} ({a.n_tokens} uint16 tokens, vocab<{a.vocab_size})")
