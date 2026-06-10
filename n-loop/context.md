# context.md — Session Handoff (AMoE LLM)

> **Resume protocol:** In a new environment, point the LLM at this file. It is self-contained:
> it re-states the working agreement (which lives in the previous machine's `~/.claude` memory and
> does NOT travel with the repo), the project shape, what this session changed, and what's left.
> Canonical files are written in **English**; explain to the user in **Korean** (he reads it faster).

---

## 0. Working agreement (carry this forward — not otherwise in the repo)

- **Model ownership:** The user authors **100% of the model definition** (architecture + forward math —
  `RoPE / MoE / AMoE / Attention / Transformer / LLM`). It is his IP. Claude's role is strictly the
  **wrapper / adapter** layer: HuggingFace Trainer compatibility, GPU/dtype/memory handling, train &
  inference orchestration, and data / optimizer / generation / wandb / sweep glue.
  → Never rewrite, "improve," or re-architect his model math. When restructuring at his request,
  **preserve exact numerics**; only add wrappers, `self.training` guards, comments, framework glue.
  For an in-model behavior change, implement exactly what's asked and **flag it for human review**.
- **Language:** persisted files (code, docs, `CLAUDE.md`, this file) in **English**; chat
  explanations and plans to the user in **Korean**.
- **Comment style the user wants** (apply when documenting model code): dense per-line shape tags
  (e.g. `# [B, N, D]`), a docstring on every method stating **input / output shapes + purpose**,
  PEP8-aspiring. Reference model: his `Chemical_Model` (a separate chemistry project — style sample
  only, not in this repo).
- Coding principles live in `CLAUDE.md` (Think Before Coding / Simplicity First / Surgical Changes /
  Goal-Driven Execution). Follow them.

---

## 1. Project

From-scratch decoder-only **AMoE LLM** (PyTorch + HF Trainer + Muon optimizer). Defining trait: the
FFN slot is an **AMoE** block = top-1 MoE routing (*which* expert) + PonderNet-style adaptive halting
(*how long* — per-token variable compute). Full architecture + philosophy is in `CLAUDE.md`.

> **현재 working tree = MoE 대조군(control).** AMoE의 적응형 halting/PonderNet을 제거하고 FFN
> 슬롯을 고정 N(=max_steps)회 MoE 재귀로 치환한 ablation이다. 손실 = `task_loss + alpha*LBL`,
> certainty/halting_probs/ponder 손실은 전부 제거됨. 주력 AMoE 버전은 git 히스토리에 보존.
> 래퍼/툴링(train/stability/flops/hooks/tests)도 이 대조군 계약에 맞춰 정리됨.

Key files:
- **`model.py`** — ALL model classes (`RoPE, MoE, AMoE, Attention, Transformer, LLM`) +
  `TiktokenHFWrapper` + `MemmapDataset`. **Shared by train & inference; behavior differs only via
  `self.training`.**
- `train_custom.py` — main training entry (HF `Trainer` + Muon optimizer split).
- `train_with_hooks.py` — alt training entry with forward-hook diagnostics + Switch balance loss.
- `inference_custom.py` — inference / generation entry (sampling loop).
- `stability_check.py` — 1-batch forward+backward diagnostic (NaN / grad / router / halting).
- `optim.py` (split/build Muon optimizer), `config.py` (`add_base_args`), `generate.py`
  (`sample_next_token`), `diagnostics.py` (`switch_gate_stats`), `train_common.py`
  (signal/wandb helpers) — pure/shared wrapper modules extracted from the entry points.
- `tests/` — CPU smoke + unit tests (`pytest tests/ -q`, no GPU). `README.md` — usage.
- `muon.py` — Muon optimizer library. `launch_agent.py` + `sweep.yaml` — W&B sweep driver.

---

## 2. What THIS session changed

1. **Merged** the previously-duplicated `custom_model_train.py` + `custom_model_inference.py` into a
   single **`model.py`** ("B" direction). The train file was already a *superset* (it guarded
   checkpointing with `self.training` and returned `halting_probs`); it was adopted verbatim. The two
   old files were **deleted**.
2. **The one behavioral change** (human-review target) — AMoE vertical loop is fixed at `max_steps`
   in training but breaks early at inference once every token has halted:
   ```python
   # AMoE.forward, appended at the end of the `for t in range(self.max_steps)` body:
   # inference: 모든 토큰이 halt되면 남은 스텝은 step_cert=0이라 출력에 무의미 → break
   # train: self.training=True 이므로 이 분기는 절대 안 탐 → 항상 max_steps 고정
   if not self.training and bool((sum_certainty >= 1 - self.eps).all()):
       break
   ```
   Numerically safe: a halted token contributes `step_cert = 0` thereafter; breaking when *all* are
   halted drops only `< eps` of unassigned mass — within the `eps` tolerance. Training never breaks.
3. **Comments**: applied the user's comment style across all of `model.py`. **No logic change** beyond #2.
4. **Rewired imports** in `train_custom.py`, `train_with_hooks.py`, `stability_check.py`,
   `inference_custom.py` → `from model import ...`.
5. **Updated `CLAUDE.md`** architecture section (old two-file references were stale).

---

## 3. Verification status

**DONE (static — the previous environment had no torch installed):**
- `py_compile` clean on all `.py` files.
- No stale `custom_model_*` imports remain (one descriptive mention in `model.py` docstring only).
- **AST parity proof:** 26 methods compared between `model.py` and the original train model —
  **25 are bit-identical** (docstring-stripped); the only differing method is `AMoE.forward`, and
  removing the early-exit `if` block makes it bit-identical too. ⇒ training numerics are provably
  unchanged, the commenting introduced **zero** logic drift, and inference gains exactly the
  requested early-exit and nothing else.

**NOT DONE — run these first on the GPU instance:**
- Runtime smoke: train forward/backward (loss finite & decreasing, no NaN), generation decodes.
- Confirm the inference early-exit actually fires (loop can break before `max_steps`) and logits are finite.

---

## 4. Next steps on the GPU instance

```bash
pip install -r requirements.txt && pip install muon      # muon is NOT in requirements.txt

# smoke train (tiny) — needs pre-tokenized train.bin / val.bin (uint16 memmap):
python train_custom.py --train_bin_path train.bin --val_bin_path val.bin \
  --max_size 200000 --dim 128 --depth 2 --project <wandb_proj> --run_name smoke

# inference smoke:
python inference_custom.py --model_dir custom-llm-out --prompt "Hello" --max_new_tokens 20

# diagnostic (1 batch):
python stability_check.py --train_bin_path train.bin
```
Pass criteria: loss finite & decreasing, no NaN/Inf; `AMoE.forward` inference loop can break before
`max_steps`; generation decodes to text.

---

## 5. CPU-side wrapper refactor — DONE (checkpoints on `main`)

Executed on CPU (torch installed locally) before moving to GPU, as incremental git checkpoints:
- **Stage 0** — `tests/` harness + merge runtime verification (forward/back, early-exit, generate).
- **Stage 1** — `optim.py`: deduped the identical `create_muon_optimizer` from both training entries.
- **Stage 2** — `generate.py` (pure sampling) + `config.py` (`add_base_args` arg dedup).
- **Stage 3** — `diagnostics.py` (`switch_gate_stats`) + `train_common.py` (signal/wandb dedup).
- **Stage 4** — `README.md` + this update.

Ownership boundary held throughout: **`model.py` was NOT touched** by the refactor; only wrapper code
was restructured. All wrapper-side pure logic now lives in standalone, unit-tested modules.

**Optional next step (not done):** repackage the flat modules into an `amoe/` package (rewrites all
imports). Deferred as low-value / higher-risk; do only if requested.
