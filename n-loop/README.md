# AMoE LLM — MoE 대조군(control) 변형

PyTorch + HuggingFace Trainer + Muon 옵티마이저로 밑바닥부터 만든 디코더-온리 언어모델입니다.
이 working tree는 AMoE 연구의 **대조군(baseline)** 으로, FFN 슬롯이 top-1 라우팅 **Mixture-of-Experts**
이며 이를 **고정 n회 재귀**로 적용합니다(PonderNet식 적응형 halting 없음). 학습 손실은
`task_loss + alpha * load_balance_loss` 입니다. 전체 아키텍처는 [`CLAUDE.md`](CLAUDE.md) 참고.

## 구성

**모델 (단일 정의처):**
- `model.py` — `RoPE, MoE, AMoE, Attention, Transformer, LLM` + `TiktokenHFWrapper` + `MemmapDataset`.
  학습/추론 공용이며 동작 차이는 `self.training`으로만 갈립니다(학습 시 MoE 호출마다 gradient
  checkpointing 적용). `AMoE`는 MoE를 고정 `n`(= `max_steps`)회 **residual 재귀**(`state = state + MoE(state)`)
  로 돌립니다 — halting/조기 종료 없음. FFN 잔차가 `AMoE` 내부에 있어 `Transformer`는 블록 출력을
  그대로 통과시킵니다(이중 잔차 방지). 이 `n`이 곧 *n-loop의 n*이며, `n=1`이면 일반 residual MoE-FFN
  레이어와 같습니다.

**래퍼 / 툴링:**
- `train_custom.py` — 메인 단일-GPU 학습 진입점 (HF `Trainer` + Muon).
- `train_with_hooks.py` — forward-hook 진단 + Switch load-balance aux loss가 붙은 학습.
- `inference_custom.py` — 체크포인트 로드 + 자기회귀 생성.
- `stability_check.py` — 1배치 forward+backward 진단 (NaN / grad / router).
- `optim.py` — `split_params` + `build_muon_optimizer` (Muon vs AdamW 파라미터 분할).
- `config.py` — `add_base_args` (두 학습 진입점이 공유하는 CLI 인자).
- `generate.py` — `sample_next_token` (temperature + top-k 샘플링).
- `diagnostics.py` — `switch_gate_stats` (순수 load-balance / router-collapse / entropy 지표).
- `train_common.py` — `install_signal_handlers` + `init_wandb`.
- `muon.py` — Muon 옵티마이저 라이브러리 (로컬; `from muon import ...`로 임포트).
- `sweep.yaml` — W&B sweep 설정: `lr`, `muon_lr`, `alpha`, `ponder_steps`(= n)를 **random** 탐색.
- `bootstrap_amoe.sh` — 등록된 sweep에 **GPU 1개당 `wandb agent` 1개**로 join (`CUDA_VISIBLE_DEVICES` 핀).
- `launch_agent.py` — 단일 agent 런처 (주의: `train_with_hooks.py`가 아니라 `train_custom.py`를 실행).
- `train_main.sh` — sweep이 아닌 단일 풀-코퍼스 학습 1회 (체크포인트 resume 포함).
- `tests/` — CPU 스모크 + 단위 테스트.

## 설치

```bash
pip install -r requirements.txt          # transformers, wandb, tiktoken, numpy, einops, torch
# muon은 로컬 muon.py로 제공됩니다 — `pip install muon` 하지 마세요 (전혀 다른 패키지입니다).
```

## 데이터 형식

학습은 사전 토크나이즈된 **`uint16` 토큰 샤드**를 `numpy.memmap`으로 읽습니다: `r50k_base`(GPT-2 BPE)
토큰 id의 평탄 바이너리를 `MemmapDataset`이 `block_size` 윈도우로 자릅니다. `train.bin` / `val.bin`을
준비하세요. (CPU 스모크용으로는 합성 가능: `python tests/make_tiny_bin.py --path tiny.bin --n_tokens 20000`.)

## 사용법

```bash
# 학습 (단일-GPU; DDP는 의도적으로 미지원)
python train_custom.py \
  --train_bin_path train.bin --val_bin_path val.bin \
  --project my-wandb-project --run_name my-run

# hook 진단 학습 (router/dispatch 히트맵, balance aux loss)
python train_with_hooks.py --train_bin_path train.bin --val_bin_path val.bin --print_console

# 추론 (아키텍처 플래그는 학습 체크포인트와 일치해야 함)
python inference_custom.py --model_dir custom-llm-out --prompt "Hello" --max_new_tokens 100

# 1배치 안정성 진단
python stability_check.py --train_bin_path train.bin

# 단일 풀-코퍼스 학습 (sweep 아님), 체크포인트 resume
./train_main.sh

# W&B sweep — 한 번 등록(짧은 sweep ID 출력) 후 agent 실행
wandb sweep --project <project> sweep.yaml
echo "<sweep-id>" > .sweep_id          # 위 명령이 출력한 짧은 ID
./bootstrap_amoe.sh                     # GPU 1개당 `wandb agent` 1개 (.sweep_id 사용)
# ...또는 단일 agent:
wandb agent <entity>/<project>/<sweep-id>
```

### 학습 상세

- **토큰 예산 기반**: `max_steps = ceil(max_size / (batch_size × grad_accum × block_size))`. `--epochs`는
  넘겨도 `max_steps`가 우선합니다(HF 스텝 컷오프).
- **옵티마이저**: 2D/3D hidden 가중치는 **Muon**(`--muon_lr`), 그 외(embedding/head/bias/norm)는
  **AdamW**(`--lr`)로 분리.
- **n(재귀 횟수)**: `--ponder_steps`가 `AMoE.max_steps`로 들어갑니다. `--grad_checkpoint`는
  `use_checkpoint`로 매핑. (레거시 `--ponder_beta`/`--lambda_p`는 받지만 대조군에선 **미사용**.)
- **메인 풀런**: `train_main.sh`가 `train_with_hooks.py`를 풀 코퍼스로 1회 실행하고 `--resume 1`로
  마지막 체크포인트에서 재개합니다(비영속 박스 대비). `train.bin`/`val.bin`과 wandb 인증이 필요합니다.

## Sweep

`sweep.yaml`은 **random** 탐색을 쓰며 4개 노브를 튜닝합니다(모델 크기/스케줄은 고정):

| 탐색 항목 | 값 |
|-----------|-----|
| `lr` | log-uniform `1e-4 ~ 5e-3` (AdamW aux 그룹) |
| `muon_lr` | log-uniform `5e-3 ~ 5e-2` (Muon 그룹) |
| `alpha` | `{0.005, 0.01, 0.02, 0.05}` (load-balance 손실 가중) |
| `ponder_steps` (n) | `{1, 2, 4, 6, 8}` — 재귀 깊이; `n=1`이면 일반 MoE |

`bootstrap_amoe.sh`는 `ENTITY` / `PROJECT` / venv 경로가 하드코딩돼 있으니 본인 환경에 맞게 고치세요.
위 표에 없는 인자는 모두 `config.py` 기본값으로 들어갑니다(예: `experts=4`).

## 테스트 (CPU)

모든 테스트는 작은 차원 + 합성 데이터로 CPU에서 돕니다(GPU 불필요):

```bash
pytest tests/ -q
```

커버 범위: 학습 forward/backward, 추론 logits, AMoE `(state, LBL)` 계약 + 고정 n회 재귀,
`MemmapDataset`, 생성, 옵티마이저 분할, 인자 기본값, 샘플링, 진단 지표.

## CPU vs GPU

스케일 학습 외에는 전부 CPU에서 돌아가므로(디바이스 자동 CPU 폴백) 리팩토링·테스트는 CPU에서 싸게
검증합니다. GPU는 실데이터 대규모 학습, `fp16/bf16` 속도, W&B sweep에 쓰세요.
