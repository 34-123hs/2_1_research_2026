# 인자/설정 레퍼런스

이 프로젝트에서 사용되는 모든 인자, 설정 키, 환경변수를 한 곳에 모은 문서.

---

## 1. `train.py` CLI 인자 (`parse_args`)

학습 본체. argparse로 받음. wandb sweep에서는 `wandb.config`로도 동일 키 주입됨.

### 1.1 입출력/경로

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--project` | str | `None` | wandb project 이름 |
| `--run_name` | str | `None` | wandb run 이름 |
| `--train_bin_path` | str | `train.bin` | 학습용 토큰 memmap 파일 (uint16) |
| `--val_bin_path` | str | `val.bin` | 검증용 토큰 memmap 파일 (uint16) |
| `--output_dir` | str | `custom-llm-out` | 체크포인트·`config.json` 저장 경로 |

### 1.2 데이터/배치

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--block_size` | int | `512` | 모델 시퀀스 길이. RoPE `max_len`도 이 값 |
| `--batch_size` | int | `8` | per-device micro-batch |
| `--grad_accum` | int | `4` | gradient accumulation steps |
| `--max_size` | int | `50_000_000` | 학습 토큰 예산. `max_steps`를 이 값에서 역산 |
| `--max_val_size` | int | `500_000` | 검증 토큰 상한 |

> **주의:** `--epochs`는 지정해도 `max_steps`가 우선되어 무시됨. 학습 길이는 사실상 `--max_size`로 제어.

### 1.3 옵티마이저 (Adam 그룹: embedding/mlp_head/RMSNorm/bias)

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--lr` | float | `3e-4` | Adam 그룹 peak lr |
| `--warmup_steps` | int | `100` | 코사인 스케줄러의 warmup |
| `--weight_decay` | float | `0.1` | 두 그룹 모두에 적용 |

### 1.4 옵티마이저 (Muon 그룹: 그 외 2D 가중치)

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--muon_lr` | float | `0.02` | Muon 그룹 peak lr |
| `--muon_momentum` | float | `0.95` | Newton-Schulz momentum |

### 1.5 모델

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--dim` | int | `512` | 모델 hidden dim |
| `--depth` | int | `6` | Transformer block 수 |
| `--heads` | int | `8` | attention head 수. `dim_head = dim // heads`로 자동 계산 |
| `--rope_base` | int | `10000` | RoPE 주파수 base |
| `--dropout` | float | `0.0` | embedding/attention/FFN dropout |

> **자동 파생**: `dim_head = dim // heads`, `mlp_dim = dim * 4`. 직접 인자 없음.

### 1.6 기타

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--eval_interval` | int | `50` | `eval_steps`. logging_steps는 코드 내 고정값 20 |
| `--epochs` | int | `3` | `num_train_epochs` (실질 무시; 위 주의 참조) |
| `--seed` | int | `576` | torch/numpy/Trainer seed |

---

## 2. `sweep.yaml` 파라미터

wandb sweep 정의. 위 train.py 인자 이름과 1:1 매칭.

| 키 | 타입 | 값 | 비고 |
|---|---|---|---|
| `max_size` | fixed | `50000000` | |
| `max_val_size` | fixed | `200000` | |
| `train_bin_path` | fixed | `train.bin` | |
| `val_bin_path` | fixed | `val.bin` | |
| `output_dir` | fixed | `sweep_outputs` | |
| `epochs` | fixed | `1` | |
| `eval_interval` | fixed | `50` | |
| `block_size` | fixed | `768` | |
| `lr` | log_uniform | `1e-5` ~ `1e-3` | Adam 그룹 lr만 swept |
| `dropout` | values | `[0.0, 0.025, 0.05]` | |
| `batch_size` | values | `[8, 12, 16]` | |
| `grad_accum` | values | `[2, 4, 6, 8]` | |
| `depth` | values | `[12, 16, 20]` | |
| `dim` | values | `[384, 512, 768]` | |
| `warmup_steps` | values | `[100, 150, 200]` | |

> **swept 안 되는 학습 관련 키** (의도 확인 권고): `muon_lr`, `muon_momentum`, `weight_decay`, `heads`, `rope_base`, `seed`.

### 2.1 Sweep 메타

| 키 | 값 | 설명 |
|---|---|---|
| `method` | `bayes` | Bayesian optimization |
| `metric.name` | `eval/loss` | 최소화 대상 |
| `metric.goal` | `minimize` | |
| `early_terminate.type` | `hyperband` | |
| `early_terminate.min_iter` | `10` | |
| `early_terminate.eta` | `3` | |

---

## 3. `launch_agent.py` CLI 인자

DDP/단일 GPU 양쪽에서 wandb agent를 띄우는 런처.

| 인자 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `--sweep_id` | str | ✓ | — | `<entity>/<project>/<sweep_id>` 형식 |
| `--count` | int | | `None` | 이 agent가 처리할 trial 수 (None이면 무한) |
| `--nproc` | int | | `1` | trial당 GPU 수. 1=`python`, 2+=`torchrun --nproc_per_node=K` |

### 3.1 환경변수 (자동 전달)

| 변수 | 설정 위치 | 용도 |
|---|---|---|
| `WANDB_RUN_ID` | launch_agent → child | child train.py가 같은 wandb run에 이어 쓰도록 |
| `WANDB_RESUME=allow` | launch_agent → child | 동일 |
| `RANK`, `WORLD_SIZE` | torchrun → train.py | train.py의 `WORLD_RANK`, `WORLD_SIZE` 전역 |
| `CUDA_VISIBLE_DEVICES` | 사용자 외부 | 병렬 agent 분리 시 |

---

## 4. `chat.py` 설정

추론/생성 스크립트. 학습 시 자동 저장되는 `config.json`을 읽어 모델 구조를 복원하므로 별도 하이퍼파라미터 동기화 불필요.

### 4.1 파일 내 상수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_DIR` | `/workspace/custom-llm-out` | 체크포인트와 `config.json`이 있는 경로 |

### 4.2 함수 인자 (`generate`)

| 인자 | 기본값 | 설명 |
|---|---|---|
| `prompt` | — | 이어쓰기 시작점 (CLI `sys.argv[1]`) |
| `max_new` | `200` | 새로 생성할 최대 토큰 수 |
| `temperature` | `0.8` | softmax 온도 |
| `top_k` | `40` | top-k 샘플링. None이면 비활성 |

생성 중 EOT (`<|endoftext|>` = 50256) 토큰 만나면 즉시 종료.

---

## 5. `config.json` 스키마

`trainer.save_model()` 직후 `output_dir/config.json`에 자동 저장. 내용은 **그 학습에서 사용된 train.py args 전체** (`vars(args)`).

`chat.py`가 모델 복원 시 읽는 키 (최소):

| 키 | 출처 | 용도 |
|---|---|---|
| `dim` | `--dim` | LLM 생성자 |
| `depth` | `--depth` | LLM 생성자 |
| `heads` | `--heads` | `dim_head = dim // heads` |
| `block_size` | `--block_size` | `max_len` + 추론 컨텍스트 길이 |
| `rope_base` | `--rope_base` | RoPE base |

나머지 키(`lr`, `batch_size`, `train_bin_path`, ...)는 저장만 되고 추론 시 사용 안 함 (기록·디버깅 용도).

---

## 6. 주요 환경변수 (분산)

| 변수 | 출처 | 용도 |
|---|---|---|
| `RANK` | torchrun | `train.py:WORLD_RANK = int(os.environ.get("RANK", 0))` |
| `WORLD_SIZE` | torchrun | `train.py:WORLD_SIZE`. `tokens_per_step` 계산에 사용 |
| `WANDB_API_KEY` | `wandb login` | wandb 인증 |
| `WANDB_RUN_ID` / `WANDB_RESUME` | launch_agent | sweep child가 부모 run 이어쓰기 |

---

## 7. 하드코딩 값 (인자화 안 됨)

차후 필요 시 인자화 후보:

| 값 | 위치 | 설명 |
|---|---|---|
| `r50k_base` | `train.py:258`, `chat.py:23` | tiktoken 토크나이저 |
| `mlp_dim = dim * 4` | `train.py:263` | FFN 확장 배수 |
| `dim_head = dim // heads` | `train.py:263` | head dim 자동 계산 |
| `logging_steps=20` | `train.py:296` | Trainer 로깅 주기 |
| `lr_scheduler_type="cosine"` | `train.py:295` | LR 스케줄 종류 |
| `save_total_limit=2` | `train.py:299` | 체크포인트 보관 수 |
