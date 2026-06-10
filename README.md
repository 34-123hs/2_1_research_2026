# 2D-AMoE 연구 레포

세 가지 **from-scratch 디코더-온리 LLM**을 동일 조건(2.0B 토큰 · 1 epoch · bf16)에서 직접 학습·비교하는 연구 모노레포다.

## 1. 프로젝트 요약

핵심 아이디어는 **"두 개의 축, 하나의 구조"** — *어떤* 전문가를 쓸지(**MoE top-1 라우팅**)와 *얼마나 오래* 계산할지(**PonderNet 식 적응적 halting**)를 트랜스포머의 FFN 슬롯 하나에 결합한 **AMoE(Adaptive Mixture-of-Experts)** 블록을 제안하고, 같은 예산에서 두 대조군과 성능을 비교한다.

| 역할 | 폴더 | 한 줄 설명 |
|---|---|---|
| 실험군 | `2D_AMOE/` | AMoE = MoE 라우팅 + PonderNet halting |
| 대조군 ① | `Vanilla/` | 표준 dense 트랜스포머 (적응 계산 없음) |
| 대조군 ② | `Ponder/` | 정통 PonderNet (적응 계산만, MoE 없음) |
| 결과 | `Presentation/` | 비교 결과 발표 슬라이드 |

**공통 토대:** PyTorch + HuggingFace `Trainer` + **Muon 옵티마이저**, 토크나이저는 tiktoken `r50k_base`(GPT-2 BPE), 데이터는 `uint16` memmap 토큰 샤드(`train.bin` / `val.bin`), 단일 GPU 학습 + **W&B sweep**으로 하이퍼파라미터 탐색. 학습 코퍼스(`.bin`)는 git에 포함되지 않는다(gitignore).

---

## 2. 각 폴더가 무엇인지

### `2D_AMOE/` — 실험군 (AMoE)
제안 모델의 본체. 모든 모델 클래스(`RoPE / MoE / AMoE / Attention / Transformer / LLM`)가 **`model.py` 한 파일**에 정의되며 학습과 추론이 이를 공유한다(동작 차이는 `self.training` 분기로만 발생: 학습은 AMoE 수직 루프를 `max_steps` 고정 실행, 추론은 모든 토큰이 halt되면 조기 종료). 학습·추론·진단·sweep 래퍼가 별도 모듈로 분리돼 있고 CPU 단위 테스트(`tests/`)를 갖춘다.

### `Vanilla/` — 대조군 ① (dense 베이스라인)
적응 계산이 없는 표준 디코더-온리 트랜스포머(RoPE + RMSNorm + SDPA). 모델 정의와 학습 루프가 **`train.py` 한 파일**에 모두 들어 있고, `chat.py`로 추론한다. 학습은 **단일 GPU**로 돈다(코드 안의 `torchrun`/DDP 경로는 쓰지 않는 잔재다).

### `Ponder/` — 대조군 ② (PonderNet)
> **주의 — 폴더가 중첩돼 있다.** `Ponder/` **최상위**는 사실상 `2D_AMOE` 코드의 사본이다(README 동일, `benchmark_batch.py` / `bootstrap_amoe.sh` / `flops_count.py` / `train_main.sh`만 빠져 있음). **실제 PonderNet 베이스라인 구현은 한 단계 안쪽인 `Ponder/Ponder/`** 에 self-contained로 들어 있다.

`Ponder/Ponder/`는 Banino et al. 2021(arXiv:2107.05407)의 정통 PonderNet을 autoregressive LM에 맞춘 구현이다. 하나의 weight-shared 코어를 최대 `ponder_steps`번 반복하며 매 스텝 예측 + halting 확률을 내고, 손실은 기대 per-step loss + KL(halting ‖ geometric prior)로 정의된다. AMoE와의 공정 비교가 목적이며, 논문 충실도까지 검증(`verify_vs_labml.py`)돼 있다.

### `Presentation/` — 결과 발표자료
세 모델의 비교 결과를 담은 **한국어 HTML 슬라이드 덱**. `index.html`이 `presentation.html`로 자동 리다이렉트되며, `images/`에 모델별 학습/평가 곡선과 라우터·halting 분석 그림이 들어 있다. 빌드 과정이 필요 없는 정적 파일이다.

---

## 3. 각 폴더의 구조도

> 트리는 `.git/`, `__pycache__/`, 생성물(`*-out`, `*_outputs` 등)을 생략한 것이다.

### `2D_AMOE/`
```
2D_AMOE/
├── model.py                 # 모델 단일 소스: RoPE/MoE/AMoE/Attention/Transformer/LLM + Tokenizer/Dataset
├── train_custom.py          # 메인 학습 진입 (HF Trainer + Muon)
├── train_with_hooks.py      # 진단 학습: forward-hook(라우터/halting) + Switch balance aux loss
├── inference_custom.py      # 체크포인트 로드 + 자동회귀 생성
├── generate.py              # sample_next_token (temperature + top-k)
├── stability_check.py       # 1-batch forward+backward 진단 (NaN/grad/router/halting)
├── diagnostics.py           # switch_gate_stats (load-balance/router-collapse/entropy)
├── check_model.py           # 모델 점검 스크립트
├── flops_count.py           # FLOPs 측정
├── benchmark_batch.py       # 배치 처리량 벤치마크
├── optim.py                 # split_params + build_muon_optimizer (Muon vs AdamW 분할)
├── muon.py                  # Muon 옵티마이저 라이브러리 (로컬)
├── config.py                # add_base_args (학습 진입점 공유 CLI 인자)
├── launch_agent.py          # W&B sweep agent 런처
├── sweep.yaml               # W&B sweep 정의
├── train_main.sh            # 비영속 GPU 박스용 풀-코퍼스 학습 스크립트
├── bootstrap_amoe.sh        # 환경 부트스트랩
├── model_check_colab.ipynb  # Colab 점검 노트북
├── requirements.txt
├── tests/                   # CPU 스모크 + 단위 테스트
│   ├── conftest.py  make_tiny_bin.py  test_config.py  test_diagnostics.py
│   └── test_generate.py  test_optim.py  test_smoke.py
├── skills/karpathy-guidelines/
├── README.md  CLAUDE.md  context.md  EXAMPLES.md
└── .gitignore
```

### `Vanilla/`
```
Vanilla/
├── train.py             # 모델 정의 + 학습 루프 (단일 파일)
├── train_with_hooks.py  # 진단 훅이 붙은 학습 변형
├── chat.py              # 추론/이어쓰기 (config.json 자동 로드)
├── muon.py              # Muon 옵티마이저
├── launch_agent.py      # W&B sweep agent 런처 (단일 GPU)
├── sweep.yaml           # W&B sweep 정의
├── requirements.txt
├── NOTES.md             # 실행 시나리오 메모 (vast.ai 기준)
├── ARGS.md              # 모든 인자/설정/환경변수 레퍼런스
├── CLAUDE.md
└── .gitignore
```

### `Ponder/`
```
Ponder/
├── (2D_AMOE 코드의 사본)   # model.py, train_custom.py, inference_custom.py, generate.py,
│                           # stability_check.py, diagnostics.py, optim.py, muon.py, config.py,
│                           # launch_agent.py, sweep.yaml, tests/, README.md, CLAUDE.md, context.md …
│
└── Ponder/                 # ★ 실제 PonderNet 베이스라인 (self-contained)
    ├── pondernet.py          # 정통 PonderNet 모델 (halting 분포 + KL prior)
    ├── train_pondernet.py    # PonderNet 학습 진입
    ├── check_pondernet.py    # E2E + 타이밍 점검
    ├── verify_vs_labml.py    # labml_nn 대비 손실 충실도 검증
    ├── measure_flops.py      # FLOPs 측정
    ├── layers.py             # Attention 등 공용 레이어
    ├── data.py               # memmap 데이터셋
    ├── common.py  config.py  optim.py  muon.py
    ├── sweep.yaml  sweep_opt4.yaml      # W&B sweep 정의
    ├── run_final.sh          # 최종 풀학습(2.0B) 실행
    ├── setup_env.sh          # 컨테이너 리셋 후 의존성 복원
    ├── bootstrap_h100.sh  bootstrap_b300.sh
    ├── TRAINING_METHODOLOGY.md  # 학습 처리량 튜닝 기록(H100 기준)
    └── .gitignore
```

### `Presentation/`
```
Presentation/
├── presentation.html   # 발표 슬라이드 본체 (한국어)
├── index.html          # presentation.html 로 자동 리다이렉트
├── images/
│   ├── AMOE_train.png   AMOE_eval.png      # 실험군 곡선
│   ├── Ponder_train.png Ponder_eval.png    # 대조군 ② 곡선
│   ├── Vanilla_train.png Vanilla_eval.png  # 대조군 ① 곡선
│   ├── AMOE_etc/  AMOE_vertex/             # 라우터/halting 등 추가 분석 그림
│   └── compare/                            # 모델 간 비교 그림
├── LICENSE
├── CLAUDE.md
└── .gitignore
```

---

## 4. 각 폴더의 상황에 따른 실행 방법

> 아래 명령은 각 폴더의 하위 문서(괄호로 출처 표기)에서 그대로 인용한 것이다. 경로/엔티티/프로젝트명은 환경에 맞게 바꿔 쓴다.

### 공통 사전 준비
```bash
pip install -r requirements.txt   # transformers, wandb, tiktoken, numpy, einops, torch
# Muon은 폴더 안 muon.py(로컬)로 제공된다. `pip install muon` 금지 (무관한 패키지임)
wandb login                        # W&B 토큰 입력

# 학습/검증 데이터: uint16 memmap 토큰 샤드 train.bin / val.bin 을 작업 폴더에 배치
# (CPU 스모크용 더미 데이터 합성 — 2D_AMOE/Ponder 한정)
python tests/make_tiny_bin.py --path tiny.bin --n_tokens 20000
```

### `2D_AMOE/` (출처: `README.md`, `context.md`, `train_main.sh`)
```bash
# 학습 (단일 GPU 전용 — DDP 미지원)
python train_custom.py \
  --train_bin_path train.bin --val_bin_path val.bin \
  --project my-wandb-project --run_name my-run

# 진단 학습 (라우터/halting 히트맵 + balance aux loss)
python train_with_hooks.py --train_bin_path train.bin --val_bin_path val.bin --print_console

# 추론 (아키텍처 플래그는 학습된 체크포인트와 반드시 일치)
python inference_custom.py --model_dir custom-llm-out --prompt "Hello" --max_new_tokens 100

# 1-batch 안정성 진단 / CPU 테스트
python stability_check.py --train_bin_path train.bin
pytest tests/ -q

# W&B sweep
wandb sweep --project <project> sweep.yaml
wandb agent <entity>/<project>/<sweep-id>

# 비영속 GPU 박스에서 풀-코퍼스 한 번에 (git clone → venv → corpus 확인 → 학습, ~18-23h)
./train_main.sh
nohup ./train_main.sh > main.log 2>&1 &   # 백그라운드
```
> CPU에서도 (스케일 학습을 제외한) 모든 것이 돌아간다 — device가 자동으로 CPU로 떨어지므로 리팩터/테스트는 CPU에서 저렴하게 검증하고, GPU는 실데이터 대규모 학습·fp16 속도·sweep에만 쓴다.

### `Vanilla/` (출처: `NOTES.md`, `ARGS.md`)
```bash
# 학습 (단일 GPU)
python train.py --project my-llm-project --run_name "single-gpu-baseline"

# W&B sweep 등록 + agent
wandb sweep --project my-llm-project sweep.yaml
wandb agent <entity>/<project>/<sweep-id> --count 10

# 추론 (chat.py 안 MODEL_DIR 의 config.json 을 읽어 모델 구조 자동 복원)
python chat.py "your prompt"
```
> 인자 전체 의미·기본값·환경변수는 `Vanilla/ARGS.md`에 표로 정리돼 있다. `torchrun`/DDP, `launch_agent.py --nproc K` 같은 멀티 GPU 경로는 코드에 남아 있어도 **쓰지 않는 잔재**다 — 학습은 단일 GPU로 한다.

### `Ponder/Ponder/` — 실제 PonderNet (출처: `TRAINING_METHODOLOGY.md`, `run_final.sh`)
```bash
# (먼저 cd Ponder/Ponder)
bash setup_env.sh    # 컨테이너 리셋 후 의존성 복원 (torch>=2.12 등을 conda에 설치)

# 최종 풀학습 (2.0B 토큰, bf16 + torch.compile + ckpt off — 스크립트 폴더에서 train.bin/val.bin 탐색)
bash run_final.sh

# 수동 학습 (bare python 금지 — torch가 없는 /usr/bin 으로 잡힌다)
/opt/conda/bin/python train_pondernet.py \
  --train_bin_path train.bin --val_bin_path val.bin \
  --dim 768 --core_depth 12 --ponder_steps 8 --heads 12 --dim_head 64 --mlp_dim 3072 \
  --block_size 768 --batch_size 8 --grad_accum 6 --lr 0.00296 \
  --max_size 2000000815 --warmup_steps 150 --eval_interval 200 \
  --project custom-llm --run_name "PonderNet baseline"

# E2E + 타이밍 점검 / 논문 충실도 검증
/opt/conda/bin/python check_pondernet.py --core_depth 12 --ponder_steps 8 --block_size 768
python verify_vs_labml.py

# W&B sweep: sweep.yaml (또는 sweep_opt4.yaml) 사용
```
> `Ponder/` **최상위**의 코드는 `2D_AMOE`와 동일하므로 실행 방법도 위 `2D_AMOE/` 절과 같다(중복 사본). PonderNet 베이스라인 작업은 반드시 `Ponder/Ponder/` 안에서 한다.

### `Presentation/`
빌드가 필요 없다. `presentation.html`을 브라우저로 직접 열거나, `index.html`을 열면 자동으로 넘어간다. 이미지 경로 문제가 있으면 로컬 서버로 띄운다.
```bash
# (cd Presentation)
python -m http.server 8000   # 이후 http://localhost:8000/presentation.html 접속
```
