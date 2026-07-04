# 2D-AMoE Research

Adaptive Mixture-of-Experts(AMoE)를 직접 구현하고, 같은 학습 조건에서 dense Transformer와 PonderNet 계열 baseline을 비교한 연구 저장소입니다.

[발표자료 보기](https://htmlpreview.github.io/?https://github.com/34-123hs/2_1_research_2026/blob/main/Presentation/presentation.html) · [논문 보기](https://htmlpreview.github.io/?https://github.com/34-123hs/2_1_research_2026/blob/main/Paper/2D-AMOE.html)

## 한 줄 요약

AMoE는 MoE의 expert 선택과 PonderNet식 adaptive halting을 한 블록 안에 넣어, 토큰마다 "어떤 expert를 쓸지"와 "얼마나 오래 계산할지"를 함께 결정하도록 만든 구조입니다. 이 저장소는 그 아이디어를 작은 decoder-only LLM 학습 환경에서 실험한 코드와 결과를 담고 있습니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `2D_AMOE/` | 제안 모델 구현. MoE top-1 routing과 adaptive halting을 결합한 AMoE 블록을 포함합니다. |
| `Vanilla/` | 비교용 dense Transformer baseline입니다. adaptive compute나 MoE 없이 같은 데이터 조건에서 학습합니다. |
| `Ponder/Ponder/` | PonderNet baseline 구현입니다. weight-shared core를 여러 번 반복하고 halting distribution을 학습합니다. |
| `Presentation/` | 실험 결과 발표용 HTML 슬라이드와 이미지 자료입니다. |
| `Paper/` | 논문 형식의 HTML 문서, 실험 그래프, 그래프 생성 스크립트입니다. |

## 실험 조건

주요 비교는 다음 조건을 맞춰 진행했습니다.

- decoder-only Transformer 기반의 from-scratch 학습
- GPT-2 BPE 계열 tokenizer(`r50k_base`)
- `train.bin` / `val.bin` 형태의 `uint16` memmap 데이터
- 단일 GPU 학습
- W&B sweep을 통한 learning rate, halting 관련 hyperparameter 탐색
- PyTorch, HuggingFace `Trainer`, Muon optimizer 사용

대용량 데이터와 checkpoint는 Git에 올리지 않습니다. `data/`, `*_checkpoint/`, `*.safetensors`, `*.bin`, W&B 원본 export 디렉터리는 `.gitignore`에서 제외했습니다.

## 실행 예시

### AMoE 학습

```bash
cd 2D_AMOE
python train_custom.py \
  --train_bin_path train.bin \
  --val_bin_path val.bin \
  --project my-wandb-project \
  --run_name amoe-run
```

### 진단용 학습

```bash
cd 2D_AMOE
python train_with_hooks.py \
  --train_bin_path train.bin \
  --val_bin_path val.bin \
  --print_console
```

### Vanilla baseline

```bash
cd Vanilla
python train.py \
  --project my-wandb-project \
  --run_name vanilla-run
```

### PonderNet baseline

```bash
cd Ponder/Ponder
bash run_final.sh
```

## 결과 자료

- 발표 슬라이드: [Presentation/presentation.html](Presentation/presentation.html)
- 논문 문서: [Paper/2D-AMOE.html](Paper/2D-AMOE.html)
- 주요 그래프: `Paper/assets/`
- 발표용 이미지: `Presentation/images/`

GitHub에서 저장소 내부 HTML 링크가 코드로 보일 수 있어, README 상단의 보기 링크는 HTML preview를 거치도록 걸어 두었습니다. 로컬에서는 해당 HTML 파일을 브라우저로 직접 열면 됩니다.

## 메모

`Ponder/` 최상위에는 AMoE 코드에서 복사된 파일들이 일부 남아 있습니다. 실제 PonderNet baseline은 `Ponder/Ponder/` 아래 구현을 기준으로 보면 됩니다.
