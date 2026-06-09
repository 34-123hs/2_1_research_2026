# Eval 분리: 품질(고정 스텝) · 효율(가변 스텝)

학습된 AMoE 체크포인트 평가를 **두 관심사**로 분리하고 결과를 W&B에 로깅한다.

| 파일 | 역할 | 실행 모드 | 보고 지표 | W&B 키 |
|---|---|---|---|---|
| `eval_custom.py` | **품질** | 고정 스텝 | `task_CE + α·LBL`, **`task_CE` 단독**, perplexity | `eval/*` |
| `eval_compute.py` | **효율** | 가변 스텝 | 시간, FLOPs, 평균 ponder step | `bench/*` |

`flops_count.py`(기존)는 그대로 유지한다.

---

## 왜 두 개로 나누나 — 설계 근거

핵심은 "스텝 수에 따라 LBL이 달라져서"가 **아니다.**

`AMoE.forward`(`model.py:176-233`)는 매 스텝 아직 halt 안 된 토큰만 동적으로 추출하고
(`active_tokens = flat[active_mask]`), `if active_tokens.numel() > 0`일 때만 MoE를 호출해
그때만 `total_LBL`을 누적한다(`model.py:198-200`). `sum_certainty`는 단조 증가라,
모든 토큰이 halt된 이후의 스텝은 **고정 스텝(train) 모드에서도** active가 비어 LBL·출력에 0을 더한다.

따라서:

- **`task_CE`, `LBL`, `FLOPs` 는 고정/가변 스텝과 무관하게 동일하다** (동적 추출 때문).
- **유일하게 달라지는 것은 벽시계 시간**(가변은 빈 루프를 건너뜀)**과 평균 ponder step**(적응성)뿐.

→ 분리의 목적은 **관심사·실행 모드·로깅 지표의 분리**다.
품질 잡은 학습 루프 의미와 맞추려 고정 스텝을 *명시적으로* 강제하고(값은 가변과 같음),
효율 잡은 조기 break이 살아있는 가변 스텝에서만 의미 있는 시간/적응성을 잰다.

---

## `eval_custom.py` — 품질 (고정 스텝)

- 보고 loss: `eval_loss = task_CE + α·LBL` (**ponder 항 제외**, LBL 유지).
  학습 loss는 `task_CE + ponder_β·ponderKL + α·LBL`(`model.py:386`)이지만, 대조군과 동일 기준으로
  비교하려 ponder만 뺀다. perplexity는 순수 `exp(task_CE)`(LBL은 NLL이 아니므로 ppl에 미포함).
- `task_CE`를 **단독으로도** 보고/로깅한다.
- `LLM.forward`는 `(loss, logits)`만 돌려주고 LBL을 노출하지 않으므로, forward 본문을 복제하되
  `model.transformer(x) → (x, halting_probs, total_LBL)`의 3-튜플에서 LBL을 직접 꺼낸다. `model.py`는 불변.
- **고정 스텝 강제(surgical):** `model.eval()` 후 AMoE 서브트리만 `amoe.train()` →
  조기 break이 꺼지고 고정 `max_steps`로 돈다. dropout=0이라 `train()`이 켜는 dropout은 no-op,
  `use_checkpoint=False` + `@torch.no_grad()`라 checkpoint/grad 문제 없음. 나머지 모델은 eval 유지.

```bash
python eval_custom.py --ckpt main_out/model.safetensors --val_bin val.bin \
    --project amoe-eval --run_name evalA --max_val_size 50000
```

로깅: `eval/loss`(=task_CE+α·LBL), `eval/task_ce`, `eval/avg_lbl`, `eval/perplexity`, `eval/n_tokens`.

---

## `eval_compute.py` — 효율 (가변 스텝)

`flops_count.py`의 모델 build·`amoe_hook`(평균 ponder step)·`FlopCounterMode` 패턴을 재사용하고
**시간 측정 + W&B 로깅**을 추가한다. eval 모드(가변 스텝)로 실행.

- **시간/FLOPs는 별도 패스로 분리한다.** `FlopCounterMode`는 op를 instrument 하느라 시간을 왜곡하므로:
  - **Pass 1 (시간):** warmup 2회 → `cuda.synchronize` → 타이밍 loop → `synchronize`.
    `time_total_s`, `tokens_per_s`, `ms_per_forward` 산출. (CPU면 `synchronize` 가드)
  - **Pass 2 (FLOPs+step):** `FlopCounterMode` + forward-hook 아래에서 loop 1회(시간 미측정).
    `flops_total`, `flops_per_token`, `avg_ponder_step`.
- FLOPs는 `flops_count.py`와 동일 측정(matmul MAC×2; sort/scatter/elementwise 미포함).

```bash
python eval_compute.py --ckpt main_out/model.safetensors --val_bin val.bin \
    --project amoe-eval --run_name evalB --max_tokens 100000
```

로깅: `bench/time_total_s`, `bench/tokens_per_s`, `bench/ms_per_forward`,
`bench/flops_total`, `bench/flops_per_token`, `bench/avg_ponder_step`, `bench/n_tokens`.

---

## 검증

- 두 파일 byte-compile 통과.
- 교차검증: File B의 `flops_total`/`flops_per_token`은 동일 ckpt·`max_tokens`로 돌린
  `flops_count.py` 결과와 일치해야 하고, `avg_ponder_step ≤ ponder_steps` 여야 한다.
- 주의: 두 스크립트 모두 `model.py`에 의존하므로, `transformers`↔`huggingface_hub` 버전이
  맞는 환경에서 실행해야 한다(`import model`이 되는 환경).
