# W&B 데이터 요약

엔티티: `choijiwan1229-hansung-science-high-school`  ·  프로젝트 1개

## 프로젝트: `nLoopMoE`  (5 runs)

| run | state | eval/loss | final/eval_loss | final/perplexity | n_params_M | total_flos | train_loss | train_runtime | train_samples_per_second | train_steps_per_second |
|---|---|---|---|---|---|---|---|---|---|---|
| full-n6 | finished | 3.369 | 3.369 | 29.04 | 332.1 | 0 | 4.096 | 7.403e+04 | 35.18 | 0.733 |
| full-n4 | finished | 3.237 | 3.237 | 25.47 | 332.1 | 0 | 3.898 | 5.548e+04 | 46.94 | 0.978 |
| full-n1 | finished | 3.187 | 3.187 | 24.22 | 332.1 | 0 | 3.973 | 2.647e+04 | 98.37 | 2.049 |
| full-n2 | finished | 3.164 | 3.164 | 23.67 | 332.1 | 0 | 3.897 | 3.361e+04 | 77.48 | 1.614 |
| full-n8 | finished | 3.461 | 3.461 | 31.86 | 332.1 | 0 | 4.119 | 9.257e+04 | 28.13 | 0.586 |

**주요 하이퍼파라미터**

| run | ponder_steps | experts | depth | dim | mlp_dim | heads | lr | muon_lr | alpha | ponder_beta | lambda_p | batch_size | block_size | max_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full-n6 | 6 | 4 | 12 | 768 | 3072 | 12 | 0.002532 | 0.01228 | 0.005 | 0.01 | 0.2 | 24 | 768 | 54254 |
| full-n4 | 4 | 4 | 12 | 768 | 3072 | 12 | 0.001598 | 0.008679 | 0.005 | 0.01 | 0.2 | 24 | 768 | 54254 |
| full-n1 | 1 | 4 | 12 | 768 | 3072 | 12 | 0.003168 | 0.02605 | 0.005 | 0.01 | 0.2 | 48 | 768 | 54254 |
| full-n2 | 2 | 4 | 12 | 768 | 3072 | 12 | 0.001475 | 0.01611 | 0.005 | 0.01 | 0.2 | 48 | 768 | 54254 |
| full-n8 | 8 | 4 | 12 | 768 | 3072 | 12 | 0.00118 | 0.008135 | 0.005 | 0.01 | 0.2 | 24 | 768 | 54254 |

- `full-n6` history: train/loss(500pts), eval/loss(56pts), train/aux/balance_loss(500pts), train/aux/base_loss(500pts), train/router/max_pct_global(500pts), train/router/entropy_norm_global(500pts), train/grad_norm/global_l2(500pts), train/learning_rate(500pts)
- `full-n4` history: train/loss(500pts), eval/loss(56pts), train/aux/balance_loss(500pts), train/aux/base_loss(500pts), train/router/max_pct_global(500pts), train/router/entropy_norm_global(500pts), train/grad_norm/global_l2(500pts), train/learning_rate(500pts)
- `full-n1` history: train/loss(500pts), eval/loss(56pts), train/aux/balance_loss(500pts), train/aux/base_loss(500pts), train/router/max_pct_global(500pts), train/router/entropy_norm_global(500pts), train/grad_norm/global_l2(500pts), train/learning_rate(500pts)
- `full-n2` history: train/loss(500pts), eval/loss(56pts), train/aux/balance_loss(500pts), train/aux/base_loss(500pts), train/router/max_pct_global(500pts), train/router/entropy_norm_global(500pts), train/grad_norm/global_l2(500pts), train/learning_rate(500pts)
- `full-n8` history: train/loss(500pts), eval/loss(56pts), train/aux/balance_loss(500pts), train/aux/base_loss(500pts), train/router/max_pct_global(500pts), train/router/entropy_norm_global(500pts), train/grad_norm/global_l2(500pts), train/learning_rate(500pts)
