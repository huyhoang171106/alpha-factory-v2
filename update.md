# Alpha Factory Update (2026-04-10)

## 1) Phạm vi research đã clone

### Automation repos (thực dụng)
- `RussellDash332/WQ-Brain` (đã có sẵn): `D:\code\worldquant\WQ-Brain`
- `xiegengcai/world-quant-brain`: `D:\code\worldquant\research\world-quant-brain` @ `97e8c852ad314b90dac90b7484cefda5b92d830a`
- `TonyMa1/wq_new`: `D:\code\worldquant\research\wq_new` @ `2a9ec4010b1aafe9868c926bae0fcc5f9aae37c7`

### Research repos/papers (generation/search nâng cao)
- `QuantaAlpha/QuantaAlpha`: `D:\code\worldquant\research\QuantaAlpha` @ `05d1b3b7174027ef8d79c5547387160037013fd8`
- `BerkinChen/AlphaSAGE` (GFlowNets): `D:\code\worldquant\research\AlphaSAGE` @ `517467a34909512a92d6e3139df54891957560de`
- `nshen7/alpha-gfn` (tiền thân AlphaSAGE): `D:\code\worldquant\research\alpha-gfn-fresh` @ `b0f415c155d3c4e0447f3231bde95f0f95b6d449`
- Tham chiếu bổ sung GP/RL baseline: `D:\code\worldquant\research\AlphaForge`

## 2) Kết luận nhanh cho alpha-factory hiện tại

Kiến trúc hiện tại đã vượt baseline script tuần tự, vì đã có:
- async pipeline (`run_async_pipeline.py`) với queue + worker
- novelty filter sơ cấp + UCB bandit sơ cấp
- circuit breaker mức API trong `wq_client.py`
- submit queue cơ bản trong `tracker.py` + `submit_governor.py`

Nhưng nếu dùng làm core dài hạn thì **chưa đạt production quant-grade** ở 4 lớp:
- **Search**: novelty còn nhị phân (seen/unseen signature), chưa tối ưu orthogonality theo correlation thực sự.
- **Budget**: bandit hiện mới ở bước pre-sim accept/reject; chưa có phân bổ quota simulate theo expected value nhiều tầng.
- **Execution**: queue đang in-memory, chưa có retry policy chuẩn theo error class + DLQ + idempotency thực sự.
- **Governance**: đã có `submit_state` nhưng chưa có state machine đóng kín (`generated -> gated -> queued -> submitted -> accepted/rejected`) với invariant rõ ràng.

## 3) Gap analysis chi tiết (so với hướng bạn chốt)

## 3.1 Search layer
- Hiện có:
  - heuristic score (`alpha_ranker.score_expression`)
  - novelty theo `parameter_agnostic_signature` (`run_async_pipeline.py`)
  - filtering theo similarity/family/theme (`submit_governor.py`)
- Thiếu:
  - objective đa mục tiêu kiểu quality-diversity (Sharpe proxy + novelty + orthogonality)
  - archive theo behavior/correlation bucket (MAP-Elites style)
  - novelty liên tục (distance/corr-based), không chỉ seen/unseen

=> Đánh giá: **đang ở mức “good heuristic filter”, chưa phải “search engine”**.

## 3.2 Budget layer (simulate quota)
- Hiện có:
  - UCB theo arm (`theme:mutation`) trong ranker async
  - giảm tốc khi 429 (cooldown adaptive)
- Thiếu:
  - two-stage budget (cheap test -> expensive sim -> resim/confirm)
  - posterior expected value / Thompson sampling / Bayesian stop rule
  - quota theo minute budget + dynamic SLA của API

=> Đánh giá: **đã có bandit mầm, chưa có budget economy thực sự**.

## 3.3 Execution layer
- Hiện có:
  - async producer/ranker/simulator
  - client retry + backoff + circuit breaker
- Thiếu:
  - queue bền vững (persisted queue) để survive restart
  - retry policy tách theo lỗi (429/5xx/network/semantic 4xx)
  - DLQ + replay workflow
  - idempotency guard cho submit (tránh double submit khi retry)

=> Đánh giá: **functional async, chưa hardened async**.

## 3.4 Governance + observability
- Hiện có:
  - DB cột `submit_state`, `submit_attempts`, timestamps
  - funnel metrics mức cơ bản (`submit_funnel_metrics`)
- Thiếu:
  - state transition validator (không cho nhảy trạng thái sai)
  - trạng thái `gated`, `rejected`, `dead_lettered` chuẩn hóa
  - KPI theo phút: pass-through, queue latency, submit success, novelty ratio, DLQ rate

=> Đánh giá: **đã có data schema nền, chưa có governance policy engine**.

## 4) Những gì đáng học ngay từ các nguồn

- `world-quant-brain`: tách process thành generator/simulator/checker/submitter và status-driven DB flow.
- `wq_new`: package hóa rõ ràng client/core/scripts, clean separation responsibility.
- `QuantaAlpha`: diversified planning + trajectory evolution (không chỉ mutate tuyến tính).
- `AlphaSAGE` / `alpha-gfn`: tối ưu đồng thời quality + diversity bằng novelty reward và pool-based exploration.
- `AlphaForge` + `gplearn warm_start`: warm-start population/memory queue giúp giảm cold-start waste.

## 5) Roadmap nâng cấp ưu tiên (impact cao -> thấp)

## Phase A (1-2 tuần): Governance + Execution hardening
- Mục tiêu:
  - chuẩn state machine + retry policy + DLQ
- Việc cần làm:
  - thêm trạng thái: `gated`, `rejected`, `dead_lettered`
  - tạo transition table hợp lệ và enforce trước mọi update
  - retry theo error class (429 exponential+jitter, 5xx capped retry, 4xx semantic -> reject)
  - DLQ table + replay command
- File chạm chính:
  - `tracker.py`, `submit_governor.py`, `wq_client.py`, `run_async_pipeline.py`

## Phase B (2-3 tuần): Budget economy
- Mục tiêu:
  - giảm simulate rác, tăng hit-rate per API call
- Việc cần làm:
  - split budget 2 tầng:
    - Tier-1: cheap gate (critic + AST novelty + policy)
    - Tier-2: simulate quota allocator (Thompson/UCB + posterior EV)
  - thêm “promote/demote quota” theo rolling performance của arm/family
- File chạm chính:
  - `alpha_policy.py`, `alpha_ranker.py`, `run_async_pipeline.py`, `tracker.py`

## Phase C (3-4 tuần): Search nâng cấp thành quality-diversity
- Mục tiêu:
  - đa dạng cấu trúc thật + orthogonality thật
- Việc cần làm:
  - archive ứng viên theo behavior descriptors (operator family, lookback regime, neutralization style)
  - score đa mục tiêu: expected_quality + novelty + decorrelation_penalty
  - thêm rerank theo “marginal contribution” với pool alpha đã pass
- File chạm chính:
  - `generator.py`, `alpha_ranker.py`, `alpha_ast.py`, `tracker.py`

## 6) KPI vận hành cần theo phút

- `ingest_rate`: số candidate tạo/phút
- `gate_pass_rate`: `% generated -> gated`
- `simulate_yield`: `% simulated -> all_passed`
- `submit_success_rate`: `% queued -> submitted`
- `queue_latency_p50/p95`: từ `queued_at` đến `submitted_at`
- `novelty_ratio`: tỷ lệ candidate mới theo signature/descriptor
- `dlq_rate`: `% job vào dead_letter`

Ngưỡng mục tiêu ban đầu:
- `submit_success_rate >= 90%`
- `queue_latency_p95 <= 5 phút`
- `simulate_yield` tăng ít nhất 25% so với baseline tuần tự hiện tại
- `dlq_rate <= 3%`

## 7) Chốt hướng triển khai

Kết luận của bạn là đúng: giữ pipeline hiện tại làm baseline là hợp lý, nhưng core dài hạn phải chuyển sang:
- **Search**: quality-diversity + novelty/orthogonality
- **Budget**: bandit/Bayesian allocation thật sự theo expected value
- **Execution**: async queue production-grade với retry/circuit breaker/DLQ
- **Governance**: state machine thống nhất + KPI minute-level

Khuyến nghị thực thi ngay: làm **Phase A trước** để ổn định vận hành và đo KPI chuẩn, rồi mới tối ưu search/budget. Nếu tối ưu search trước khi governance ổn định, dữ liệu feedback loop sẽ nhiễu và khó đánh giá hiệu quả thật.

