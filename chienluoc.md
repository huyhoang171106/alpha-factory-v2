# Chiến lược vận hành Alpha Factory để đạt 100 alpha/day (Sharpe > 1.5)

## 1) Kết luận nhanh sau khi rà soát code + chạy dry-run

Hệ thống hiện tại đã đi đúng hướng và đã triển khai được phần lớn 5 hạng mục lõi. Tuy nhiên, còn chênh giữa "đã có tính năng" và "đã tối ưu để đạt target 100/day > 1.5".

Đánh giá thực tế:

1. Fundamental + Alt Data: **Đạt một phần tốt**
   - Đã có trong seed (`eps`, `sales`, `book_value`, `ebitda`, `operating_margin`, `sentiment`, `short_interest_ratio`, `analyst_estimates`).
   - Chưa có cơ chế ép tỷ trọng đủ cao cho nhóm orthogonal này trong mọi batch.

2. AST Crossover: **Đạt bản đầu tiên**
   - Có `alpha_ast.py` và crossover trong `evolve.py`.
   - Crossover hiện là pseudo-AST theo ngoặc, chưa phải typed-AST có kiểm tra semantic sâu (kiểu dữ liệu/time-series/group compatibility).

3. Collinearity Filter: **Đã hoạt động**
   - Có lọc token overlap trên 500 mẫu gần nhất với ngưỡng mặc định 0.85.
   - Mạnh ở tốc độ, nhưng độ chính xác còn hạn chế vì chỉ dựa token Jaccard (chưa "hiểu cấu trúc biểu thức").

4. Auto-Tuning Grid Search: **Đã có, cần tinh chỉnh**
   - Có grid lookback và group trong `evolve.py`.
   - Đang thay số đầu tiên trong biểu thức -> có thể thay nhầm tham số không phải lookback quan trọng.

5. Cross-Region Replication: **Đã có**
   - Có replicate qua `EUR/ASI/CHN`.
   - Chưa tối ưu tiêu chí trigger theo liquidity/turnover profile từng region, chưa có rank vùng theo hiệu quả lịch sử.

## 2) Điểm nghẽn đang cản mục tiêu 100/day > 1.5

1. **Ngưỡng submit chưa khớp target**
   - `is_submittable` đang dùng `sharpe > 1.25`, trong khi mục tiêu của bạn là `> 1.5`.

2. **Throughput submit bị chặn cứng**
   - `submit_alphas` đang cắt `max_submit=10` và hiện tại thiên về STAGING log.
   - Nếu muốn 100/day, phải tách rõ lane "simulate lớn" và lane "submit production".

3. **OpenRouter bottleneck vẫn xuất hiện**
   - Dry-run có cảnh báo `Connection pool is full` lặp lại.
   - Nghĩa là phần "vá triệt để quá tải gọi OpenRouter" chưa đạt mức production-hardening.

4. **Ranker chưa dùng đầy đủ Tier-2 penalties trong luồng chính**
   - Trong pipeline chính, rank theo `score_expression` từng alpha; chưa gọi đầy đủ `filter_and_rank(...)` với family crowding + similarity penalty ngay trước simulate.

5. **Mục tiêu 100/day thiếu công thức kiểm soát funnel**
   - Chưa có dashboard chuẩn theo công thức:
   - generated -> valid -> unique -> simulated -> sharpe>1.0 -> sharpe>1.5 -> pass all checks -> submitted.

## 3) Chiến lược nâng cấp để đạt mục tiêu thực chiến

### Giai đoạn A (1-2 ngày): Chuẩn hóa mục tiêu và funnel

1. Chuẩn hóa KPI cứng:
   - `Target_1`: `>= 100` alpha có `Sharpe > 1.5` / ngày.
   - `Target_2`: `>= 25%` trong nhóm `Sharpe > 1.5` đạt all checks.
   - `Target_3`: tỷ lệ duplicate/collinear trước simulate < 35%.

2. Đồng bộ ngưỡng:
   - Submission gate nâng thành `Sharpe > 1.5` (production).
   - Giữ lane nghiên cứu `Sharpe > 1.25` để học DNA riêng, không submit.

3. Bổ sung dashboard funnel theo round/day:
   - Bắt buộc log các tầng lọc và conversion rate từng tầng.

### Giai đoạn B (2-4 ngày): Tăng chất lượng candidate trước simulate

1. Thay pre-rank của pipeline bằng `filter_and_rank(...)` end-to-end:
   - score + family crowding penalty + in-batch similarity penalty.

2. Nâng collinearity từ token-based lên structure-aware:
   - fingerprint theo operator sequence + field classes + lookback bins.
   - giữ token Jaccard như lớp lọc nhanh, thêm lớp semantic distance phía sau.

3. Giới hạn "family crowding":
   - Mỗi family chỉ cho 1-2 candidate vào simulate/round.
   - Bắt buộc diversification theo theme.

### Giai đoạn C (3-5 ngày): Tối ưu evolution và tuning

1. Grid-search có điều kiện:
   - Chỉ grid với alpha có Sharpe trong vùng "gần pass" (vd 0.8-1.5).
   - Tránh grid đại trà gây nổ quota.

2. AST crossover v2:
   - typed node replacement (ts-series node, cross-sectional node, scalar node).
   - reject sớm child vi phạm semantic template.

3. Multi-armed bandit cho template allocation:
   - tự tăng quota cho theme/mutation đang cho pass-rate cao trong 3-7 ngày gần nhất.

### Giai đoạn D (1 tuần): Cross-region thành profit center thật sự

1. Replicate theo score vùng:
   - Mỗi expression chọn 2-3 vùng có historical win-rate tốt nhất, không replicate mù.

2. Theo dõi matrix hiệu quả:
   - theme x region x neutralization x delay.
   - dùng matrix để ưu tiên generate theo vùng thay vì chỉ USA-first.

## 4) Cấu hình vận hành đề xuất để tối đa hiệu suất

1. Chia 3 lane chạy song song:
   - Lane 1 (Discovery): generate nhiều, rank gắt, chỉ simulate.
   - Lane 2 (Refinement): auto-tune/grid cho near-pass.
   - Lane 3 (Production): chỉ submit alpha `>1.5` + all checks + low similarity.

2. Quản trị quota theo giờ:
   - Không dồn toàn bộ sims đầu ngày.
   - Chạy theo "quota buckets" để có feedback learning liên tục.

3. Chống OpenRouter choke:
   - Giảm burst song song ở RAG mutator.
   - Retry có jitter + connection pool riêng + circuit breaker theo lỗi.

4. Tách "learning DB" và "submission DB":
   - tránh nhiễu giữa alpha để học và alpha đạt chuẩn production.

## 5) Kết luận vận hành

Hệ thống hiện tại **đã lên một đẳng cấp tốt hơn rõ rệt** so với random generator ban đầu, và 5 hạng mục lõi đều đã có implementation thực tế. Nhưng để đạt mục tiêu rất cao `100 alpha/day` với chuẩn `>1.5`, cần thêm 1 lớp "engineer hóa sản xuất" gồm:

1. đồng bộ ngưỡng mục tiêu,
2. rank/filter theo diversity thực sự,
3. tuning có điều kiện thay vì quét mù,
4. replicate theo dữ liệu vùng,
5. hardening hạ tầng gọi LLM/API.

Khi 5 điểm này được khóa lại, cỗ máy mới đi từ "mạnh" sang "ổn định + scale được".
