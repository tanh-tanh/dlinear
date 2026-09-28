| | λ = 0 | λ > 0 |
|---|---|---|
| **DLinear so với Linear** | **Trùng**, cả 4 kernel. rank Z = 336/672; max\|Δŷ\| test ≤ 2e-13; ‖ΔW‖/‖W‖ ≤ 4e-13; ‖∇‖ qua module ≈ 6e-15 × lúc khởi tạo | **Tách**, mức tách phụ thuộc k: ‖ΔW‖/‖W‖ tăng theo k ở 32/33 giá trị λ > 0. RMS Δŷ tăng theo λ tới đỉnh ở λ ≈ 5.6e+04–1e+06 rồi giảm khi mọi trọng số co về 0. Tại λ = 10⁴ (k = 5/15/25/49): ‖ΔW‖/‖W‖ = 0.12/0.19/0.28/0.37; RMS Δŷ = 0.017/0.018/0.022/0.029; sau khi khớp λ' vẫn còn 0.015/0.018/0.021/0.027 |
| **NLinear so với Linear** | **Tách.** ‖ΔW‖_F = ‖r‖·‖u‖/s = 0.0694; RMS Δŷ test = 0.0514; SSE train tăng đúng ‖r‖²/s = 17002.8 | **Tách**, tăng đơn điệu theo λ. RMS Δŷ = 0.069 (λ = 10⁴) → 0.973 (λ = 10⁸); nhỏ nhất trên toàn lưới sau khi khớp λ' là 0.0514 |
