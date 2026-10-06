## A.2. `gd_zero` so với đường ridge

- λ_max của Hessian 2G/(nH): power iteration 0,533617 (8 vòng), eigvalsh 0,533617; lr của GD = 1/λ_max = 1,8740.
- Ridge λ* = 1000: val 0,6673, test 0,7404 (tính lại bằng ridge_path: 0,667265 / 0,740365).
- `gd_zero`, bước tốt nhất 2920: val 0,6670, test 0,7401; Δ so với λ*: val −0,0002, test −0,0003.
- ‖W_gd − W_ridge(λ)‖ / ‖W_ridge(λ)‖ (chỉ phần W, không gồm bias) nhỏ nhất trên lưới 141 giá trị: 0,0948 tại λ = 1995; tại λ*: 0,1624.

## A.3. Bảng

Δ = cách − ridge λ*. CI 95% theo t với n − 1 bậc tự do. `gd_zero` tất định (n = 1): Δ là một số, quy tắc ba trường hợp không áp được, chỉ ghi dấu. lr của Adam toàn batch: 0,05 (seed 2021, val tốt nhất: lr 0,05 → 0,666916, lr 0,005 → 0,667051, lr 0,0005 → 0,667358).

| Cách | n | MSE val (TB ± sd) | MSE test (TB ± sd) | Δ val so với λ* [CI] | Δ test so với λ* [CI] | bước tốt nhất (TB; min–max) | dừng ở bước cuối | nhãn |
|---|---|---|---|---|---|---|---|---|
| ridge λ* (λ = 1000) | — | 0,6673 | 0,7404 | — | — | — | — | — |
| `gd_zero` | 1 | 0,6670 | 0,7401 | −0,0002 (dấu âm) | −0,0003 (dấu âm) | 2920; 2920–2920 | 0/1 | — (n = 1, không có CI) |
| `gd_init` | 10 | 0,6675 ± 0,0000 | 0,7406 ± 0,0000 | 0,0003 [0,0003; 0,0003] (thua) | 0,0003 [0,0003; 0,0003] (thua) | 18276; 18020–18640 | 0/10 | dạng đóng tốt hơn thật |
| `adam_full` | 10 | 0,6661 ± 0,0018 | 0,6853 ± 0,0473 | −0,0011 [−0,0024; 0,0002] (hòa) | −0,0551 [−0,0889; −0,0212] (thắng) | 405; 80–915 | 0/10 | khác biệt do val khác test |
| Adam mini-batch (20 seed, đã có) | 20 | 0,6591 ± 0,0076 | 0,6967 ± 0,0484 | −0,0082 [−0,0118; −0,0046] (thắng) | −0,0437 [−0,0663; −0,0211] (thắng) | 1375; 237–2370 | — | SGD tốt hơn thật |

Bước tốt nhất của Adam mini-batch = epoch tốt nhất × số batch mỗi epoch (bước ở cuối epoch đó).

