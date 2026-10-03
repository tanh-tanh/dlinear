# ETTh2: khoảng cách SGD với dạng đóng, đoạn cảm biến bị kẹt, MAE, và số 0,605

Ngày chạy: 03/10/2026. Nhiệm vụ: [NHIEM_VU_4.md](NHIEM_VU_4.md). Tiếp theo [BAO_CAO_ETTH2_SGD.md](BAO_CAO_ETTH2_SGD.md).

- Script: [`scripts/sgd_etth2.py`](../scripts/sgd_etth2.py) (A, B), [`scripts/mae_filtered.py`](../scripts/mae_filtered.py) (C), [`scripts/official_ltsf.py`](../scripts/official_ltsf.py) (D), [`scripts/report_etth2_2.py`](../scripts/report_etth2_2.py) (mọi bảng dưới đây).
- Số liệu:
  - A: [`results/sgd_etth2_20seed/`](../results/sgd_etth2_20seed/);
  - B: [`results/sgd_etth2_filtered/`](../results/sgd_etth2_filtered/);
  - C: [`results/mae_filtered/`](../results/mae_filtered/);
  - D: [`results/official_ltsf/`](../results/official_ltsf/), gồm `patch.diff` và log của code gốc;
  - bảng: [`results/etth2_report_2/`](../results/etth2_report_2/).
- Mọi JSON có khối phần cứng và `git_commit` (`46526c3`, `df41e86` hoặc `926efb2`), không file nào mang nhãn `-dirty`.
- Máy: RTX 5060 Ti (IRLS ở C; code gốc ở D chạy trên GPU như mặc định của nó), CPU Intel (SGD của repo), torch 2.14.0+cu130, TF32 tắt trong code của repo.
- Thay đổi code: [NHAT_KY_THAY_DOI.md](NHAT_KY_THAY_DOI.md), mục 9.
- Thời gian chạy: A 21 phút, B 19 phút (CPU, song song); C 33 phút (GPU); D 5 phút (GPU). Không phần nào vượt giới hạn.

Ký hiệu: "dạng đóng" là nghiệm MSE ở λ* (chọn theo val MSE trên lưới 141 λ). Δ = trung bình SGD − dạng đóng; âm là SGD tốt hơn. CI là khoảng tin cậy 95% theo phân phối t (n − 1 bậc tự do).

## 1. Kết luận

**A. Khoảng cách SGD − dạng đóng có thật với DLinear; với Linear thì chưa đủ theo tiêu chí, do vài seed hỏng.**

- **Tiêu chí:** CI 95% của Δ (MSE test) nằm hẳn dưới 0 ở ít nhất 3/4 ô, giữ cả seed hỏng.
  - **DLinear đạt 4/4 ô.** Δ từ −0,010 (H = 96) đến −0,053 (H = 720); SGD tốt hơn λ* ở 15–19/20 seed.
  - **Linear đạt 2/4 ô** (H = 336: −0,036 [−0,047; −0,025]; H = 720: −0,044 [−0,066; −0,021]). Ở H = 96 và 192, mỗi ô có 4 seed hỏng làm CI vắt qua 0.
  - Bỏ các seed hỏng thì Linear đạt 4/4. Nhưng phép bỏ này dùng MSE test (trung vị + 3·MAD trên test), tức **có dùng thông tin test**. Nó chỉ để tham khảo, như nhiệm vụ yêu cầu báo cáo, không phải một kết quả.
- **Khoảng cách có trên cả val:** CI của Δ val nằm dưới 0 ở 7/8 ô Linear/DLinear. Nó không chỉ do dịch chuyển phân phối giữa val và test.
- **NLinear (đối chứng) đi ngược lại:** SGD kém dạng đóng ở cả 4 ô, CI nằm hẳn trên 0.
- **Theo tiêu chí cũ (2 sd), không ô nào đạt.** Tiêu chí cũ hỏi "một lần chạy SGD có chắc chắn tốt hơn không"; câu hỏi ở đây là về trung bình, và sd theo seed không co lại khi thêm seed. Vì vậy tiêu chí đổi sang CI của trung bình (mục 2).
- **20 seed khác hẳn 3 seed ở DLinear H = 720:** trung bình 0,687 ± 0,059, so với 0,736 ± 0,018 ở nhiệm vụ 3. Ba seed đầu (2021–2023) tình cờ đều nằm ở đuôi trên.

**B. Đoạn hằng không giải thích được khoảng cách: ở 3/4 ô có H ≥ 336, SGD cũng hưởng lợi khi lọc và khoảng cách vẫn âm.**

- **Tiêu chí B.4 ở H ∈ {336, 720}** chỉ đạt ở **Linear H = 336**. Ở đó SGD không đổi khi lọc (Δ_SGD = −0,002 [−0,017; 0,013]) trong khi dạng đóng giảm 0,030, và khoảng cách trên train đã lọc còn −0,008 [−0,028; 0,013].
- **Ba ô còn lại không đạt:**
  - **Linear H = 720:** SGD giảm 0,095, đúng bằng dạng đóng (0,094). Khoảng cách trên train đã lọc vẫn là −0,045 [−0,077; −0,013].
  - **DLinear H = 336 và 720:** SGD giảm 0,022 và 0,076. Khoảng cách vẫn âm: −0,024 [−0,035; −0,013] và −0,035 [−0,055; −0,015].
- **Phần dư trên chính các cửa sổ hằng (B.3) cũng không ủng hộ cơ chế "dạng đóng khớp đúng đoạn kẹt".**
  - Trên các cửa sổ có đích hằng, SGD khớp **tốt hơn** dạng đóng (tỷ lệ MSE phần dư SGD/dạng đóng 0,60–0,93).
  - Trên các cửa sổ chỉ có đầu vào hằng, dạng đóng khớp tốt hơn 4–11%. Mức này chỉ nhỉnh hơn chút so với 1–5% trên các cửa sổ bình thường, vốn là điều tất nhiên vì dạng đóng là nghiệm tối ưu trên train.
- **Kết luận:** đoạn hằng làm hại cả hai cách huấn luyện ở H = 720 (cả hai giảm 0,08–0,095 khi lọc), và làm hại SGD ít hơn dạng đóng ở H = 336. Phần lớn khoảng cách SGD − dạng đóng phải đến từ chỗ khác.

**C. Khoảng một nửa lợi của MAE với Linear ở H ≥ 336 đến từ đoạn hằng; nửa còn lại thì không.**

- **Lợi của MAE** (MSE test của MSE λ = 0 trừ của MAE λ = 0) với Linear còn **47% ở H = 336** (0,074 → 0,035) và **60% ở H = 720** (0,151 → 0,090) khi lọc. Ở H = 192 còn 67%; ở H = 96 thì tăng (121%).
- **Chỉ H = 336 co dưới một nửa**, nên không đạt mức "co lại mạnh" ở cả hai horizon. MAE đang sửa cả đoạn kẹt lẫn một vấn đề khác.
- **MAE trên train đủ vẫn tốt hơn MSE ở λ* trên train đã lọc** ở cả 4 H của Linear. Ví dụ H = 720: 0,590 so với 0,646.
- **NLinear:** MAE không có lợi trên cả train đủ lẫn train đã lọc (lợi âm, −0,001 đến −0,019). Phần trăm trong bảng của NLinear vì vậy không có ý nghĩa.
- **Phụ:** trên train đã lọc, IRLS ở λ = 0 cần **0 bước fallback** ở cả 8 ô. Chỉ ô NLinear H = 720 có thông tin: đường λ của nó trên train đủ từng cần 26 713 bước, nhưng số bước riêng ở λ = 0 trên train đủ không được ghi. Kiểm lại ở C.1 của nhiệm vụ 3 đã cho 0 bước trên train đủ ở ETTh2 H = 336 (Linear, NLinear). Vì vậy đây chỉ là dấu hiệu phù hợp với giải thích ở mục 8.1 của `BAO_CAO_TIEU_CHI_DUNG.md`, chưa phải phép kiểm.

**D. Số 0,605 nằm trong độ phân tán theo seed của chính code gốc; seed mặc định 2021 không cho ra nó.**

- **Code gốc, seed 2021, DLinear H = 720:** **0,7328 / 0,6057** (MSE/MAE test). Không nằm trong 0,605 ± 2%, nên không phải điểm dừng.
- **Năm seed 2021–2025 của code gốc** cho 0,575–0,733, trung bình 0,666 ± 0,070. Seed 2025 cho 0,6083, trong vòng 1% của số công bố; seed 2023 cho 0,5746.
- **SGD của repo (20 seed)** cho 0,687 ± 0,059. Chênh với code gốc là +0,021, CI 95% Welch [−0,064; 0,106]: **không phát hiện khác biệt** giữa hai pipeline, dù với 5 seed của code gốc thì khoảng tin cậy còn rộng. Số 0,605 phù hợp với một lần chạy rơi vào phía thấp của phân phối.
- **Linear H = 720:** code gốc 0,690 ± 0,042, repo 0,697 ± 0,048, công bố 0,698.
- **Cách tính metric không đóng góp gì:** chấm lại checkpoint của code gốc bằng pipeline của repo (float64, W_eff) lệch số code gốc in ra tối đa 1,5e-7. Mọi khác biệt đến từ quá trình huấn luyện (seed).
- **Hệ quả cho bảng 2 của bài:** chênh lệch DLinear 0,605 so với Linear 0,698 ở ETTh2 H = 720 nằm trong nhiễu theo seed. Code gốc cho DLinear 0,666 ± 0,070 và Linear 0,690 ± 0,042.

## 2. Phần A: 20 seed

Seed 2021–2040 cho Linear và DLinear, 2021–2030 cho NLinear. Siêu tham số và cách chấm như nhiệm vụ 3. Ba seed 2021–2023 chạy lại: **36/36 file W trùng từng bit** với nhiệm vụ 3 (3 mô hình × 4 H × 3 seed).

Seed "hỏng" là seed dừng ở epoch 1, hoặc có MSE test lớn hơn trung vị + 3·MAD (MAD = trung vị của |x − trung vị|, không nhân 1,4826). Hàng "(bỏ hỏng)" chỉ in khi có seed hỏng. Kết luận chính dùng hàng giữ. Vì tiêu chí "hỏng" dựa trên MSE test, các hàng "(bỏ hỏng)" có dùng thông tin test và chỉ để tham khảo.

| Mô hình | H | n | MSE test: TB ± sd (trung vị) | MSE val: TB ± sd (trung vị) | λ* test / val | Δ test [CI 95%] | Δ val [CI 95%] | seed tốt hơn λ* | seed hỏng | CI dưới 0 | tiêu chí cũ (2 sd) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Linear | 96 | 20 | 0,2968 ± 0,0191 (0,2910) | 0,2175 ± 0,0064 (0,2161) | 0,3009 / 0,2212 | −0,0041 [−0,0130; 0,0049] | −0,0037 [−0,0068; −0,0007] | 15/20 | 2022, 2031, 2036, 2038 | không | không |
| Linear (bỏ hỏng) | 96 | 16 | 0,2891 ± 0,0061 (0,2893) | 0,2153 ± 0,0029 (0,2153) | 0,3009 / 0,2212 | −0,0118 [−0,0150; −0,0085] | −0,0059 [−0,0075; −0,0044] | 15/16 | — | có | không |
| Linear | 192 | 20 | 0,3961 ± 0,0472 (0,3792) | 0,3000 ± 0,0198 (0,2948) | 0,3966 / 0,3028 | −0,0005 [−0,0226; 0,0216] | −0,0028 [−0,0121; 0,0065] | 16/20 | 2021, 2025, 2033, 2036 | không | không |
| Linear (bỏ hỏng) | 192 | 16 | 0,3754 ± 0,0110 (0,3761) | 0,2919 ± 0,0059 (0,2931) | 0,3966 / 0,3028 | −0,0212 [−0,0270; −0,0154] | −0,0110 [−0,0141; −0,0078] | 16/16 | — | có | không |
| Linear | 336 | 20 | 0,4496 ± 0,0241 (0,4445) | 0,3955 ± 0,0099 (0,3943) | 0,4853 / 0,4087 | −0,0357 [−0,0470; −0,0245] | −0,0132 [−0,0179; −0,0086] | 18/20 | 2034 | có | không |
| Linear (bỏ hỏng) | 336 | 19 | 0,4460 ± 0,0184 (0,4434) | 0,3939 ± 0,0071 (0,3943) | 0,4853 / 0,4087 | −0,0393 [−0,0482; −0,0304] | −0,0148 [−0,0182; −0,0114] | 18/19 | — | có | có |
| Linear | 720 | 20 | 0,6967 ± 0,0484 (0,7150) | 0,6591 ± 0,0076 (0,6590) | 0,7404 / 0,6673 | −0,0437 [−0,0663; −0,0211] | −0,0082 [−0,0118; −0,0046] | 16/20 | 2036 | có | không |
| Linear (bỏ hỏng) | 720 | 19 | 0,6933 ± 0,0473 (0,7133) | 0,6579 ± 0,0059 (0,6585) | 0,7404 / 0,6673 | −0,0470 [−0,0698; −0,0242] | −0,0093 [−0,0122; −0,0065] | 16/19 | — | có | không |
| DLinear | 96 | 20 | 0,2913 ± 0,0060 (0,2917) | 0,2167 ± 0,0034 (0,2168) | 0,3008 / 0,2213 | −0,0095 [−0,0123; −0,0067] | −0,0046 [−0,0062; −0,0030] | 18/20 | — | có | không |
| DLinear | 192 | 20 | 0,3807 ± 0,0145 (0,3770) | 0,2979 ± 0,0050 (0,2964) | 0,3966 / 0,3029 | −0,0159 [−0,0227; −0,0091] | −0,0050 [−0,0074; −0,0027] | 17/20 | 2021, 2028 | có | không |
| DLinear (bỏ hỏng) | 192 | 18 | 0,3773 ± 0,0107 (0,3745) | 0,2969 ± 0,0042 (0,2964) | 0,3966 / 0,3029 | −0,0193 [−0,0246; −0,0140] | −0,0060 [−0,0081; −0,0039] | 17/18 | — | có | không |
| DLinear | 336 | 20 | 0,4529 ± 0,0242 (0,4609) | 0,3986 ± 0,0070 (0,3994) | 0,4853 / 0,4087 | −0,0325 [−0,0438; −0,0212] | −0,0101 [−0,0134; −0,0068] | 19/20 | — | có | không |
| DLinear | 720 | 20 | 0,6869 ± 0,0592 (0,7022) | 0,6594 ± 0,0087 (0,6611) | 0,7403 / 0,6672 | −0,0534 [−0,0811; −0,0257] | −0,0078 [−0,0119; −0,0038] | 15/20 | — | có | không |
| NLinear | 96 | 10 | 0,2770 ± 0,0020 (0,2765) | 0,2068 ± 0,0010 (0,2069) | 0,2717 / 0,2048 | 0,0053 [0,0039; 0,0067] | 0,0020 [0,0013; 0,0028] | 0/10 | 2026 | không | không |
| NLinear (bỏ hỏng) | 96 | 9 | 0,2766 ± 0,0016 (0,2761) | 0,2068 ± 0,0011 (0,2066) | 0,2717 / 0,2048 | 0,0049 [0,0037; 0,0061] | 0,0020 [0,0012; 0,0029] | 0/9 | — | không | không |
| NLinear | 192 | 10 | 0,3444 ± 0,0041 (0,3429) | 0,2722 ± 0,0017 (0,2723) | 0,3336 / 0,2713 | 0,0108 [0,0079; 0,0137] | 0,0009 [−0,0003; 0,0021] | 0/10 | 2022, 2028 | không | không |
| NLinear (bỏ hỏng) | 192 | 8 | 0,3427 ± 0,0021 (0,3428) | 0,2723 ± 0,0019 (0,2724) | 0,3336 / 0,2713 | 0,0091 [0,0073; 0,0109] | 0,0010 [−0,0006; 0,0026] | 0/8 | — | không | không |
| NLinear | 336 | 10 | 0,3804 ± 0,0139 (0,3781) | 0,3609 ± 0,0024 (0,3604) | 0,3581 / 0,3607 | 0,0223 [0,0123; 0,0322] | 0,0001 [−0,0016; 0,0018] | 0/10 | 2023 | không | không |
| NLinear (bỏ hỏng) | 336 | 9 | 0,3767 ± 0,0078 (0,3779) | 0,3608 ± 0,0025 (0,3598) | 0,3581 / 0,3607 | 0,0186 [0,0125; 0,0246] | 0,0001 [−0,0018; 0,0020] | 0/9 | — | không | không |
| NLinear | 720 | 10 | 0,4058 ± 0,0122 (0,4010) | 0,6049 ± 0,0076 (0,6024) | 0,3924 / 0,6035 | 0,0134 [0,0047; 0,0221] | 0,0014 [−0,0041; 0,0068] | 0/10 | 2023, 2024 | không | không |
| NLinear (bỏ hỏng) | 720 | 8 | 0,4004 ± 0,0038 (0,4005) | 0,6020 ± 0,0051 (0,6005) | 0,3924 / 0,6035 | 0,0080 [0,0049; 0,0112] | −0,0015 [−0,0058; 0,0028] | 0/8 | — | không | không |

Linear_giữ: CI dưới 0 ở 2/4 ô, 2 sd ở 0/4; Linear_bỏ: CI dưới 0 ở 4/4 ô, 2 sd ở 1/4; DLinear_giữ: CI dưới 0 ở 4/4 ô, 2 sd ở 0/4; DLinear_bỏ: CI dưới 0 ở 4/4 ô, 2 sd ở 0/4; NLinear_giữ: CI dưới 0 ở 0/4 ô, 2 sd ở 0/4; NLinear_bỏ: CI dưới 0 ở 0/4 ô, 2 sd ở 0/4

SGD 20 seed so với công bố (trung bình, giữ seed hỏng): Linear H=96 +3,1 %; Linear H=192 +5,1 %; Linear H=336 -0,5 %; Linear H=720 -0,2 %; DLinear H=96 +0,8 %; DLinear H=192 -0,6 %; DLinear H=336 +1,1 %; DLinear H=720 +13,5 %; NLinear H=96 -0,0 %; NLinear H=192 +0,1 %; NLinear H=336 +6,6 %; NLinear H=720 +3,0 %

**Vì sao đổi tiêu chí.** Tiêu chí cũ (Δ < −2 sd) đòi khoảng cách lớn hơn hai lần độ phân tán của **từng lần chạy**. Nó không chặt hơn khi thêm seed, vì sd theo seed là đặc tính của SGD, không phải sai số đo. Câu hỏi "trung bình SGD có tốt hơn dạng đóng không" cần sai số của **trung bình**, tức sd/√n. Với 20 seed, nửa độ rộng CI nhỏ hơn 2 sd khoảng 4,3 lần (2√20 / t₀,₉₇₅;₁₉).

**Seed hỏng.**
- Linear: 4 seed ở H = 96 và ở H = 192, 1 seed ở H = 336 và ở H = 720. DLinear: 2 seed ở H = 192.
- Với Linear, 5/10 seed hỏng có epoch tốt nhất là epoch 1 (dừng ở epoch 4), số còn lại dừng sớm ở epoch 6–8 với epoch tốt nhất 3–5. Cả hai trường hợp đều do val tăng ba epoch liền.
- Hai seed hỏng của DLinear (H = 192) chạy đủ 10 epoch. Tiêu chí MAD bắt chúng vì test cao hơn hẳn phần còn lại (0,411 so với trung vị 0,377).
- Bỏ chúng thì sd của Linear ở H = 96, 192 giảm 3–4 lần.

## 3. Phần B: SGD trên train đã lọc

**Cách lọc cho SGD.** Cặp (cửa sổ, kênh) hằng bị bỏ khỏi hàm mất mát (`MaskedETTDataset`, `masked_mse`). Batch nhiều kênh, thứ tự xáo, khởi tạo, val và test giữ nguyên.
- Điều này tương đương bỏ đúng các hàng đó khỏi ma trận hồi quy của dạng đóng (hàm lọc `constant_mask` của `constant_windows.py`).
- Cùng seed thì cùng khởi tạo và cùng thứ tự batch, nên Δ_SGD so cặp theo seed.
- Không có mask thì code chạy y như cũ (36/36 trùng từng bit, mục 2).

**Dạng đóng trên train đã lọc** lấy lại từ C.3 của nhiệm vụ 3 (`results/constant_windows/`). Kiểm lại: 12/12 (mô hình, H) có đủ 142 λ, λ* khớp `summary.json`.

### MSE test

| Mô hình | H | dạng đóng: đủ → lọc (Δ_CF) | SGD: đủ → lọc | Δ_SGD [CI] | Δ_SGD − Δ_CF [CI] | SGD − CF, đủ [CI] | SGD − CF, lọc [CI] |
|---|---|---|---|---|---|---|---|
| Linear | 96 | 0,3009 → 0,3146 (0,0137) | 0,2968 → 0,3160 | 0,0191 [0,0072; 0,0310] | 0,0054 [−0,0065; 0,0173] | −0,0041 [−0,0130; 0,0049] | 0,0013 [−0,0128; 0,0155] |
| Linear | 192 | 0,3966 → 0,3954 (−0,0012) | 0,3961 → 0,3981 | 0,0020 [−0,0256; 0,0296] | 0,0032 [−0,0244; 0,0308] | −0,0005 [−0,0226; 0,0216] | 0,0027 [−0,0190; 0,0243] |
| Linear | 336 | 0,4853 → 0,4553 (−0,0300) | 0,4496 → 0,4478 | −0,0018 [−0,0166; 0,0131] | 0,0283 [0,0134; 0,0431] | −0,0357 [−0,0470; −0,0245] | −0,0075 [−0,0276; 0,0127] |
| Linear | 720 | 0,7404 → 0,6462 (−0,0942) | 0,6967 → 0,6015 | −0,0952 [−0,1267; −0,0637] | −0,0010 [−0,0325; 0,0305] | −0,0437 [−0,0663; −0,0211] | −0,0447 [−0,0765; −0,0129] |
| DLinear | 96 | 0,3008 → 0,3145 (0,0138) | 0,2913 → 0,2989 | 0,0076 [0,0032; 0,0120] | −0,0062 [−0,0106; −0,0018] | −0,0095 [−0,0123; −0,0067] | −0,0157 [−0,0199; −0,0114] |
| DLinear | 192 | 0,3966 → 0,3954 (−0,0012) | 0,3807 → 0,3814 | 0,0007 [−0,0080; 0,0094] | 0,0019 [−0,0068; 0,0106] | −0,0159 [−0,0227; −0,0091] | −0,0140 [−0,0239; −0,0041] |
| DLinear | 336 | 0,4853 → 0,4553 (−0,0300) | 0,4529 → 0,4313 | −0,0215 [−0,0342; −0,0089] | 0,0085 [−0,0042; 0,0212] | −0,0325 [−0,0438; −0,0212] | −0,0240 [−0,0349; −0,0130] |
| DLinear | 720 | 0,7403 → 0,6461 (−0,0942) | 0,6869 → 0,6109 | −0,0760 [−0,1106; −0,0415] | 0,0182 [−0,0164; 0,0527] | −0,0534 [−0,0811; −0,0257] | −0,0353 [−0,0551; −0,0154] |

### MSE val

| Mô hình | H | dạng đóng: đủ → lọc (Δ_CF) | SGD: đủ → lọc | Δ_SGD [CI] | Δ_SGD − Δ_CF [CI] | SGD − CF, đủ [CI] | SGD − CF, lọc [CI] |
|---|---|---|---|---|---|---|---|
| Linear | 96 | 0,2212 → 0,2322 (0,0109) | 0,2175 → 0,2306 | 0,0132 [0,0075; 0,0188] | 0,0022 [−0,0035; 0,0079] | −0,0037 [−0,0068; −0,0007] | −0,0015 [−0,0078; 0,0048] |
| Linear | 192 | 0,3028 → 0,3084 (0,0055) | 0,3000 → 0,3043 | 0,0043 [−0,0081; 0,0166] | −0,0013 [−0,0136; 0,0111] | −0,0028 [−0,0121; 0,0065] | −0,0041 [−0,0128; 0,0047] |
| Linear | 336 | 0,4087 → 0,4035 (−0,0051) | 0,3955 → 0,3968 | 0,0014 [−0,0050; 0,0077] | 0,0065 [0,0001; 0,0129] | −0,0132 [−0,0179; −0,0086] | −0,0067 [−0,0150; 0,0016] |
| Linear | 720 | 0,6673 → 0,6535 (−0,0138) | 0,6591 → 0,6451 | −0,0140 [−0,0216; −0,0064] | −0,0002 [−0,0078; 0,0074] | −0,0082 [−0,0118; −0,0046] | −0,0084 [−0,0137; −0,0031] |
| DLinear | 96 | 0,2213 → 0,2322 (0,0109) | 0,2167 → 0,2239 | 0,0072 [0,0049; 0,0094] | −0,0037 [−0,0059; −0,0015] | −0,0046 [−0,0062; −0,0030] | −0,0083 [−0,0107; −0,0060] |
| DLinear | 192 | 0,3029 → 0,3084 (0,0055) | 0,2979 → 0,3026 | 0,0047 [0,0023; 0,0072] | −0,0008 [−0,0032; 0,0017] | −0,0050 [−0,0074; −0,0027] | −0,0058 [−0,0087; −0,0028] |
| DLinear | 336 | 0,4087 → 0,4035 (−0,0052) | 0,3986 → 0,3947 | −0,0039 [−0,0093; 0,0015] | 0,0013 [−0,0041; 0,0066] | −0,0101 [−0,0134; −0,0068] | −0,0089 [−0,0132; −0,0045] |
| DLinear | 720 | 0,6672 → 0,6534 (−0,0138) | 0,6594 → 0,6474 | −0,0120 [−0,0163; −0,0077] | 0,0018 [−0,0025; 0,0061] | −0,0078 [−0,0119; −0,0038] | −0,0060 [−0,0093; −0,0027] |

Tiêu chí B.4 (H ∈ {336, 720}, MSE test):

| Mô hình | H | Δ_CF < 0 | CI của Δ_SGD − Δ_CF nằm trên 0 | khoảng cách trên train đã lọc không còn âm | cơ chế |
|---|---|---|---|---|---|
| Linear | 336 | có | có | có | **ủng hộ** |
| Linear | 720 | có | không | không | không |
| DLinear | 336 | có | không | không | không |
| DLinear | 720 | có | không | không | không |

### 3.1. B.3: phần dư trên các cửa sổ hằng của train

MSE phần dư trên các cửa sổ hằng của train đủ, tách theo nhóm. "Không hằng" để so. Dạng đóng ở λ* (train đủ), SGD là 20 seed của Phần A (train đủ).

| H | Mô hình | nhóm | số cửa sổ | dạng đóng λ* | SGD (TB ± sd theo seed) | SGD / dạng đóng |
|---|---|---|---|---|---|---|
| 96 | Linear | x hằng, y không | 477 | 2,6796 | 2,9220 ± 0,1855 | 1,09 |
| 96 | Linear | chỉ y hằng | 2447 | 0,2041 | 0,1803 ± 0,0287 | 0,88 |
| 96 | Linear | cả hai | 688 | 0,0973 | 0,0588 ± 0,0494 | 0,60 |
| 96 | Linear | không hằng | 53851 | 0,3717 | 0,3840 ± 0,0109 | 1,03 |
| 96 | DLinear | x hằng, y không | 477 | 2,6796 | 2,7956 ± 0,0707 | 1,04 |
| 96 | DLinear | chỉ y hằng | 2447 | 0,2038 | 0,1842 ± 0,0167 | 0,90 |
| 96 | DLinear | cả hai | 688 | 0,0973 | 0,0607 ± 0,0198 | 0,62 |
| 96 | DLinear | không hằng | 53851 | 0,3716 | 0,3757 ± 0,0039 | 1,01 |
| 192 | Linear | x hằng, y không | 667 | 3,0072 | 3,3413 ± 0,2951 | 1,11 |
| 192 | Linear | chỉ y hằng | 1757 | 0,3534 | 0,3256 ± 0,0466 | 0,92 |
| 192 | Linear | cả hai | 498 | 0,2062 | 0,1558 ± 0,0953 | 0,76 |
| 192 | Linear | không hằng | 53869 | 0,4525 | 0,4741 ± 0,0217 | 1,05 |
| 192 | DLinear | x hằng, y không | 667 | 3,0074 | 3,1858 ± 0,1771 | 1,06 |
| 192 | DLinear | chỉ y hằng | 1757 | 0,3527 | 0,3267 ± 0,0242 | 0,93 |
| 192 | DLinear | cả hai | 498 | 0,2061 | 0,1471 ± 0,0470 | 0,71 |
| 192 | DLinear | không hằng | 53869 | 0,4525 | 0,4613 ± 0,0119 | 1,02 |
| 336 | Linear | x hằng, y không | 811 | 2,9836 | 3,2988 ± 0,1751 | 1,11 |
| 336 | Linear | chỉ y hằng | 811 | 0,5199 | 0,4316 ± 0,0492 | 0,83 |
| 336 | Linear | cả hai | 354 | 0,4026 | 0,2820 ± 0,0564 | 0,70 |
| 336 | Linear | không hằng | 53807 | 0,5267 | 0,5402 ± 0,0143 | 1,03 |
| 336 | DLinear | x hằng, y không | 811 | 2,9837 | 3,2101 ± 0,1814 | 1,08 |
| 336 | DLinear | chỉ y hằng | 811 | 0,5195 | 0,4546 ± 0,0527 | 0,87 |
| 336 | DLinear | cả hai | 354 | 0,4025 | 0,3003 ± 0,0724 | 0,75 |
| 336 | DLinear | không hằng | 53807 | 0,5267 | 0,5371 ± 0,0111 | 1,02 |
| 720 | Linear | x hằng, y không | 1033 | 2,2636 | 2,4772 ± 0,2039 | 1,09 |
| 720 | Linear | chỉ y hằng | 306 | 1,6463 | 1,4371 ± 0,2237 | 0,87 |
| 720 | Linear | không hằng | 51756 | 0,6562 | 0,6783 ± 0,0279 | 1,03 |
| 720 | DLinear | x hằng, y không | 1033 | 2,2637 | 2,4294 ± 0,2153 | 1,07 |
| 720 | DLinear | chỉ y hằng | 306 | 1,6454 | 1,5285 ± 0,1318 | 0,93 |
| 720 | DLinear | không hằng | 51756 | 0,6562 | 0,6687 ± 0,0198 | 1,02 |

- **Nhóm có đích hằng** ("chỉ y hằng", "cả hai"): SGD có phần dư **nhỏ hơn** dạng đóng ở cả 14 dòng (tỷ lệ 0,60–0,93; H = 720 không có cửa sổ thuộc nhóm "cả hai").
- **Nhóm chỉ có đầu vào hằng:** dạng đóng tốt hơn 4–11%. Trên các cửa sổ bình thường, dạng đóng cũng tốt hơn 1–5%. Đó là điều tất nhiên: dạng đóng tối thiểu MSE trên toàn train.
- Vậy dạng đóng không "khớp đúng đoạn kẹt" hơn SGD. Cơ chế này không được ủng hộ.

## 4. Phần C: MAE trên train đã lọc

MAE, λ = 0, IRLS chế độ pha, `tol = 3e-11`, patience 1, chunk 16. Khởi tạo từ nghiệm MSE λ = 0 trên cùng train đã lọc. Dòng "đủ" lấy từ `lambda_path.csv`. "Lợi của MAE" = MSE test của MSE λ = 0 − MSE test của MAE λ = 0.

| Mô hình | H | train | MSE test: MSE λ = 0 | MSE test: MAE λ = 0 | lợi của MAE | MAE test: MSE / MAE λ = 0 | vòng | fallback |
|---|---|---|---|---|---|---|---|---|
| Linear | 96 | đủ | 0,3019 | 0,2789 | 0,0230 | 0,3655 / 0,3342 | 456 | không ghi |
| | | đã lọc | 0,3151 | 0,2874 | 0,0278 (121 % của lợi cũ) | 0,3776 / 0,3474 | 462 | 0 |
| Linear | 192 | đủ | 0,3972 | 0,3541 | 0,0431 | 0,4276 / 0,3861 | 452 | không ghi |
| | | đã lọc | 0,3958 | 0,3668 | 0,0290 (67 % của lợi cũ) | 0,4289 / 0,4002 | 473 | 0 |
| Linear | 336 | đủ | 0,4858 | 0,4116 | 0,0742 | 0,4837 / 0,4301 | 453 | không ghi |
| | | đã lọc | 0,4557 | 0,4209 | 0,0348 (47 % của lợi cũ) | 0,4687 / 0,4389 | 453 | 0 |
| Linear | 720 | đủ | 0,7407 | 0,5898 | 0,1509 | 0,6116 / 0,5337 | 474 | không ghi |
| | | đã lọc | 0,6463 | 0,5564 | 0,0900 (60 % của lợi cũ) | 0,5738 / 0,5225 | 476 | 0 |
| NLinear | 96 | đủ | 0,2726 | 0,2781 | −0,0055 | 0,3352 / 0,3317 | 444 | không ghi |
| | | đã lọc | 0,2724 | 0,2755 | −0,0031 (55 % của lợi cũ) | 0,3362 / 0,3318 | 487 | 0 |
| NLinear | 192 | đủ | 0,3343 | 0,3464 | −0,0121 | 0,3762 / 0,3765 | 455 | không ghi |
| | | đã lọc | 0,3324 | 0,3424 | −0,0100 (83 % của lợi cũ) | 0,3760 / 0,3755 | 473 | 0 |
| NLinear | 336 | đủ | 0,3588 | 0,3779 | −0,0191 | 0,4004 / 0,4039 | 445 | không ghi |
| | | đã lọc | 0,3562 | 0,3731 | −0,0168 (88 % của lợi cũ) | 0,3992 / 0,4017 | 442 | 0 |
| NLinear | 720 | đủ | 0,3930 | 0,3945 | −0,0015 | 0,4346 / 0,4255 | 439 | không ghi |
| | | đã lọc | 0,3915 | 0,3921 | −0,0006 (41 % của lợi cũ) | 0,4337 / 0,4248 | 459 | 0 |

So MAE λ = 0 trên train đủ với MSE ở λ* trên train đã lọc (MSE test):

| Mô hình | H | MAE, train đủ | MSE λ*, train đã lọc |
|---|---|---|---|
| Linear | 96 | 0,2789 | 0,3146 |
| Linear | 192 | 0,3541 | 0,3954 |
| Linear | 336 | 0,4116 | 0,4553 |
| Linear | 720 | 0,5898 | 0,6462 |
| NLinear | 96 | 0,2781 | 0,2715 |
| NLinear | 192 | 0,3464 | 0,3317 |
| NLinear | 336 | 0,3779 | 0,3556 |
| NLinear | 720 | 0,3945 | 0,3908 |

- **Linear:** MAE trên train đủ tốt hơn MSE trên train đã lọc ở cả 4 H, cách biệt 0,036–0,056. Lọc đoạn kẹt không thay được MAE.
- **NLinear:** ngược lại, MSE trên train đã lọc tốt hơn MAE trên train đủ ở cả 4 H. Hợp với việc MAE không có lợi cho NLinear.
- **NLinear H = 720 chạy đủ** (9 phút), vì ước lượng dưới 30 phút.

## 5. Phần D: code gốc của LTSF-Linear

**Chuẩn bị** (`official_ltsf.py --prepare`): chép `third_party/LTSF-Linear` (commit `0c11366`) sang `scratch/ltsf_official/` (trong `.gitignore`). Bản chép áp [`patch.diff`](../results/official_ltsf/patch.diff), gồm hai thay đổi:
- `run_longExp.py`: thêm `--seed` (mặc định 2021) và đặt seed sau `parse_args`. Không có gì trước đó dùng số ngẫu nhiên, nên seed 2021 cho đúng hành vi gốc.
- `utils/tools.py`: `np.Inf` → `np.inf`, vì numpy 2.3 đã bỏ `np.Inf` (`EarlyStopping` lỗi ngay khi khởi tạo nếu không sửa).

Không sửa gì khác. Lệnh chạy dùng đúng tham số của `etth2.sh` (lr 0,05, batch 32), thêm:
- `--model`, `--seed`;
- `--num_workers 0`: trên Windows, worker của DataLoader được spawn và chạy lại toàn bộ `run_longExp.py`, vì file không có `if __name__ == "__main__"`. Thứ tự xáo do tiến trình chính lấy, nên không đổi.

Dữ liệu chép vào `scratch/ltsf_official/dataset/`. Code gốc chạy trên GPU, dùng mặc định TF32 của torch (matmul tắt; cuDNN bật, nhưng mô hình không có tích chập).

| Mô hình | H | seed | code gốc: MSE / MAE test | chấm lại bằng repo: MSE / MAE test | chênh MSE | epoch tốt nhất |
|---|---|---|---|---|---|---|
| DLinear | 336 | 2021 | 0,4629 / 0,4726 | 0,4629 / 0,4726 | -1,2e-07 | 8/10 |
| DLinear | 336 | 2022 | 0,4376 / 0,4523 | 0,4376 / 0,4523 | -8,4e-08 | 8/10 |
| DLinear | 336 | 2023 | 0,4852 / 0,4828 | 0,4852 / 0,4828 | -8,6e-08 | 10/10 |
| DLinear | 336 | 2024 | 0,4504 / 0,4623 | 0,4504 / 0,4623 | -4,3e-08 | 6/9 |
| DLinear | 336 | 2025 | 0,4883 / 0,4856 | 0,4883 / 0,4856 | -1,4e-07 | 10/10 |
| Linear | 336 | 2021 | 0,4259 / 0,4487 | 0,4259 / 0,4487 | -1,3e-08 | 8/10 |
| Linear | 336 | 2022 | 0,5586 / 0,5180 | 0,5586 / 0,5180 | +1,9e-08 | 1/4 |
| Linear | 336 | 2023 | 0,4742 / 0,4718 | 0,4742 / 0,4718 | +8,3e-09 | 7/10 |
| Linear | 336 | 2024 | 0,4630 / 0,4719 | 0,4630 / 0,4719 | -4,2e-08 | 3/6 |
| Linear | 336 | 2025 | 0,4942 / 0,4885 | 0,4942 / 0,4885 | -2,2e-09 | 5/8 |
| DLinear | 720 | 2021 | 0,7328 / 0,6057 | 0,7328 / 0,6057 | -1,4e-07 | 9/10 |
| DLinear | 720 | 2022 | 0,7162 / 0,5986 | 0,7162 / 0,5986 | -1,5e-07 | 8/10 |
| DLinear | 720 | 2023 | 0,5746 / 0,5398 | 0,5746 / 0,5398 | -2,9e-08 | 5/8 |
| DLinear | 720 | 2024 | 0,6958 / 0,5946 | 0,6958 / 0,5946 | -1,1e-07 | 10/10 |
| DLinear | 720 | 2025 | 0,6083 / 0,5550 | 0,6083 / 0,5550 | -5,8e-08 | 4/7 |
| Linear | 720 | 2021 | 0,6995 / 0,5950 | 0,6995 / 0,5950 | +1,9e-08 | 10/10 |
| Linear | 720 | 2022 | 0,7106 / 0,5978 | 0,7106 / 0,5978 | +3,9e-09 | 4/7 |
| Linear | 720 | 2023 | 0,6176 / 0,5523 | 0,6176 / 0,5523 | +8,5e-09 | 3/6 |
| Linear | 720 | 2024 | 0,6992 / 0,5887 | 0,6992 / 0,5887 | +3,4e-08 | 6/9 |
| Linear | 720 | 2025 | 0,7241 / 0,6008 | 0,7241 / 0,6008 | +2,3e-08 | 7/10 |

| Mô hình | H | code gốc: TB ± sd (n) | SGD repo 20 seed: TB ± sd | repo − code gốc [CI 95% Welch] | công bố |
|---|---|---|---|---|---|
| DLinear | 336 | 0,4649 ± 0,0219 (5) | 0,4529 ± 0,0242 | −0,0120 [−0,0387; 0,0147] | 0,4480 |
| Linear | 336 | 0,4832 ± 0,0489 (5) | 0,4496 ± 0,0241 | −0,0336 [−0,0935; 0,0264] | 0,4520 |
| DLinear | 720 | 0,6655 ± 0,0699 (5) | 0,6869 ± 0,0592 | 0,0214 [−0,0635; 0,1062] | 0,6050 |
| Linear | 720 | 0,6902 ± 0,0418 (5) | 0,6967 ± 0,0484 | 0,0065 [−0,0447; 0,0576] | 0,6980 |

- **Chấm chéo (D.3):** số code gốc in ra và số chấm lại bằng pipeline của repo lệch tối đa 1,5e-7 (float32 so với float64). Hai pipeline dùng cùng 2161 cửa sổ test.
- **Theo D.4:** code gốc cho mức như repo, không phải 0,605. Nhưng "mức như repo" ở đây là một phân phối rộng (0,575–0,733), và 0,605 nằm trong đó. Với commit `0c11366`, số công bố không tái lập được từ seed mặc định, nhưng tái lập được về mặt phân phối.
- **Linear H = 336, seed 2022** dừng ở epoch 1 (val tăng ba epoch liền), giống các seed hỏng của repo ở Phần A. Hiện tượng này có ở cả code gốc.

## 6. Đoạn nháp cho báo cáo cuối: "ETTh2 và các đoạn cảm biến bị kẹt"

> ETTh2 chứa các đoạn mà ba kênh tải (MUFL, LUFL, LULL) giữ nguyên một giá trị trong thời gian dài, có đoạn ít nhất hai tuần, nhiều khả năng do cảm biến bị kẹt. Sau khi chia cửa sổ, 2,5–6,3% số cửa sổ train có đầu vào hoặc đích hằng; val và test hầu như không có, trừ ở H ≤ 192. Mọi kết quả chính của báo cáo dùng dữ liệu gốc. Phần này là phân tích độ nhạy: bỏ các cửa sổ hằng khỏi train, giữ nguyên val và test. Ở H = 720, việc này làm MSE test của Linear và DLinear giảm 0,08–0,095, với cả nghiệm dạng đóng (0,740 → 0,646) lẫn SGD (Linear 0,697 → 0,602, trung bình 20 seed). Ở H = 96 thì ngược lại, MSE test tăng 0,01–0,02. Ở 3/4 ô có H ≥ 336, SGD cũng cải thiện khi lọc và vẫn tốt hơn nghiệm dạng đóng, nên đoạn hằng không giải thích được phần lớn việc SGD tốt hơn nghiệm dạng đóng trên ETTh2. Đoạn hằng giải thích được khoảng một nửa lợi thế của hàm mục tiêu MAE với Linear ở H ≥ 336 (47–60% lợi thế còn lại sau khi lọc). NLinear gần như không bị ảnh hưởng bởi việc lọc (MSE test đổi không quá 0,003).

## 7. Tái lập

```bash
python -m unittest discover -s tests -t .                                        # 49 test, tất cả đạt
python scripts/sgd_etth2.py --step run --out sgd_etth2_20seed --model Linear DLinear --seed $(seq 2021 2040)
python scripts/sgd_etth2.py --step run --out sgd_etth2_20seed --model NLinear --seed $(seq 2021 2030)
python scripts/sgd_etth2.py --step run --out sgd_etth2_filtered --filtered --model Linear DLinear --seed $(seq 2021 2040)
python scripts/mae_filtered.py --model Linear NLinear
python scripts/official_ltsf.py --prepare
python scripts/official_ltsf.py --H 720 336                                       # DLinear, Linear; seed 2021–2025
python scripts/report_etth2_2.py                                                  # results/etth2_report_2/tables.md
```

Ghi chú về các lần chạy:

- **D chạy hai lần.** Lần đầu, `prepare` lỗi khi xóa một thư mục `.git` tạm do tôi tạo bằng tay. Tôi lại xóa nhầm bản chép trong lúc nó đang chạy, nên seed 2023 lỗi.
  - Phát hiện thêm: `git apply` chạy bên trong `scratch/` (thư mục con của repo) **lặng lẽ bỏ qua patch**. Đã sửa: áp từ gốc repo với `--directory`, rồi kiểm patch đã vào (commit `926efb2`).
  - Kết quả lần đầu đã xóa; mọi số ở mục 5 là của lần thứ hai.
  - Seed 2021 và 2022 cho cùng số ở cả hai lần (0,7328 và 0,7162).
- **`results/sgd_etth2_20seed/*.npy` và `results/sgd_etth2_filtered/*.npy` không commit** (giống `results/weights/`); tái lập trùng từng bit bằng các lệnh trên. Checkpoint của code gốc nằm ở `scratch/official_ckpt/` (không commit).
