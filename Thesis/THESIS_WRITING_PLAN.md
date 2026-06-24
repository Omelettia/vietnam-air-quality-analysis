# Kế hoạch viết luận văn — viết gì, vì sao, theo thứ tự nào

Luận văn thạc sĩ, SOICT/HUST.
Tên đề tài chính thức (giữ nguyên theo phiếu đã đăng ký):
**"Ước tính nồng độ bụi mịn PM2.5 mặt đất từ dữ liệu vệ tinh và khí tượng sử dụng học máy".**

Cấu hình dứt điểm: `unified_thesis_v4`, QC mask mạnh hơn, giữ cả 40 trạm.

> **Chiến lược thứ tự viết.** Chương 1–3 (bài toán + lý thuyết + phương pháp đề xuất/xử lý
> dữ liệu) **không** phụ thuộc kết quả mô hình và có thể viết **ngay**. Chương 4–5 phụ thuộc
> kết quả; phần *ổn định* (tương quan AOD–PM2.5, baseline thất bại, trần nội-trạm) cũng có thể
> nháp trước, còn các *con số tiêu đề LOSO / MoE / diverse-kNN* để placeholder đến khi rerun v4
> xong (dự kiến chỉ xê dịch nhẹ).

Ký hiệu trạng thái kết quả: **[LOCKED]** ổn định / không phụ thuộc cấu hình · **[v4-PENDING]** làm mới sau khi chạy.

---

## NGUYÊN TẮC XUYÊN SUỐT (đọc trước khi viết bất cứ chương nào)

Ba ranh giới phải giữ ở **mọi** chương khi viết:

1. **Con số định lượng** (0.80 nội suy thời gian / 0.20 LOSO / 0.04 median, gradient theo tier,
   ablation, các % đóng góp đặc trưng) **chỉ** xuất hiện ở Ch.4–5. **Không** rò lên Ch.1–3.
   Ở Ch.1–3, các khó khăn được nêu ở mức **định tính/tiên nghiệm** (suy luận hợp lý hoặc văn
   liệu sẵn có ủng hộ), không phải dưới dạng kết quả đo được của luận văn.

2. **Ngã rẽ đã bỏ không xuất hiện dưới bất kỳ hình thức nào** — luận văn đứng trong thế giới
   của chính nó, không phải nhật ký hành trình thử nghiệm. Cụ thể: **không nhắc DART, không
   nhắc GHAP/sản phẩm reanalysis PM2.5 toàn cầu như một thứ "đã cân nhắc rồi bỏ", không nhắc
   bản đồ grid phủ toàn quốc** (ngoài phạm vi vì nặng tính toán). Không viết cả câu kiểu "vì
   lý do X nên không dùng Y" — đó vẫn là kéo nhiễu vào. Sự lựa chọn phương pháp được trình bày
   như **lẽ tự nhiên**, tuyên ngôn nằm ở chính sự vắng mặt sạch sẽ.

3. **Tên kỹ thuật cụ thể** (RFSI / MoE / kNN / conformal) chỉ giới thiệu ở **Ch.3**. Ch.1 chỉ
   nói **định hướng** ("đặc trưng không gian", "tổ hợp nhiều mô hình con theo điều kiện nền"),
   không gọi tên cơ chế.

Một quy ước nội dung quan trọng (đã thống nhất qua thảo luận):

- **Khó khăn cốt lõi = sự đa dạng điều kiện nền (regime) giữa các khu vực.** Đây là cái gộp
  của hai thứ trước đây tách rời ("decoupling theo mức ô nhiễm" và "thiếu thông tin đặc trưng
  theo địa điểm"). Một mô hình chung khó nắm bắt toàn bộ sự đa dạng đó — mệnh đề **định tính**
  này nêu được ở Ch.1.
- **tier là một thành phần *bên trong* pipeline cuối, không phải đối thủ của nó.** Trạm mới
  được **ước lượng** tier (không giả định biết trước) — đây là điểm tránh vòng lặp logic
  (circular). "oracle tier" (giả định biết trước tier) là một **lát phân tích** ở Ch.5 về cách
  sắp xếp dữ liệu tạo mô hình con, không phải bộ phận triển khai.
- **global** và **chia theo địa lý** ở Ch.5 là các **baseline/mốc so sánh có chủ đích**, không
  phải ngã rẽ bỏ đi — chúng được phép xuất hiện vì là bước thực nghiệm hợp pháp.

---

## Luận điểm một-đoạn (cho người viết tự định hướng — KHÔNG đưa nguyên văn vào Ch.1)

Ước tính PM2.5 theo giờ trên lãnh thổ Việt Nam từ vệ tinh + khí tượng là **hai bài toán khác
nhau** mà văn liệu thường gộp làm một. *Nội suy thời gian* (dự đoán trạm đã thấy trong huấn
luyện) là dễ. *Ngoại suy không gian* (dự đoán trạm chưa từng thấy — kịch bản triển khai thật)
là khó. Khoảng cách này bắt nguồn từ **sự đa dạng điều kiện nền** giữa các khu vực và **mật độ
trạm** (khoảng cách tới anchor gần nhất). Luận văn xây một pipeline **triển khai được** (định
tuyến chuyên gia theo tier ô nhiễm + đa luồng đặc trưng + dịch tiên nghiệm kNN) thu hồi phần
lớn kỹ năng không gian *khả thi*, và kiểm định ngoại trên mạng cảm biến chi phí thấp (LCS) cùng
trạm tham chiếu US-Embassy.

Thiết bị trung thực lặp lại: luôn báo cáo **R² pooled**, **mean-station**, **median-station**
cùng nhau, kèm **% trạm có R² > 0**, vì chúng phân kỳ mạnh.

---

## Chương 1 — Giới thiệu đề tài · VIẾT NGAY (không cần kết quả)

**Mục tiêu:** đóng khung bài toán — rộng, trung tính, đúng tên đề tài. Không lộ kết quả, không
thu hẹp phạm vi bằng một kết luận tiêu cực ngay từ đầu.

**Tổng quan (đoạn mở chương):** giới thiệu ngắn nội dung chương — bối cảnh và nhu cầu giám sát
không gian, các hạn chế hiện có, mục tiêu/định hướng, đóng góp, bố cục.

### 1.1 Đặt vấn đề
Ba đoạn, thuần động lực; chưa nói khó ở đâu về mặt kỹ thuật.
- PM2.5 là vấn đề cấp bách ở Việt Nam — số liệu thiệt hại sức khỏe/kinh tế; Hà Nội thường xuyên
  trong nhóm đô thị ô nhiễm nhất. *(Chèn citation số liệu.)*
- Quản lý và cảnh báo cần **bản đồ phân bố theo không gian**; mạng trạm tham chiếu thưa, đắt,
  không phủ hết lãnh thổ.
- Viễn thám (AOD vệ tinh) + khí tượng + học máy là hướng ước tính PM2.5 ở nơi không có trạm.
- **Kết mục:** phát biểu bài toán — ước tính PM2.5 mặt đất theo giờ từ vệ tinh và khí tượng
  bằng học máy.

### 1.2 Các giải pháp hiện tại và hạn chế
Mạch B → A, ba khối:
- *Khối B — khoảng trống đánh giá (một đoạn vừa, CÓ dẫn chứng):* văn liệu AOD+ML đạt R² cao,
  nhưng phần lớn đánh giá bằng cross-validation ngẫu nhiên — đo nội suy trên trạm đã thấy,
  không tách bạch với ngoại suy không gian (kịch bản triển khai). Đây là khoảng trống đánh giá
  mà luận văn lấp. *(2–3 citation nghiên cứu dùng random-CV.)*
- *Khối A1 — đa dạng điều kiện nền (định tính):* chất lượng không khí thay đổi theo các regime
  nền khác nhau giữa khu vực, nên một mô hình chung khó nắm bắt toàn bộ sự đa dạng đó.
  *(Có thể citation; KHÔNG con số.)*
- *Khối A2 — hạn chế tín hiệu/dữ liệu (tiên nghiệm, có citation):* AOD thiếu do mây; AOD là
  tích phân cột nên lệch PM2.5 bề mặt (chi phối bởi độ ẩm — hygroscopic growth, và PBLH — phân
  bố thẳng đứng); sản phẩm CTM/reanalysis toàn cầu không nắm động lực theo giờ; cảm biến chi
  phí thấp có sai số.
- *Lưu ý viết:* chỉ nêu **sự tồn tại** của hạn chế ở mức "văn liệu ghi nhận / nguyên lý vật lý
  dự đoán". Không đưa con số đo được của luận văn.

### 1.3 Mục tiêu và định hướng giải pháp
- Mục tiêu: pipeline ước tính PM2.5 **triển khai được** + **đánh giá trung thực** khả năng đó.
- Bắt cặp từng hạn chế ở 1.2 với một định hướng:
  - khoảng trống đánh giá → đánh giá **song song** nội suy thời gian (random CV) và ngoại suy
    không gian (LOSO); báo cáo **đa chỉ số** (pooled / mean / median R², % trạm R²>0).
  - đa dạng điều kiện nền → thiết kế mô hình **biểu diễn được sự đa dạng nền** (đặc trưng không
    gian + tổ hợp nhiều mô hình con theo điều kiện nền).
  - AOD thiếu/lệch, cảm biến nhiễu → QC mạnh, hợp nhất đa nguồn, hiệu chỉnh đặc trưng.
- *Lưu ý:* chỉ giới thiệu **định hướng**, không gọi tên RFSI/MoE/kNN (để Ch.3). Nội suy thời
  gian vs ngoại suy không gian phát biểu như **giả thuyết phương pháp luận sẽ kiểm chứng**,
  không như kết quả.

### 1.4 Đóng góp của đồ án
Liệt kê theo ký tự La Mã (đúng phong cách template):
- (i) bộ dữ liệu 40 trạm + LCS, đa nguồn (vệ tinh + khí tượng + phát thải), QC mạnh;
- (ii) khung đánh giá tách **nội suy thời gian / ngoại suy không gian**, đa chỉ số trung thực;
- (iii) một pipeline ước tính PM2.5 **triển khai được**, kết hợp đặc trưng không gian RFSI,
  định tuyến chuyên gia theo tier ô nhiễm, đa luồng đặc trưng và dịch tiên nghiệm kNN;
- (iv) kiểm định ngoại với mạng LCS và trạm tham chiếu US-Embassy.
- *Không con số ở đây — chỉ nêu bản chất đóng góp.*

### 1.5 Bố cục đồ án
Văn xuôi đầy đủ, mô tả từng chương (Ch.2 → Ch.6). **Tuyệt đối không gạch đầu dòng** (template).

**Kết chương:** tóm bài toán + định hướng; câu nối sang Chương 2.

---

## Chương 2 — Nền tảng lý thuyết · VIẾT NGAY (không cần kết quả)

**Mục tiêu:** trình bày khái niệm và triết lý đánh giá. Phần lớn là giải thích/làm nền — dễ
viết. **Ranh giới với Ch.3:** Ch.2 nói dữ liệu ở tầng "sản phẩm là gì, vật lý thế nào, văn liệu
nói gì"; KHÔNG nói "tôi thu thập/xử lý ra sao" (đó là Ch.3).

**Tổng quan (đoạn mở chương):** nối từ Ch.1; nói Ch.2 trình bày nền tảng — bản chất PM2.5 và
AOD, vì sao quan hệ AOD–PM2.5 phức tạp, các nguồn dữ liệu, và khung đánh giá. Liệt kê mục lớn.

### 2.1 Ngữ cảnh bài toán — PM2.5, AOD và quan hệ giữa chúng
- PM2.5 là gì, tác hại sức khỏe, chuẩn/ngưỡng (WHO, QCVN) — ngắn (Ch.1 đã nêu động lực).
- AOD là gì: độ dày quang học sol khí, **tích phân toàn cột khí quyển**.
- Quan hệ AOD–PM2.5 và vì sao **không** trực tiếp: cột vs bề mặt; PBLH (aerosol nén ở lớp biên
  thấp → AOD trung bình nhưng PM2.5 cao); độ ẩm (hygroscopic growth); loại sol khí (Angstrom).
  *(Chèn citation: decoupling do độ ẩm chi phối; phân bố thẳng đứng/PBLH theo mùa. Ở chế độ
  PM2.5 thấp, tỷ lệ tín hiệu/nhiễu thấp nên tương quan yếu hơn — phát biểu thận trọng: không
  phải mức ô nhiễm thấp "gây ra" tương quan yếu, mà SNR thấp.)*
- *Cầu nối tới Ch.1:* vì quan hệ này phụ thuộc điều kiện khí quyển/nền địa phương nên nó thay
  đổi theo regime — đặt nền cho lý do cần khí tượng và mô hình phân biệt nền.

### 2.2 Các kết quả nghiên cứu tương tự (related work)
- Các hướng ước tính PM2.5 từ AOD: hồi quy tuyến tính/vật lý, mô hình thống kê địa lý (GWR),
  học máy (RF/GBM/XGBoost), học sâu.
- **Phân tích rõ ưu/nhược** (template nhấn mạnh), đặc biệt về **giao thức đánh giá** — nối lại
  khoảng trống random-CV vs spatial đã nêu ở 1.2.
- Đề cập các phương pháp khai thác đặc trưng **không gian từ trạm lân cận** (có tiền lệ văn
  liệu) — làm nền cho RFSI ở Ch.3.
- Kết: nêu bật động lực — khoảng trống mà luận văn lấp.

### 2.3 Dữ liệu vệ tinh và khí tượng (kiến thức nền số 1)
*(Mô tả sản phẩm, KHÔNG mô tả thu thập.)*
- AOD vệ tinh: MODIS MAIAC (1km); Himawari-8 (địa tĩnh, phân giải thời gian cao); đặc tính,
  **hạn chế mây che** (nối Ch.1).
- Khí tượng tái phân tích: ERA5 / Open-Meteo — biến liên quan phân tán ô nhiễm (nhiệt độ, độ
  ẩm, gió, PBLH).
- Nguồn bổ trợ: TROPOMI (khí vết SO2/CO/NO2 — proxy phát thải), MODIS LST, GPM mưa, mật độ xây
  dựng, địa hình. Mỗi nguồn ngắn: là gì, liên quan PM2.5 thế nào.
- Cảm biến chi phí thấp (lớp Plantower): đặc tính và sai số (nối Ch.1; làm nền cho QC ở Ch.3).

### 2.4 Phương pháp học máy và khung đánh giá (kiến thức nền số 2)
- Cây tăng cường gradient (GBM/XGBoost/HGB) — vì sao phù hợp dữ liệu bảng đa nguồn phi tuyến.
  Ngắn gọn, chỉ phần liên quan.
- **Nội suy thời gian vs ngoại suy không gian** — *phần khái niệm cốt lõi (conceptual heart)*:
  định nghĩa hai chế độ; hai giao thức đánh giá tương ứng (random CV vs LOSO); vì sao chúng đo
  hai năng lực khác nhau. Lập luận "thế giới lý tưởng": nếu đặc trưng nắm đủ trạng thái vật lý
  đặc trưng-địa-điểm thì hai bài toán hội tụ; khoảng cách giữa chúng chính là bằng chứng đặc
  trưng còn thiếu một biến tiềm ẩn theo địa điểm (mức nền của trạm). *Viết kỹ.*
- Chỉ số đánh giá: R² pooled / mean-station / median-station, % trạm R²>0 — **vì sao cần cả
  bốn** (chúng phân kỳ mạnh). Đặt nền cho cách báo cáo trung thực ở Ch.5.
- *Lưu ý:* vẫn là **lý thuyết** — định nghĩa giao thức và chỉ số, chưa áp con số nào.

*(Tùy độ dài: nếu phần khung đánh giá đủ nặng, tách 2.4 = phương pháp ML và 2.5 = khung đánh
giá. Mặc định gộp làm 2.4.)*

**Kết chương:** tóm các nền tảng; nối sang Ch.3 (áp các nền này vào pipeline cụ thể).

---

## Chương 3 — Phương pháp đề xuất · VIẾT NGAY (không cần kết quả)

**Mục tiêu:** trình bày pipeline cuối **như một tổng thể đã hoàn thiện**. Mô tả "cái dùng";
arc tiến hóa thiết kế (global → địa lý → tier → cuối) để dành Ch.5.

**Tổng quan (đoạn mở chương):** nối Ch.2; nói Ch.3 trình bày pipeline cụ thể — thu thập và kiểm
soát chất lượng, hợp nhất và xây dựng đặc trưng, kiến trúc mô hình. Liệt kê mục lớn.

### 3.1 Tổng quan giải pháp
- Mô tả pipeline end-to-end thành một mạch: nguồn dữ liệu → QC → hợp nhất → đặc trưng → mô hình
  → đánh giá.
- **Chèn sơ đồ luồng (block diagram).**
- Nêu nguyên tắc thiết kế: triển khai được (đặc trưng suy được tại điểm chưa có trạm); đánh giá
  song song hai chế độ.

### 3.2 Thu thập và kiểm soát chất lượng dữ liệu
*(Phần "tôi làm gì" — đối lập Ch.2 "sản phẩm là gì".)*
- **Nguồn và phạm vi:** 40 trạm (TEDP) + 57 LCS cho kiểm định ngoại; nêu rõ KK là cảm biến chi
  phí thấp (trung thực). Khoảng thời gian 2023-01 → 2026-04.
- **Kiểm soát chất lượng PM2.5 (QC mask mạnh):** quy tắc — flatline ≥5h, stuck-low ≤2µg ≥48h,
  range check. **[LOCKED]** che ~25k dòng (so với 2.6k mask cũ). Quyết định: **giữ cả 40 trạm**;
  3 cảm biến lỗi cũ (Da Nang Pham Hung, Soc Trang, Tra Vinh Dong Hai) được **làm sạch theo
  dòng, không bỏ trạm** (`include_with_sensor_warning`). Tham chiếu
  `results/06_data_quality/report_pm25_qc_effect_validation.txt`.
- **Hợp nhất → `unified_thesis_v4.csv`:** 137 cột, 2023-01 → 2026-04, nới coverage để giữ cả 40.
- **Khí tượng và các nguồn vệ tinh:** khớp theo tọa độ trạm và thời gian; xử lý thiếu hụt (đặc
  biệt AOD do mây).

### 3.3 Xây dựng đặc trưng
*(Mục riêng — đặc trưng là phần nặng và là nơi RFSI nối lại khó khăn cốt lõi.)*
- Nhóm khí tượng + dẫn xuất (PBLH, thông gió VC, đình trệ; persistence).
- Cyclical thời gian (giờ/ngày/mùa).
- AOD (Himawari + MODIS climatology, dạng directional).
- Phát thải (TROPOMI SO2/CO/NO2 anomaly), LST anomaly, mật độ xây dựng, địa hình.
- **RFSI — đặc trưng không gian từ trạm lân cận (PM2.5):** nhóm đặc trưng then chốt cho ngoại
  suy không gian. Nối khó khăn "đa dạng nền / thiếu thông tin địa-điểm" ở Ch.1–2.

### 3.4 Kiến trúc mô hình
*(Pipeline cuối như tổng thể hoàn thiện; tier là thành phần bên trong, dùng **ước lượng**.)*
- Mô hình nền (gradient boosting) dùng cho cả hai chế độ đánh giá.
- **Định tuyến chuyên gia theo tier ô nhiễm (soft-gate):** chọn chuyên gia theo tier; tier của
  trạm mới được **ước lượng** (không giả định biết trước) — điểm tránh circular. *(Cơ chế ước
  lượng tier cụ thể: để Claude Code viết rõ.)*
- **Đa luồng đặc trưng + dịch tiên nghiệm kNN:** tổ hợp nhiều mô hình con trên các tập đặc
  trưng khác nhau; dịch theo mức nền của trạm lân cận (anchoring station means).
- **Conformal (Mondrian)** cho khoảng tin cậy / độ phủ chọn lọc — đánh dấu vùng đáng tin,
  khớp tinh thần trung thực của luận văn. **(Giữ trong phạm vi — đã chạy, không nặng.)**
- *Lưu ý:* mô tả "cái dùng". *Vì sao* thiết kế thế này (baseline global → địa lý → tier → cuối)
  là arc của Ch.5, không kể ở đây.

### 3.5 (tùy chọn) Quy trình đánh giá
- Mô tả ngắn cách dựng random CV và LOSO trên pipeline này, làm cầu sang Ch.5. Hoặc gộp vào 3.1
  tùy độ dài.

**Kết chương:** tóm pipeline đã trình bày; nối sang Ch.4/5.

---

## Chương 4 — Phân tích lý thuyết (mô hình học được gì) · NHÁP PHẦN ỔN ĐỊNH

**Mục tiêu:** cơ chế trước số liệu — *vì sao* nó hành xử như vậy.

- §4.1 **Tương quan AOD–PM2.5** (`aod_pm25_correlation_paper.py`) — **[LOCKED]** cấu trúc tương
  quan giờ/tháng; caveat cột-vs-bề-mặt.
- §4.1 **Trần nội-trạm theo thời gian** (`within_station_predictability_v4.py`) —
  **[v4-PENDING, nhưng ổn định]** KFold R² ≈ 0.73 (40) / 0.80 (all); cận trên khi trạm đã biết.
- §4.2 **Baseline CTM thất bại** (`exp_satellite_products.py`) — **[LOCKED]** GEOS-CF bias
  ≈ +244%, diurnal r ≈ −0.14; MERRA-2 IOA ≈ 0.42. Các chế độ thất bại riêng biệt.
- **Nội suy không gian chiếm ưu thế / ràng buộc anchor gần nhất** — RFSI là nhóm đặc trưng hàng
  đầu (~37% gain); kỹ năng bám theo khoảng cách-tới-anchor. **[v4-PENDING]** cho số ablation.

---

## Chương 5 — Đánh giá thực nghiệm · CẦN KẾT QUẢ v4

**Mục tiêu:** đánh giá định lượng đầy đủ. Đây cũng là nơi kể **arc tiến hóa thiết kế** qua chuỗi
thực nghiệm có chủ đích.

- §5.1 Tham số đánh giá — pooled vs mean/median station R², % trạm R²>0 (viết ngay).
- §5.2 **Nội suy thời gian (random 5-fold CV)** — **[v4-PENDING]** ≈ 0.80; chế độ dễ.
- §5.3 **Ngoại suy không gian (LOSO)** — **[v4-PENDING]** baseline **global** (cận dưới / mốc
  so sánh) → khoảng cách tới nội suy thời gian.
- §5.4 **Chuỗi tiến hóa thiết kế** — **[v4-PENDING]** global (baseline) → **chia theo địa lý**
  (baseline, một cách nhóm regime) → **chia theo tier** (lát phân tích; nêu vấn đề oracle/
  circular khi cần biết trước tier) → **mô hình cuối** tiệm cận vấn đề đa-regime bằng đa luồng
  + dịch kNN, *bao gồm* tier như thành phần (dùng tier ước lượng, không cần biết trước).
  Cô lập tier boost (≈ +0.17 in-file, tới +0.26). Ghi chú trung thực: *mức ô nhiễm vs vùng miền
  như yếu tố điều khiển là giả thuyết dữ liệu không tách được hoàn toàn.*
- §5.5 **Triển khai-được vs trần oracle** — **[v4-PENDING]** so deployable (tier ước lượng,
  đa luồng + kNN; OOF ≈ +0.23) với oracle (giả định biết trước mức nền trạm); chi phí của việc
  không biết mean trạm.
- §5.6 **Kiểm định ngoại** — **[v4-PENDING]** mạng LCS + US-Embassy; hiệu ứng dịch tiên nghiệm
  không gian (≈ +0.16–0.19).
- §5.7 **Bản đồ độ tin cậy conformal** (`conformal_trustmap.py`) — độ phủ chọn lọc / nơi dự
  đoán đáng tin.
- §5.8 **Nguyên lý thống nhất** — nội suy thời gian vs ngoại suy không gian dưới một giải thích
  chung cho dải R² 0.1→0.8. Kèm: nghiên cứu vùng Đồng bằng sông Hồng (`exp_red_river_delta.py`).

---

## Chương 6 — Kết luận · SAU Ch.5

- Phát biểu lại các phát hiện theo trục nội suy thời gian / ngoại suy không gian.
- **Kết luận về khả năng triển khai:** pipeline làm/không làm được gì như một lớp bản đồ.
- Hạn chế: ground-truth cảm biến chi phí thấp; mật độ trạm; confound mức-ô-nhiễm/vùng-miền;
  kiểm định ngoại chỉ mùa khô.
  *(Lưu ý: KHÔNG nhắc DART/GHAP/bản đồ-grid như giới hạn — chúng không tồn tại trong luận văn.)*
- Hướng phát triển: anchor dày hơn (tích hợp LCS làm neo RFSI); mô hình hóa trạm sạch tốt hơn;
  tiering vận hành; kiểm định mùa mưa.

---

## Checklist trạng thái kết quả (điều phối Ch.4–5 sau khi chạy xong)

| Kết quả | Script | Trạng thái |
|--------|--------|--------|
| Hiệu ứng QC mask mạnh | `validate_pm25_qc_effect.py` | **LOCKED** (`results/06_data_quality/`) |
| Tương quan AOD–PM2.5 | `aod_pm25_correlation_paper.py` | **LOCKED** |
| Baseline CTM thất bại | `exp_satellite_products.py` | **LOCKED** (rerun trên v4 để lấy số chính xác) |
| Trần nội-trạm | `within_station_predictability_v4.py` | v4-PENDING (≈0.73/0.80) |
| Baseline LOSO **global** | `exp_true_tier_moe_xgb.py` cfg `no_t4f` | v4-PENDING |
| **Chia theo địa lý** (region_split) | `exp_true_tier_moe_xgb.py` cfg `region_split` | v4-PENDING (rung mới, đã có trên v4) |
| MoE soft-gate + chuyên gia tier | `exp_true_tier_moe_xgb.py` cfg `true_tier_moe_expert` + `tierexpert_t0..3` | v4-PENDING (≈0.43 pooled) |
| **Trần oracle** (giả định biết tier) | `exp_true_tier_moe_xgb.py` cfg `oracle_t4f` | v4-PENDING (mốc tham chiếu) |
| Đa luồng + kNN-3 | `exp_diverse_streams.py` → `exp_diverse_knn_diagnostic.py` | v4-PENDING (≈+0.23 OOF) |
| Kiểm định ngoại LCS | `validate_diverse_knn_lcs.py` | v4-PENDING (≈+0.16–0.19) |
| Độ phủ conformal | `conformal_trustmap.py` | v4-PENDING |

**Thứ tự chạy đầy đủ:** xem `Thesis/RUN_PIPELINE.md` (runbook chính thức). Tóm tắt:
`build_station_feature_table.py` → `exp_true_tier_moe_xgb.py` (1 run, **8 config**:
`oracle_t4f,no_t4f,region_split,true_tier_moe_expert,tierexpert_t0..3` → `himawari_v4_definitive_oof`)
→ `exp_diverse_streams.py` → `exp_diverse_knn_diagnostic.py` →
`within_station_predictability_v4.py` → `validate_diverse_knn_lcs.py`.
Toàn bộ arc §5.4 (global → region → tier → oracle) đến từ **một** lần chạy MoE đó.