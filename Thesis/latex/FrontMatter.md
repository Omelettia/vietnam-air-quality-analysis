# Học máy tăng cường bằng vệ tinh để ước tính PM2.5 theo giờ trên toàn lãnh thổ Việt Nam dưới đánh giá không gian nghiêm ngặt

---

> **Tác giả:** Nguyễn Tài Khoa
>
> **Bậc đào tạo:** Kỹ sư Công nghệ Thông tin (Chương trình Việt–Nhật)
>
> **Cơ sở đào tạo / Đơn vị:** Trường Công nghệ Thông tin và Truyền thông, Đại học Bách khoa Hà Nội (HUST)
>
> **Giảng viên hướng dẫn:** TS. Trần Nguyên Ngọc
>
> **Thời gian:** Tháng 7 năm 2026

---

## Tóm tắt

Ô nhiễm không khí là một trong những mối đe dọa sức khỏe môi trường nghiêm trọng nhất mà Việt Nam đang đối mặt, thế nhưng chỉ khoảng bốn mươi trạm quan trắc tự động trên toàn bộ 331.000 km² lãnh thổ cung cấp dữ liệu PM2.5 liên tục trong giai đoạn nghiên cứu — ba mươi bảy trạm sau kiểm soát chất lượng — khiến phần lớn dân số không có thông tin chất lượng không khí tại địa phương. Học máy tăng cường bằng vệ tinh đã được đề xuất như một giải pháp, với các nghiên cứu ở những khu vực được quan trắc dày đặc thường xuyên báo cáo hệ số xác định trên 0,8. Luận văn này đặt câu hỏi liệu hiệu suất như vậy có khả thi đối với mạng lưới thưa thớt, nhiệt đới của Việt Nam hay không, và đối diện với một vấn đề phương pháp luận mang tính hệ thống: sự chênh lệch giữa đánh giá chéo ngẫu nhiên được dùng trong phần lớn tài liệu khoa học và đánh giá chéo không gian, vốn phản ánh bài toán vận hành thực tế là dự đoán tại các vị trí không có trạm quan trắc.

Một mô hình XGBoost-DART với 66 đặc trưng được huấn luyện trên 37 trạm quan trắc tự động, sử dụng độ sâu quang học sol khí từ MODIS và Himawari, khí vết TROPOMI, khí tượng ERA5, đầu ra mô hình vận chuyển hóa học (CTM), và các biến dự đoán sử dụng đất, rồi được đánh giá dưới đánh giá chéo bỏ-một-trạm-ra (LOSO) và đánh giá độc lập với một mạng lưới cảm biến chi phí thấp (LCS) và một trạm tham chiếu chuẩn của Đại sứ quán Hoa Kỳ. Kết quả trọng tâm là một chuỗi ba con số: R² khoảng 0,80 dưới đánh giá chéo ngẫu nhiên sụt xuống còn khoảng 0,20 dưới LOSO trung thực, và còn khoảng 0,04 (trung vị theo trạm) khi rút bỏ thông tin oracle. Phân tầng theo mức ô nhiễm (sơ đồ T4F) là đòn bẩy hiệu suất lớn nhất, tạo ra con số LOSO nói trên, nhưng nó kéo theo một phụ thuộc vòng tròn — việc gán tầng cho một trạm đòi hỏi phải biết trung bình PM2.5 của trạm đó, chính là đại lượng cần dự đoán — và bảy họ phương pháp xấp xỉ dựa trên biến quan sát được từ vệ tinh đều thất bại trong việc giải quyết nó. Tuy vậy, phụ thuộc này vẫn được giải quyết từ mạng lưới quan trắc chứ không phải từ các biến quan sát: một mô hình định tuyến tiên nghiệm không gian neo đường cơ sở của mỗi trạm vào trung bình quan sát của các trạm lân cận, phục hồi R² trung bình theo trạm triển khai được là 0,197 so với ngưỡng trần oracle 0,203, không có phân loại sai nguy hiểm thấp-thành-cao, với điều kiện gần mạng lưới. Mô hình CTM và các sản phẩm toàn cầu hiện có thất bại theo những cách riêng biệt: GEOS-CF cho chu kỳ ngày đêm đảo ngược kèm sai số vượt quá 200%, MERRA-2 chỉ đạt Chỉ số Phù hợp 0,39, và GHAP chỉ đạt khoảng 50% mức phù hợp về phân tầng. Đánh giá độc lập cho kết quả tốt trong vùng phủ dày đặc, với R² trung vị LCS là 0,53 và R² tại Đại sứ quán là 0,68, suy giảm theo khoảng cách đến trạm neo gần nhất. Do đó, ràng buộc cuối cùng là mật độ trạm chứ không phải năng lực thuật toán, và việc làm dày mạng lưới bằng các cảm biến chi phí thấp đã hiệu chuẩn là con đường khả thi nhất để cải thiện độ chính xác trên toàn quốc.

**Từ khóa:** PM2.5, viễn thám, độ sâu quang học sol khí, học máy, đánh giá chéo không gian, Việt Nam, cảm biến chi phí thấp

---

## Mục lục

**Chương 1: Giới thiệu**
- 1.1 Đặt vấn đề
- 1.2 Các giải pháp hiện có và hạn chế
- 1.3 Mục tiêu và hướng giải pháp
- 1.4 Đóng góp của luận văn
- 1.5 Cấu trúc luận văn

**Chương 2: Cơ sở lý thuyết**
- 2.1 Ngữ cảnh của bài toán dự đoán
- 2.2 Các kết quả nghiên cứu tương tự
- 2.3 Mối quan hệ AOD–PM2.5
- 2.4 XGBoost và chính quy hóa DART
- 2.5 Phương pháp luận đánh giá không gian

**Chương 3: Phương pháp đề xuất**
- 3.1 Tổng quan giải pháp
- 3.2 Khu vực nghiên cứu và nguồn dữ liệu
- 3.3 Kiểm soát chất lượng dữ liệu
- 3.4 Kỹ thuật đặc trưng
- 3.5 Mô hình hóa phân tầng (T4F)
- 3.6 Các phương pháp triển khai được
- 3.7 Mô hình vùng đồng bằng

**Chương 4: Phân tích và bàn luận**
- 4.1 Phân tích tầm quan trọng đặc trưng
- 4.2 Đánh giá các sản phẩm mô hình vận chuyển hóa học
- 4.3 Nghiên cứu loại bỏ: chỉ vệ tinh so với trạm mặt đất
- 4.4 Đánh giá các bản đồ PM2.5 hiện có
- 4.5 Mật độ trạm như ràng buộc then chốt

**Chương 5: Đánh giá thực nghiệm**
- 5.1 Các tham số đánh giá
- 5.2 Dự đoán thời gian (đánh giá chéo ngẫu nhiên)
- 5.3 Dự đoán không gian (LOSO — cấu hình toàn cục)
- 5.4 Kết quả phân tầng theo mức ô nhiễm
- 5.5 So sánh các mô hình triển khai được
- 5.6 Đánh giá độc lập (mạng lưới LCS)
- 5.7 Tổng hợp các thí nghiệm không thành công
- 5.8 Một góc nhìn thống nhất: nội suy và ngoại suy

**Chương 6: Kết luận**
- 6.1 Tổng hợp các phát hiện
- 6.2 Khuyến nghị cho chiến lược quan trắc chất lượng không khí của Việt Nam
- 6.3 Hạn chế
- 6.4 Hướng phát triển trong tương lai
- Lời kết

**Tài liệu tham khảo**

---

## Danh mục hình vẽ

**Hình 3.1** Quy trình định tuyến tiên nghiệm không gian — kiến trúc mô hình triển khai được gồm sáu giai đoạn, từ đầu vào 66 biến qua tổ hợp XGBoost-DART, tính toán tiên nghiệm không gian, định tuyến ba chế độ, bộ bảo vệ độ tin cậy với hiệu chỉnh mùa từ MODIS, đến ước tính PM2.5 theo giờ cuối cùng.

**Hình 4.1** Chu kỳ ngày đêm PM2.5 của GEOS-CF so với quan trắc mặt đất, cho thấy mối quan hệ gần như ngược pha trong đó sản phẩm đạt cực tiểu vào giờ rạng đông khi nồng độ bề mặt quan trắc đạt đỉnh.

**Hình 4.2** Mẫu sai số theo trạm của GEOS-CF trên toàn mạng lưới quan trắc, minh họa mức ước tính vượt hệ thống gấp hai đến ba lần nồng độ tuyệt đối.

**Hình 4.3** Biểu đồ tán xạ tại một trạm đại diện của GEOS-CF giữa giá trị mô hình hóa và giá trị quan trắc PM2.5, đặc trưng cho hệ số xác định âm theo trạm của sản phẩm.

**Hình 4.4** Biểu đồ tán xạ tại một trạm đại diện của MERRA-2 giữa giá trị mô hình hóa và giá trị quan trắc PM2.5, đặc trưng cho tương quan thời gian theo giờ yếu của nó mặc dù xấp xỉ cân bằng khối lượng.

**Hình 4.5** Thành phần loài sol khí của MERRA-2 theo vùng, cho thấy sản phẩm tái tạo được gradient khối lượng sol khí bắc–nam tổng thể nhưng không tái tạo được biến thiên theo từng giờ.

**Hình 4.6** Xếp hạng trạm theo khí hậu năm của GHAP so với trung bình quan trắc theo trạm, cho thấy kỹ năng không gian ở mức trung bình đi kèm sàn giá trị bị nâng cao làm sai thứ tự các vị trí sạch nhất.

**Hình 5.1** R² theo trạm dưới đánh giá bỏ-một-trạm-ra trên 37 trạm quan trắc tự động, ánh xạ lên bản đồ Việt Nam và tô màu theo kỹ năng: một vùng đồng bằng sông Hồng sáng và liền mạch ở phía Bắc, hai điểm sáng nhưng cô lập ở Thành phố Hồ Chí Minh, và một vành đai ven biển và đồng bằng sông Cửu Long mờ nhạt — một địa lý của mức ô nhiễm chứ không phải của vùng miền.

---

## Danh mục bảng

**Bảng 5.1** Config H đầy đủ — R² đánh giá chéo ngẫu nhiên theo từng fold (năm fold và gộp ngoài-fold).

**Bảng 5.2** Hiệu suất đánh giá chéo ngẫu nhiên trên toàn bộ quét cấu hình (đặc trưng, R² KFold, MAE, RMSE).

**Bảng 5.3** Các khung đánh giá cho Config H — đánh giá chéo ngẫu nhiên, LOSO theo ngày giả (rò rỉ), LOSO theo ngày và theo giờ trung thực, và khoảng cách rò rỉ.

**Bảng 5.4** Các cấu hình LOSO toàn cục — không phân nhóm, true-tier (T4F), và base-margin oracle: các tham số gộp, trung bình, trung vị, và tỷ lệ dương.

**Bảng 5.5** Sáu biến thể huấn luyện DART của definitive_v3 dưới LOSO.

**Bảng 5.6** R² LOSO theo trạm cho cấu hình phi-oracle tốt nhất (T4F), sắp xếp theo tầng và R² trong tầng.

**Bảng 5.7** Kết quả LOSO phân tầng cho biến thể DART cơ sở (số trạm theo tầng, trung bình PM2.5, R², RMSE, MAE, sai số).

**Bảng 5.8** Kết quả LOSO phân tầng cho biến thể tổ hợp tốt nhất tổng thể.

**Bảng 5.9** So sánh từng vùng của các trạm t2, chứng minh rằng mức ô nhiễm chứ không phải vùng miền chi phối khả năng dự đoán.

**Bảng 5.10** Mức tăng +0,21 do phân tầng — so sánh true-tier với không phân nhóm trên các biến thể và tập trạm.

**Bảng 5.11** So sánh mô hình triển khai được với mô hình oracle (trung bình, trung vị, tỷ lệ dương, trung bình t3).

**Bảng 5.12** Chi phí thông tin của trung bình trạm bị thiếu — khoảng cách oracle-trừ-triển-khai trên các tập trạm tương ứng.

**Bảng 5.13** Mô hình định tuyến tiên nghiệm không gian so với ngưỡng trần oracle và dải triển khai dựa trên biến quan sát từ vệ tinh dưới LOSO không-GHAP.

**Bảng 5.14** Kết quả LOSO vùng đồng bằng sông Hồng (base-margin đồng bằng, RFSI, và ngưỡng trần oracle).

**Bảng 5.15** Các tập con đánh giá LCS độc lập — chỉ LCS, tất cả các vị trí bao gồm Đại sứ quán, và kết quả đơn vị trí Đại sứ quán Hoa Kỳ.

**Bảng 5.16** Mười vị trí chi phí thấp độc lập được dự đoán tốt nhất (R², khoảng cách đến trạm KK gần nhất, số giờ, RMSE).

**Bảng 5.17** Các vị trí chi phí thấp độc lập được dự đoán kém nhất (R², khoảng cách, số giờ, RMSE).

**Bảng 5.18** Thống kê tóm tắt mối quan hệ khoảng cách–kỹ năng (tương quan Pearson và Spearman, trung vị nhóm gần và xa, dải khoảng cách).

**Bảng 5.19** Quy trình định tuyến tiên nghiệm không gian trên bốn mươi sáu cảm biến chi phí thấp chưa từng thấy (huấn luyện trên bốn mươi, dự đoán phần chưa thấy).

**Bảng 5.20** Tổng hợp bảy họ thí nghiệm không thành công (phương pháp, R² trung bình theo trạm tốt nhất, và lý do thất bại).

**Bảng 5.21** Mô hình đặc trưng ngoại sinh theo trạm được đánh giá như bốn bài toán khác nhau (lấp khoảng trống, dự báo, lập bản đồ không gian), theo tầng ô nhiễm.

---

## Lời cảm ơn

Tác giả xin gửi lời cảm ơn đến giảng viên hướng dẫn, TS. Trần Nguyên Ngọc, và hội đồng phản biện vì sự hướng dẫn và những phản hồi quý báu trong suốt quá trình thực hiện công trình này. Tác giả chân thành cảm ơn các thầy cô và cán bộ Trường Công nghệ Thông tin và Truyền thông cũng như Bộ môn Kỹ thuật Máy tính tại Đại học Bách khoa Hà Nội vì sự hỗ trợ. Tác giả vô cùng biết ơn Trung tâm Quan trắc Môi trường Việt Nam và những người vận hành mạng lưới cảm biến chi phí thấp đã cung cấp dữ liệu thực địa mà nghiên cứu này dựa vào. Cuối cùng, tác giả xin cảm ơn gia đình, bạn bè và đồng nghiệp, những người đã động viên và hỗ trợ để luận văn này được hoàn thành.
