# CHƯƠNG 5: ĐÁNH GIÁ THỰC NGHIỆM

Chương 4 đã giải thích những gì mô hình học và vì sao nó hành xử như vậy: nội suy không gian từ các trạm lân cận mặt đất chi phối sức dự đoán của nó, vệ tinh thay thế một phần cho tín hiệu đó, các sản phẩm mô hình vận chuyển hóa học thất bại theo ba phương diện riêng biệt, và mật độ trạm — được biểu đạt như sự gần kề với trạm neo gần nhất — là ràng buộc then chốt. Chương này trình bày đánh giá định lượng đầy đủ mà các phân tích đó được xây dựng để giải thích. Mục 5.1 định nghĩa các tham số và giải thích vì sao các bản tóm tắt gộp và theo-trạm phải được báo cáo cùng nhau. Mục 5.2 báo cáo kết quả đánh giá chéo ngẫu nhiên, con số "bài-báo-công-bố" bị phóng đại mà phần còn lại của chương tồn tại để xì hơi. Mục 5.3 trình bày đánh giá bỏ-một-trạm-ra (LOSO) trung thực của mô hình toàn cục và định lượng khoảng cách rò rỉ giữa hai chế độ. Mục 5.4 phân tầng kết quả LOSO theo mức ô nhiễm, xác lập gradient đơn điệu, bất biến-mô-hình mà theo đó khả năng dự đoán tỷ lệ với mức ô nhiễm — đồng thời coi khẳng định xa hơn rằng mức chứ không phải vùng chi phối kỹ năng đó là một giả thuyết mà dữ liệu chưa thể tách bạch — và cô lập mức tăng do phân tầng (khoảng +0,17 trong-cùng-tệp, lên đến +0,26 qua các cách diễn đạt). Mục 5.5 so sánh mọi cấu hình thực sự triển khai được với ngưỡng trần oracle và đo chi phí thông tin của việc không biết trung bình của một trạm. Mục 5.6 trình bày đánh giá độc lập với mạng lưới cảm-biến-chi-phí-thấp và trạm tham chiếu Đại sứ quán Hoa Kỳ. Mục 5.7 tổng hợp bảy họ thí nghiệm không thành công vốn giới hạn không gian giải pháp. Mục 5.8 gắn kết ba khung lại với nhau dưới một nguyên lý nội-suy-so-với-ngoại-suy duy nhất giải thích vì sao cùng một mô hình trải dài R² từ dưới 0,1 đến trên 0,8. Xuyên suốt, chuỗi ba-con-số trọng tâm đã nêu ở Chương 1 được làm rõ: R² khoảng 0,80 dưới đánh giá chéo ngẫu nhiên, khoảng 0,20 dưới LOSO trung thực, và một trung vị theo trạm triển khai được gần 0,04.


## 5.1 Các tham số đánh giá

Tất cả các mô hình trong luận văn này được đánh giá so với PM2.5 thực địa theo giờ bằng bốn tham số. Hệ số xác định được tính ở độ phân giải theo giờ, ký hiệu R²_hourly, đo phần phương sai trong các nồng độ theo giờ quan trắc mà mô hình giải thích so với một đường cơ sở trung-bình-hằng-số; nó là tham số nổi bật chính và bị chặn trên bởi 1, bằng 0 khi mô hình không làm tốt hơn việc dự đoán trung bình dài hạn, và không bị chặn dưới, trở nên âm bất cứ khi nào mô hình tệ hơn trung bình đó. Sai số quân phương trung bình (RMSE) và sai số tuyệt đối trung bình (MAE), cả hai tính bằng µg/m³, định lượng độ lớn điển hình của sai số dự đoán, với RMSE đặt trọng số nặng hơn cho các sai số lớn; và độ thiên lệch (bias), phần dư có dấu trung bình tính bằng µg/m³, ghi lại sự dự đoán vượt hoặc thiếu một cách hệ thống. Các tham số này được báo cáo cùng với R² bởi vì hai mô hình có R² giống hệt nhau có thể khác biệt đáng kể về độ chính xác tuyệt đối, và bởi vì tại các vị trí sạch một R² gần-không hoặc âm cùng tồn tại với một RMSE nhỏ — phương sai cần giải thích bản thân nó đã nhỏ.

Một điểm phương pháp luận trọng tâm của chương này là sự phân định giữa R² gộp và R² trung bình (hoặc trung vị) theo trạm, và lý do cả hai phải được báo cáo. R² gộp được tính bằng cách nối các phần dư theo giờ của tất cả các trạm thành một vectơ duy nhất và đánh giá hệ số xác định một lần trên toàn bộ bộ dữ liệu; nó bị chi phối bởi các trạm phương sai cao, nồng độ cao bởi vì các trạm đó đóng góp cả phần phương sai nhiều nhất cần giải thích và, thường, số giờ nhiều nhất. R² theo trạm, ngược lại, được tính riêng cho từng trạm giữ lại rồi được tóm tắt qua các trạm bằng trung bình hoặc trung vị của nó; nó đặt trọng số mỗi trạm bằng nhau bất kể nồng độ hay độ dài bản ghi của nó và do đó phơi bày cách mô hình hoạt động tại vị trí điển hình thay vì tại số ít vị trí chi phối phương sai gộp. Hai con số phân kỳ mạnh trong bộ dữ liệu này: một số ít trạm sạch, phương sai thấp với R² âm kéo trung bình theo trạm xuống thấp hơn nhiều so với trung vị theo trạm, trong khi con số gộp, được neo bởi các trạm phía Bắc ô nhiễm, nằm cao hơn cả hai. Việc chỉ báo cáo một trong ba bản tóm tắt sẽ gây hiểu lầm một cách hệ thống — giá trị gộp phóng đại hiệu suất điển hình, trung bình theo trạm hạ thấp nó vì một số ít vị trí sạch âm nặng, và trung vị theo trạm là xu hướng trung tâm bền vững nhất. Ở nơi một kết quả phụ thuộc vào lựa chọn bản tóm tắt, cả ba đều được đưa ra. Một lưu ý thực tiễn áp dụng xuyên suốt: các tệp kết quả LOSO theo trạm chỉ lưu R²_hourly theo trạm (cùng với RMSE, MAE, và bias đi kèm) và không giữ lại các phần dư theo giờ thô, nên đối với các thí nghiệm đó một R² gộp liên-trạm thực sự không thể được tính lại sau khi đã xong; ở nơi thuật ngữ "gộp" được dùng cho các cấu hình LOSO, nó biểu thị trung bình có trọng số-số-giờ của các giá trị theo trạm, điều được nêu rõ ở mỗi lần sử dụng và không giống hệt với một R² vectơ-phần-dư duy nhất.

Một chỉ báo thứ ba, thực tiễn một cách có chủ ý, đi kèm với các tham số phương sai: tỷ lệ phần trăm các trạm giữ lại đạt R² dương. Bởi vì một R² âm nghĩa là mô hình tệ hơn việc chỉ đơn giản giả định trung bình dài hạn riêng của trạm, tỷ lệ các trạm trên không trả lời một câu hỏi mà R² trung bình che khuất — tại bao nhiêu vị trí mô hình thực sự sẽ hữu dụng như một bộ ước tính được triển khai? Một cấu hình có thể đăng một R² trung bình đáng nể trong khi thất bại hoàn toàn tại gần một nửa số trạm của nó, và chỉ báo tỷ-lệ-dương là cái phơi bày thất bại đó. Nó được báo cáo cho mọi cấu hình LOSO và đánh giá độc lập trong chương này.


## 5.2 Dự đoán thời gian (đánh giá chéo ngẫu nhiên)

Chế độ đánh giá đầu tiên là đánh giá chéo ngẫu nhiên, chuẩn mực mà phần lớn tài liệu khoa học PM2.5 đã công bố báo cáo độ chính xác của nó. Trong chế độ này 244.489 quan trắc theo giờ của mô hình chính được xáo trộn và phân hoạch không quan tâm đến danh tính trạm, sao cho các giờ từ mọi trạm xuất hiện ở cả fold huấn luyện và fold kiểm tra. Một làm rõ về tên gọi là cần thiết ngay từ đầu: kế hoạch dự án ban đầu mô tả đây là đánh giá chéo mười-fold, nhưng phần triển khai tạo ra các tệp kết quả sử dụng một phân hoạch xáo trộn năm-fold (một phép chia KFold với năm fold, bật xáo trộn, và một hạt giống ngẫu nhiên cố định). Không có tệp kết quả mười-fold nào tồn tại ở bất cứ đâu trong bản ghi thực nghiệm; do đó khẳng định được hỗ trợ là đánh giá chéo ngẫu nhiên năm-fold, và các con số dưới đây được báo cáo trên cơ sở đó. Sự phân định không ảnh hưởng đến con số nổi bật — số lượng fold thay đổi ước tính phương sai, không phải xu hướng trung tâm — nhưng nó được ghi lại ở đây để chính xác.

Mô hình được đánh giá trong chế độ này là Config H — ba mươi hai đặc trưng dẫn dắt một bộ hồi quy XGBoost-DART với tám trăm cây ở độ sâu tám — cấu hình mà kết quả đánh giá chéo ngẫu nhiên đã được tính. Mô hình hoàn chỉnh sáu mươi sáu đặc trưng được mô tả ở Chương 3 được mô tả đặc trưng dưới đánh giá bỏ-một-trạm-ra thay vì chạy lại dưới đánh giá chéo ngẫu nhiên; do đó khoảng cách rò rỉ của Mục 5.3 được chứng minh trên Config H, cấu hình duy nhất mà cả hai chế độ đều được đánh giá trên một mô hình giống hệt, và mô hình hoàn chỉnh là đối tượng của phân tích không gian từ Mục 5.3 trở đi. Dưới đánh giá chéo ngẫu nhiên năm-fold, Config H đạt R² ngoài-fold là 0,8125, với MAE gộp 7,63 µg/m³ và RMSE 11,12 µg/m³ trên tất cả 244.489 giờ. Đây là "con số bài-báo-công-bố" — con số trực tiếp so sánh được với các kết quả R² > 0,8 chi phối tài liệu khoa học cho Trung Quốc, Hoa Kỳ, và châu Âu — và nó là giá trị đã nêu ở Chương 1 như neo đánh-giá-chéo-ngẫu-nhiên khoảng 0,80. Kết quả ổn định qua các fold: năm giá trị R² ngoài-fold chỉ trải từ 0,8078 đến 0,8172, một độ trải dưới một điểm rất nhiều, xác nhận rằng con số 0,81 không phải là một hiện vật của một phân hoạch may mắn.

*Bảng 5.1: R² ngoài-fold theo từng fold của Config H dưới đánh giá chéo ngẫu nhiên năm-fold, với giá trị gộp.*

| Config H ĐẦY ĐỦ — R² đánh giá chéo ngẫu nhiên theo từng fold |
|---|
| fold 0 = 0,8141 |
| fold 1 = 0,8095 |
| fold 2 = 0,8139 |
| fold 3 = 0,8172 |
| fold 4 = 0,8078 |
| gộp ngoài-fold = 0,8125 |

Trên toàn bộ quét cấu hình, R² đánh giá chéo ngẫu nhiên chiếm một dải nhất quán với khoảng 0,72–0,81 được dự đoán trong dàn ý, với con số mô-hình-chính nằm ở đỉnh dải đó và cấu hình giàu-đặc-trưng nhất ngay trên nó. Một mô hình chỉ-khí-tượng (Track A, mười hai đặc trưng) và mô hình hợp nhất khí-tượng-cộng-AOD-thô đều đạt 0,7386; Config G hai mươi ba đặc trưng đạt 0,7398; Config H ở siêu tham số trung gian đạt 0,7217; Config H chính ở siêu tham số đầy đủ đạt 0,8125; và Config I bốn mươi đặc trưng đạt 0,8348. Do đó cận dưới của dải là một tập đặc trưng tinh gọn hơn hoặc các cây nông hơn, và cận trên là cấu hình giàu nhất, với mô hình chính được chỉ định có chủ ý chọn ở 0,8125 như sự thỏa hiệp có thể bảo vệ nhất giữa độ khớp và tính tiết kiệm.

*Bảng 5.2: R², MAE, và RMSE đánh giá chéo ngẫu nhiên trên toàn bộ quét cấu hình, từ đường cơ sở chỉ-khí-tượng đến mô hình giàu-đặc-trưng nhất.*

| Cấu hình | Đặc trưng | R² KFold ngẫu nhiên | MAE | RMSE |
|---|---|---|---|---|
| Track A (chỉ khí tượng) | 12 | 0,7386 | 8,90 | 13,14 |
| Hợp nhất (khí tượng + AOD thô) | 16 | 0,7386 | 8,90 | 13,14 |
| Config G ĐẦY ĐỦ | 23 | 0,7398 | — | — |
| Config H (trung gian, n300/d7) | 32 | 0,7217 | 9,30 | 13,55 |
| **Config H ĐẦY ĐỦ (mô hình chính)** | **32** | **0,8125** | **7,63** | **11,12** |
| Config I ĐẦY ĐỦ | 40 | 0,8348 | 7,20 | 10,44 |

Các quét độ bền độc lập dưới các hạt giống fold thay thế chứng thực mức này: trên năm tệp thí nghiệm KFold riêng biệt, mô hình lớp-Config-H trả về các giá trị R² tụ giữa 0,7225 và 0,8117, với RMSE gần 11–13 µg/m³ và MAE gần 6,5–8,0 µg/m³, xác nhận rằng khoảng 0,80–0,81 là hiệu suất đánh giá chéo ngẫu nhiên thực sự của mô hình này chứ không phải một lần chạy thuận lợi đơn lẻ.

Điều mà chế độ này xác lập, và điều nó không, phải được nêu chính xác. Nó xác nhận rằng tập đặc trưng và thuật toán học có khả năng biểu diễn động học theo giờ của PM2.5 Việt Nam: cho các giờ từ một trạm trong huấn luyện, mô hình tái dựng các giờ còn lại của trạm đó đến trong vòng một R² 0,81. Nhưng bởi vì mọi giờ kiểm tra thuộc về một trạm mà mô hình đã thấy, kết quả đo nội suy thời gian dưới rò rỉ danh-tính-trạm, không phải dự đoán không gian tại một vị trí không có quan trắc. Hai kênh rò rỉ làm điều này cụ thể. Kênh thô là bản thân danh tính trạm: mô hình ngầm học mức nền của mỗi trạm từ các giờ huấn luyện của nó và chỉ cần dự đoán các độ lệch quanh một đường cơ sở mà nó đã biết. Kênh tinh được ghi nhận trong tập đặc trưng — một trong ba mươi hai đặc trưng, khí hậu AOD theo tháng theo trạm, được tính theo trạm và theo tháng, nên chữ ký khí hậu riêng của mỗi trạm được nướng trực tiếp vào các biến dự đoán của nó; đặc trưng đơn lẻ này là tác nhân đóng góp lớn nhất cho độ tăng của mô hình trong chế độ này. Do đó con số 0,81 xác nhận rằng mô hình hoạt động như một bộ nội suy thời gian nhưng về cơ bản không nói gì về năng lực của nó trong việc tổng quát hóa đến một vị trí không có lịch sử mặt đất. Việc định lượng chính xác bao nhiêu trong 0,81 đó là rò rỉ chứ không phải kỹ năng không gian là nhiệm vụ của Mục 5.3.


## 5.3 Dự đoán không gian (LOSO — cấu hình toàn cục)

Đánh giá chéo bỏ-một-trạm-ra loại bỏ rò rỉ danh-tính-trạm vốn phóng đại con số đánh giá chéo ngẫu nhiên. Mỗi trong ba mươi bảy trạm được giữ lại lần lượt được giữ lại, mô hình được huấn luyện lại trên các trạm còn lại với trạm giữ lại bị loại khỏi tập lân cận của mọi trạm khác, và trạm giữ lại sau đó được dự đoán như thể nó là một vị trí không có quan trắc. Đây là bài toán có ý nghĩa vận hành — ước tính PM2.5 ở nơi không có phép đo mặt đất — và nó là tham số đánh giá chính của luận văn này.

Một làm rõ về nguồn là cần thiết, bởi vì các cấu hình LOSO toàn cục được phân bố trên một số tệp thí nghiệm chứ không tập trung trong một. Tệp biến-thể-mô-hình definitive_v3 chứa sáu biến thể huấn luyện DART — một mô hình cơ sở, ba biến thể trạm-lân-cận, một mô hình được tinh chỉnh, và một tổ hợp — mỗi biến thể mang nhãn bốn-tầng thực và do đó là một mô hình true-tier; biến thể được tinh chỉnh trong tệp đó giống-hệt-từng-byte, trên tất cả ba mươi bảy trạm, với cấu hình được gán nhãn "T4F" trong tệp thí-nghiệm-phân-nhóm, điều này xác lập liên kết giữa hai. Mô hình toàn cục thực sự mù-tầng — một mô hình gộp duy nhất không phân nhóm tầng — là cấu hình "no-grouping" trong tệp thí-nghiệm-phân-nhóm, và cận trên oracle tiêm trung bình PM2.5 thực của mỗi trạm giữ lại như một độ dời cộng tính là cấu hình "oracle base-margin" trong tệp hai-pha. Ba cấu hình này là các đối tượng riêng biệt, và việc phân biệt chúng là thiết yếu bởi vì chuỗi LOSO nổi bật phụ thuộc vào cái nào là cái nào.

Sự tương phản với đánh giá chéo ngẫu nhiên là kết quả trọng tâm của chương. Cùng một mô hình Config H đạt 0,8125 dưới đánh giá chéo ngẫu nhiên năm-fold rơi xuống R² trung bình theo trạm 0,2093 dưới LOSO theo giờ trung thực — một mức giảm 0,603 gần như hoàn toàn quy cho việc loại bỏ rò rỉ danh-tính-trạm. Trung vị theo trạm dưới LOSO trung thực là 0,3305, MAE gộp tăng lên 11,99 µg/m³, và RMSE lên 16,86 µg/m³. Khoảng cách 0,603 là nội dung định lượng của khẳng định phương pháp luận trọng tâm của luận văn này: khoảng ba phần tư của con số 0,81 được ca ngợi là rò rỉ, không phải tổng quát hóa.

Một điểm chính xác về cái mà khoảng cách 0,603 lấy hiệu số. Con số LOSO theo giờ trung thực 0,2093 là kết quả của lần chạy Config H mười-lăm-trạm, trong đó mô hình Config H ba mươi hai đặc trưng được huấn luyện lại theo từng trạm giữ lại; nó khác biệt với mô hình hoàn chỉnh ba mươi bảy trạm được báo cáo từ Mục 5.3 trở đi, mà trung bình theo trạm của nó là 0,1989. Do đó khoảng cách rò rỉ được đo trên mô hình Config H giống hệt — R² KFold ngoài-fold 244.489-giờ trên tất cả các trạm của nó là 0,8125 so với R² bỏ-một-trạm-ra trung thực riêng của cùng mô hình đó là 0,2093 — chứ không phải bằng cách lấy hiệu số hai mô hình được đánh giá trên các tập trạm khác nhau. Hai con số LOSO, 0,2093 (Config H, mười lăm trạm) và 0,1989 (mô hình hoàn chỉnh, ba mươi bảy trạm), thống nhất chặt chẽ và cả hai đều hiện thực hóa neo LOSO khoảng 0,20, nhưng chỉ con số đầu chia sẻ mô hình của nó với con số đánh-giá-chéo-ngẫu-nhiên và do đó là con số được dùng để định lượng chi phí rò rỉ.

*Bảng 5.3: R² và MAE của Config H qua các chế độ đánh giá, từ LOSO ngẫu nhiên rò rỉ và theo-ngày-giả đến LOSO huấn-luyện-lại-theo-fold trung thực, với khoảng cách rò rỉ.*

| Chế độ đánh giá (Config H, 32 đặc trưng) | R² | MAE | Ghi chú |
|---|---|---|---|
| Đánh giá chéo ngẫu nhiên 5-fold (con số công bố) | 0,8125 | 7,63 | rò rỉ danh tính trạm |
| LOSO theo ngày giả, chỉ mô hình (rò rỉ) | 0,8684 | 3,77 | dựa trên KFold, rò rỉ danh tính |
| LOSO theo ngày giả, mô hình + AOD + kriging (rò rỉ, tốt nhất) | 0,8882 | 3,32 | con số rò rỉ cao nhất |
| LOSO theo ngày thực, chỉ mô hình (trung thực) | 0,168 | — | huấn luyện lại theo từng trạm giữ lại |
| LOSO theo giờ trung thực (trung thực, tham số chính) | 0,2093 | 11,99 | Config H, LOSO trung thực 15-trạm |
| **Khoảng cách rò rỉ (CV ngẫu nhiên − LOSO theo giờ trung thực)** | **−0,6032** | — | cái giá của danh tính trạm |

Bảng này cũng ghi lại một kết quả trung gian mang tính cảnh báo. Một đánh giá độ-phân-giải-theo-ngày trước đó báo cáo một R² LOSO có vẻ xuất sắc 0,8684 cho chỉ mô hình, tăng lên 0,8882 khi AOD vệ tinh và hậu xử lý kriging-thông-thường được thêm vào. Các con số này không phải LOSO trung thực. Mặc dù được gán nhãn "LOSO" trong script tạo ra chúng, các dự đoán theo giờ cấp cho chúng đến từ một KFold năm-fold được huấn luyện trên tất cả các trạm rồi được tổng hợp thành các giá trị theo ngày và kriged qua các trạm theo từng ngày, nên danh tính trạm rò rỉ qua giai đoạn theo giờ chính xác như ở Mục 5.2. LOSO theo ngày trung thực, thu được bằng cách huấn luyện lại một mô hình tươi cho mỗi trạm giữ lại, chỉ là 0,168 cho chỉ mô hình và 0,224 với kriging — nghĩa là con số 0,8684 bị phóng đại khoảng +0,66. Ví dụ này được giữ lại trong chương như một minh họa cụ thể về việc một quy trình rò rỉ có thể tạo ra một con số cấp-công-bố dễ dàng đến mức nào, và như lý do mọi con số nổi bật trong luận văn này được báo cáo dưới LOSO huấn-luyện-lại-theo-fold tường minh.

Chuyển sang các cấu hình LOSO toàn cục đúng nghĩa, vai trò của phân nhóm tầng là quyết định. Mô hình toàn cục mù-tầng thực sự không phân nhóm đạt một R²_hourly gộp có trọng số-số-giờ chỉ 0,0387, một trung bình theo trạm 0,0273, một trung vị theo trạm −0,0012, và chỉ 48,6% số trạm dương — tức là, một mô hình gộp duy nhất không biết chế độ ô nhiễm của mỗi trạm về cơ bản là vô dụng cho dự đoán không gian, thất bại tại hơn một nửa tất cả các trạm. Việc tiêm nhãn tầng thực (cấu hình T4F, giống hệt biến thể được tinh chỉnh trong definitive_v3) nâng R² gộp lên 0,2004, trung bình lên 0,1989, trung vị lên 0,1787, và tỷ lệ dương lên 70,3%. Đây là cấu hình phi-oracle tốt nhất theo định nghĩa riêng của luận văn và là sự hiện thực hóa neo LOSO khoảng 0,20 đã nêu ở Chương 1. Một kiểm tra chéo độc lập trong một tệp thí nghiệm riêng tái lập sự tương phản: một cấu hình oracle-tier ở đó đạt gộp 0,2074 so với −0,0438 cho đối ứng không-phân-nhóm của nó, xác nhận hiệu ứng không đặc thù cho một tệp.

*Bảng 5.4: Các cấu hình LOSO toàn cục qua các tệp thí nghiệm — không phân nhóm, true-tier (T4F), và oracle base-margin — với R² gộp, trung bình, trung vị, tỷ lệ dương, và trung bình t3.*

| Tệp | Cấu hình | Vai trò | n | Gộp (trọng số n-giờ) | Trung bình | Trung vị | % > 0 | trung bình t3 |
|---|---|---|---|---|---|---|---|---|
| satellite_grouping | no_group | Toàn cục không phân nhóm | 37 | 0,0387 | 0,0273 | −0,0012 | 48,6% | 0,2683 |
| satellite_grouping | t4f (= dart_tuned) | T4F true-tier (phi-oracle tốt nhất) | 37 | 0,2004 | 0,1989 | 0,1787 | 70,3% | 0,5682 |
| twophase_bm | oracle_bm | Oracle base-margin (độ dời trung bình thực) | 40 | 0,2487 | 0,2464 | 0,2669 | 82,5% | 0,5011 |
| twophase_bm | global_bm | Base-margin đơn toàn cục | 40 | −0,0602 | −0,0733 | 0,0261 | 55,0% | 0,2402 |
| tier_operational | oracle_t4f | Oracle true-tier (kiểm tra chéo) | 40 | 0,2074 | 0,1980 | 0,1547 | 75,0% | 0,5666 |
| tier_operational | no_t4f | Không phân nhóm (kiểm tra chéo) | 40 | −0,0438 | −0,0615 | 0,0142 | 52,5% | 0,2552 |

Sáu biến thể DART trong definitive_v3 chỉ khác nhau một cách cận biên với nhau, tất cả đều tụ gần con số nổi bật 0,20: tổ hợp cao nhất ở gộp 0,2061 và trung bình theo trạm 0,2048, biến thể được tinh chỉnh (T4F) theo sau ở 0,2004 gộp và 0,1989 trung bình, và mô hình cơ sở chưa-tinh-chỉnh đứng cuối ở 0,1912 gộp và 0,1902 trung bình. Sự hẹp của độ trải này — dưới 0,02 về trung bình theo trạm qua sáu chiến lược huấn luyện — chỉ ra rằng ngưỡng trần LOSO cho mô hình true-tier toàn cục được đặt bởi thông tin sẵn có, không phải bởi lựa chọn chính quy hóa hay tổ hợp. Việc giới hạn sự chú ý vào tập ba-mươi-bảy-trạm đã-loại-cảm-biến-hỏng thay vì tập bốn-mươi-trạm nâng các trung vị một chút cho các cấu hình oracle, nhưng thứ tự bất biến.

*Bảng 5.5: Sáu biến thể huấn luyện DART trong definitive_v3, tất cả đều tụ gần ngưỡng trần LOSO 0,20, sắp xếp theo R² gộp.*

| Cấu hình | Gộp (trọng số n-giờ) | Trung bình | Trung vị | % > 0 | trung bình t3 (n=9) |
|---|---|---|---|---|---|
| dart_ensemble | 0,2061 | 0,2048 | 0,2030 | 73,0% | 0,5639 |
| dart_tuned (= T4F) | 0,2004 | 0,1989 | 0,1787 | 70,3% | 0,5682 |
| dart_nn23_pruned | 0,1996 | 0,1987 | 0,1881 | 70,3% | 0,5608 |
| dart_nn23 | 0,1971 | 0,1969 | 0,2074 | 73,0% | 0,5622 |
| dart_nn23_dow | 0,1932 | 0,1930 | 0,1929 | 73,0% | 0,5606 |
| dart_base | 0,1912 | 0,1902 | 0,1819 | 70,3% | 0,5548 |

Phân tách theo trạm của cấu hình phi-oracle tốt nhất (T4F, biến thể được tinh chỉnh của definitive_v3) cho thấy rằng con số nổi bật trung bình 0,20 che giấu một dải khổng lồ và một sự phân tầng sạch sẽ theo mức ô nhiễm. Kỹ năng tăng đơn điệu với nồng độ: các trạm ô nhiễm nhất được dự đoán tốt, với trạm tốt nhất — Hưng Yên trên Nguyễn Văn Linh, PM2.5 trung bình 43,2 µg/m³ — đạt R² 0,7553, trong khi các trạm sạch phương sai thấp được dự đoán kém hoặc tệ hơn trung bình riêng của chúng, tệ nhất là Trà Vinh Dân Thành ở trung bình 9,1 µg/m³ và R² −0,8139. Bảng đầy đủ ba-mươi-bảy-trạm dưới đây được sắp xếp theo tầng rồi theo R² trong tầng, làm cho sự phân tầng hiện rõ trực tiếp.

*Bảng 5.6: R² LOSO trung thực theo trạm cho cấu hình phi-oracle tốt nhất (T4F), tất cả ba mươi bảy trạm sắp xếp theo tầng rồi theo R² trong tầng.*

| Tầng | Trạm | Trung bình PM2.5 | R²_hourly |
|---|---|---|---|
| t0 | Trà Vinh xã Dân Thành, TX Duyên Hải | 9,13 | −0,8139 |
| t0 | Quảng Ninh Gần KCN Cái Lân | 7,62 | −0,1553 |
| t0 | Quảng Ninh Nhuệ Hổ – Đông Triều | 9,24 | −0,1185 |
| t0 | Quảng Ninh Km11 – Minh Thành | 9,49 | −0,0537 |
| t0 | Quảng Ninh NM tuyển than Nam Cầu Trắng – Hạ Long | 6,67 | −0,0389 |
| t0 | Quảng Ninh TT văn hóa thể thao Cẩm Phả – Cẩm Trung | 6,96 | −0,0232 |
| t0 | Quảng Ninh Phường Cẩm Thịnh – Cẩm Phả | 6,85 | 0,1787 |
| t1 | Bình Định Hoa Lư – TP. Quy Nhơn | 18,48 | −0,5356 |
| t1 | Vũng Tàu Huyền Trân Công Chúa – Phường 8 | 14,67 | −0,3118 |
| t1 | Quảng Ninh UBND TP Uông Bí | 10,64 | −0,1273 |
| t1 | Thái Bình xã Thái Thọ, huyện Thái Thụy | 15,48 | −0,0567 |
| t1 | Ninh Thuận Công viên – TP Phan Rang | 15,49 | 0,0572 |
| t1 | Đà Nẵng 41 đường Lê Duẩn | 13,19 | 0,0617 |
| t1 | Gia Lai KCN Trà Đa – Tp Pleiku | 10,92 | 0,0696 |
| t1 | Lâm Đồng Vườn hoa – TP Đà Lạt | 18,08 | 0,1197 |
| t1 | Tây Ninh Thị xã Trảng Bàng | 10,92 | 0,1490 |
| t1 | Bình Định huyện Tuy Phước | 12,06 | 0,1883 |
| t2 | Bắc Ninh UBND xã Cao Đức – Gia Bình | 25,40 | −0,0552 |
| t2 | Quảng Ngãi UBND P. Nguyễn Nghiêm | 27,41 | 0,0228 |
| t2 | Quảng Nam KDC Hồ Xuân Hương | 20,07 | 0,2646 |
| t2 | Đà Nẵng ĐH Sư phạm Đà Nẵng | 23,60 | 0,2843 |
| t2 | Bắc Ninh Khu liên cơ Thuận Thành | 27,95 | 0,2866 |
| t2 | Bình Dương 593 Đại lộ Bình Dương | 24,22 | 0,2966 |
| t2 | Long An UBND Tp Tân An | 21,17 | 0,3692 |
| t2 | Phú Thọ đường Hùng Vương – Việt Trì | 27,10 | 0,4924 |
| t2 | Bắc Ninh TT Quan trắc – Bắc Ninh | 23,44 | 0,5071 |
| t2 | HCM Lê Hữu Kiều – Quận 2 | 22,60 | 0,5753 |
| t2 | HCM 20 Lý Chính Thắng | 21,34 | 0,6121 |
| t3 | Bắc Ninh UBND xã Xuân Lâm – Thuận Thành | 52,19 | 0,0696 |
| t3 | Thái Nguyên SVĐ Gang thép | 55,22 | 0,4361 |
| t3 | Thái Bình Cầu Thái Bình | 37,19 | 0,5197 |
| t3 | Hà Nội 556 Nguyễn Văn Cừ | 48,54 | 0,6153 |
| t3 | Hà Nội Công viên Nhân Chính | 36,98 | 0,6311 |
| t3 | Hà Nội ĐHBK Giải Phóng | 46,78 | 0,6722 |
| t3 | Hà Nam Công viên Nam Cao – Phủ Lý | 39,20 | 0,7036 |
| t3 | Hải Dương UBND TP. Hải Dương | 37,63 | 0,7110 |
| t3 | Hưng Yên 437 Nguyễn Văn Linh | 43,19 | 0,7553 |

Kỹ năng LOSO theo trạm được ánh xạ về mặt địa lý trong Hình 5.1, với ba mươi bảy trạm được vẽ trên bản đồ nền Việt Nam và tô màu theo R².

![Hình 5.1: R² bỏ-một-trạm-ra theo trạm trên 37 trạm quan trắc, tô màu theo kỹ năng.](figures/fig_5_3_station_r2_map.png)

*Hình 5.1: R² bỏ-một-trạm-ra theo trạm trên 37 trạm quan trắc, tô màu theo kỹ năng.*

Mẫu không gian có thể được đọc trực tiếp từ bảng theo trạm, bởi vì kỹ năng bám theo mức ô nhiễm và mức ô nhiễm có một chữ ký địa lý mạnh. Các trạm kỹ-năng-cao tụ chặt ở đồng bằng sông Hồng phía Bắc — Hưng Yên, Hà Nam, Hải Dương, ba vị trí Hà Nội, Thái Bình, Thái Nguyên, và Bắc Ninh đều vượt R² 0,4 — tạo thành một vùng hiệu-suất-cao liền mạch quanh Hà Nội. Một túi kỹ năng cao thứ hai thì cô lập và phía Nam: hai trạm Thành phố Hồ Chí Minh đạt các giá trị R² 0,61 và 0,58, đứng tách biệt khỏi các vị trí phía Nam kỹ-năng-thấp xung quanh. Phần còn lại của bản đồ sẽ hiện ra chủ yếu trong dải thấp và âm: cụm ven-biển-công-nghiệp Quảng Ninh ở phía đông bắc, các trạm ven biển miền Trung từ Đà Nẵng xuống Bình Định, và các vị trí đồng bằng sông Cửu Long sạch đều rơi gần hoặc dưới không. Do đó ấn tượng thị giác sẽ là một đồng bằng phía Bắc sáng, liền mạch, hai điểm đô thị phía Nam sáng nhưng cô lập, và một vành đai mờ nhạt — một địa lý mà, như Mục 5.4 lập luận, được đọc một cách tiết kiệm nhất như một địa lý của mức ô nhiễm, mà vùng miền bị lẫn lộn một phần với nó trong mạng lưới này.

Sự sụp đổ LOSO này đặt luận văn vững chắc trong tài liệu khoa học quốc tế về đánh giá không gian. Kawano et al. (2025), với khoảng một nghìn trạm huấn luyện trên khắp Ấn Độ, đã thấy R² của họ rơi từ 0,85 dưới đánh giá chéo ngẫu nhiên xuống 0,67 dưới đánh giá chéo không gian — một mức giảm tương đối nhẹ 0,18, được duy trì bởi mật độ mạng lưới. Meyer et al. (2018) báo cáo một mức rơi từ 0,90 xuống 0,24, và Ploton et al. (2020) một mức rơi từ 0,53 xuống 0,14 — các sụp đổ lần lượt là 0,66 và 0,39, cùng bậc với mức giảm 0,60 được ghi nhận ở đây. Khoảng cách của Việt Nam (0,81 xuống 0,21) lớn hơn nhiều của Ấn Độ vì mạng lưới của Việt Nam thưa hơn nhiều; do đó kết quả luận văn nhất quán với phê phán trong tài liệu khoa học và mở rộng nó đến một bối cảnh nhiệt đới, mạng-lưới-thưa nơi hình phạt không gian ở mức nghiêm trọng nhất.


## 5.4 Kết quả phân tầng theo mức ô nhiễm

Bảng theo trạm của Mục 5.3 đã gợi ý phát hiện diễn giải quan trọng nhất của chương: kỹ năng được tổ chức theo mức ô nhiễm. Mục này làm cho sự tổ chức đó định lượng bằng cách phân tách kết quả LOSO true-tier theo tầng. Một làm rõ về nguồn phản chiếu mục trước: tệp biến-thể-mô-hình definitive_v3 không chứa một cấu hình mang tên đúng nghĩa "oracle T4F," nhưng mỗi trong sáu biến thể DART của nó mang nhãn bốn-tầng thực và do đó thực chất là một mô hình true-tier (oracle-tier); phân tách theo tầng dưới đây được báo cáo cho biến thể cơ sở, vốn khớp gần nhất với các con số tầng đã nêu và mức tăng +0,21, với biến thể tổ hợp tốt nhất tổng thể được đưa ra bên cạnh nó. So sánh oracle-tier-so-với-không-phân-nhóm mang tên đúng nghĩa nằm trong tệp vận hành ghép đôi và được dùng làm kiểm tra chéo cho phép tính mức tăng.

Mẫu theo tầng rõ ràng và đơn điệu. Tầng ô nhiễm cao nhất t3, gồm chín trạm với PM2.5 trung bình 44,1 µg/m³, đạt R² trung bình 0,5548 (trung vị 0,6084). Tầng trung bình t2 — mười một trạm, trung bình 24,0 µg/m³ — rơi xuống R² trung bình 0,3154 (trung vị 0,2953). Tầng thấp t1 — mười trạm, trung bình 14,0 µg/m³ — tụt xuống R² trung bình −0,0302 (trung vị 0,0438), về cơ bản là đường không-kỹ-năng. Và tầng sạch t0 — bảy trạm, trung bình 8,0 µg/m³ — âm một cách có ý nghĩa trên trung bình ở −0,1604 (trung vị −0,0534), phản ánh rằng tại các vị trí sạch nhất mô hình không thể vượt qua trung bình riêng của trạm. Do đó dạng viết tắt đã nêu t3 ≈ 0,56, t2 ≈ 0,31, t1 ≈ 0, t0 ≈ 0 được xác nhận trên trung bình cho t3 và t2 và về mặt hướng cho t1 và t0, với hiệu chỉnh nhỏ rằng trung bình t0 dưới không một cách đáng kể chứ không phải tại không — trung vị của nó, −0,053, là bản tóm tắt khoan dung hơn nhưng vẫn dưới-không.

Bởi vì mỗi tầng chỉ giữ một số ít trạm, các trung bình theo tầng này mang độ bất định lấy mẫu đáng kể và được báo cáo kèm nó. Trên biến thể cơ sở các sai số chuẩn của R² trung bình theo tầng là khoảng 0,13 (t0, n = 7), 0,065 (t1, n = 10), 0,066 (t2, n = 11), và 0,078 (t3, n = 9), tương ứng với các nửa-độ-rộng khoảng tin cậy 95% khoảng 0,15–0,18 mỗi tầng (đối với t3, n = 9, SD ≈ 0,234, nửa-độ-rộng CI ≈ 0,15). Hai trong ba khoảng cách tầng-liền-kề có kích thước tương đương với độ bất định lấy mẫu này: khoảng cách t0→t1 (khoảng +0,13) nhỏ hơn độ bất định kết hợp của hai trung bình, và ngay cả khoảng cách t2→t3 (khoảng +0,24) cũng chỉ lớn hơn nó một cách khiêm tốn; khoảng cách t1→t2 (khoảng +0,35) là chuyển tiếp duy nhất vượt sàn nhiễu một cách dứt khoát. Thứ tự đơn điệu nhất quán qua tất cả sáu biến thể, nên gradient như một tổng thể không phải một hiện vật của lấy mẫu, nhưng các gia số tầng-sang-tầng riêng lẻ — và các trung bình tầng chính xác — nên được đọc như các ước tính điểm được bao quanh bởi các khoảng cỡ ±0,15 chứ không phải như các mức được tách biệt sắc nét.

*Bảng 5.7: Kết quả LOSO trung thực phân tầng cho biến thể DART cơ sở — số trạm theo tầng, trung bình PM2.5, R², RMSE, MAE, và bias.*

| Tầng | n trạm | Trung bình PM2.5 | Trung bình R²_hourly | Trung vị R²_hourly | Trung bình RMSE | Trung bình MAE | Trung bình bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | 7 | 7,99 | −0,1604 | −0,0534 | 8,92 | 5,52 | −1,75 |
| t1 | 10 | 13,99 | −0,0302 | 0,0438 | 13,45 | 8,11 | −3,35 |
| t2 | 11 | 24,03 | 0,3154 | 0,2953 | 14,83 | 9,85 | −2,74 |
| t3 | 9 | 44,10 | 0,5548 | 0,6084 | 23,31 | 15,20 | −4,17 |
| TẤT CẢ | 37 | — | 0,1902 | — | — | — | — |

Biến thể tổ hợp tốt nhất tổng thể tạo ra cùng cấu trúc tầng, cao hơn một cách cận biên ở mỗi mức: t3 ở 0,5639, t2 ở 0,3324, t1 ở −0,0195, và t0 ở −0,1371. Tính bất biến của mẫu qua các chiến lược huấn luyện xác nhận rằng gradient tầng là một thuộc tính của dữ liệu, không phải của mô hình.

*Bảng 5.8: Kết quả LOSO trung thực phân tầng cho biến thể tổ hợp tốt nhất tổng thể, cho thấy cùng cấu trúc đơn điệu cao hơn một cách cận biên ở mỗi tầng.*

| Tầng | n trạm | Trung bình PM2.5 | Trung bình R²_hourly | Trung vị R²_hourly | Trung bình RMSE | Trung bình MAE | Trung bình bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| t0 | 7 | 7,99 | −0,1371 | −0,0685 | 8,84 | 5,40 | −1,83 |
| t1 | 10 | 13,99 | −0,0195 | 0,0429 | 13,38 | 8,09 | −3,26 |
| t2 | 11 | 24,03 | 0,3324 | 0,3106 | 14,62 | 9,74 | −2,59 |
| t3 | 9 | 44,10 | 0,5639 | 0,6120 | 23,13 | 15,13 | −3,51 |
| TẤT CẢ | 37 | — | 0,2048 | — | — | — | — |

Gradient tầng này là phát hiện định lượng đơn lẻ quan trọng nhất của chương, và nó là bằng chứng chính cho khẳng định rằng khả năng dự đoán tỷ lệ với mức ô nhiễm. Bản thân gradient bền vững: nó đơn điệu (t3 > t2 > t1 > t0) và bất biến-mô-hình, được tái lập qua tất cả sáu biến thể huấn luyện, nên việc kỹ năng tăng theo nồng độ đứng trên nền tảng vững chắc. Khẳng định đi kèm rằng mức, chứ không phải vùng, là biến vận hành thì đứng trên nền tảng yếu hơn và được đưa ra ở đây như một giả thuyết chứ không phải một kết quả đã xác lập, bởi vì mức và vùng bị lẫn lộn ở đỉnh mạng lưới. Dàn ý ban đầu đề xuất chứng minh khẳng định mức-trên-vùng bằng cách cho thấy tầng đỉnh t3 chứa cả những trạm hiệu suất cao phía Bắc và phía Nam. Trong bộ dữ liệu này nó không như vậy: cả chín trạm t3 đều ở phía Bắc, trong và quanh đồng bằng sông Hồng, bởi vì không có trạm phía Nam hay miền Trung nào đạt ngưỡng t3 35 µg/m³ — miền Nam đơn giản là không đủ ô nhiễm để lấp đầy tầng đỉnh. Nơi duy nhất hai vùng cùng tồn tại ở nồng độ tương đương là một tầng xuống dưới, ở t2. Ở đó hai trạm được dự đoán tốt nhất là các vị trí Thành phố Hồ Chí Minh phía Nam, ở các giá trị R² 0,59 và 0,56 trên biến thể cơ sở, vượt điểm phần lớn các trạm t2 phía Bắc, và R² trung bình t2 phía Nam (0,459, n = 4) vượt trung bình t2 phía Bắc (0,278, n = 4). Tuy nhiên, sự khác biệt này không có ý nghĩa thống kê: một kiểm định Welch hai-mẫu cho p = 0,30, một kiểm định Mann–Whitney U cho p = 0,34, và khoảng tin cậy 95% trên sự khác biệt trung bình Nam-trừ-Bắc trải khoảng [−0,13, +0,49] — một khoảng dễ dàng bao gồm không. Với chỉ bốn trạm mỗi vùng ở t2, so sánh đơn giản thiếu sức mạnh để tách một hiệu ứng mức khỏi một hiệu ứng vùng, và không thể có kiểm định có ý nghĩa nào ở cỡ mẫu này. Khi các trung bình cấp-vùng được tính qua tất cả các tầng chúng có vẻ thiên về phía Bắc (bắc 0,266, nam 0,134, trung 0,057), nhưng gradient vùng biểu kiến đó phần lớn là một hiện vật của hỗn hợp tầng — phía Bắc giữ tất cả các trạm t3 — và co lại một khi tầng được giữ cố định. Do đó cách đọc trung thực là mức và vùng bị lẫn lộn trong mạng lưới này: gradient mức mạnh và bất biến-mô-hình, nhưng dữ liệu sẵn có không thể chứng minh rằng mức chứ không phải vùng là biến chi phối, và bằng chứng t2 mang tính gợi ý chứ không phải kết luận.

*Bảng 5.9: Mười một trạm t2 theo vùng, PM2.5, và R², cho thấy so sánh Bắc–Nam ở mức ô nhiễm tương đương.*

| Trạm | Vùng | PM2.5 | R²_hourly |
|---|---|---:|---:|
| HCM Lý Chính Thắng | Nam | 21,34 | 0,5947 |
| HCM Lê Hữu Kiều Q2 | Nam | 22,60 | 0,5610 |
| Bắc Ninh Suối Hoa | Bắc | 23,44 | 0,5042 |
| Phú Thọ Việt Trì | Bắc | 27,10 | 0,4578 |
| Long An Tân An | Nam | 21,17 | 0,3844 |
| Bình Dương Hiệp Thành | Nam | 24,22 | 0,2953 |
| Quảng Nam | Trung | 20,07 | 0,2582 |
| Bắc Ninh Thuận Thành | Bắc | 27,95 | 0,2551 |
| Đà Nẵng ĐHSP | Trung | 23,60 | 0,2432 |
| Quảng Ngãi | Trung | 27,41 | 0,0221 |
| Bắc Ninh Cao Đức Gia Bình | Bắc | 25,40 | −0,1062 |

Cuối cùng, bản thân kỹ thuật phân tầng là đòn bẩy hiệu suất đơn lẻ lớn nhất được phát hiện trong luận văn này. Thước đo có thể bảo vệ nhất của mức tăng lấy hiệu số mô hình true-tier so với đường cơ sở không-phân-nhóm thực sự mù-tầng trong cùng một tệp, sao cho cả hai chia sẻ một tập trạm giống hệt: ở đó trung bình theo trạm T4F 0,1989 vượt trung bình không-phân-nhóm cùng-tệp 0,0273 khoảng +0,17, đưa mô hình từ về cơ bản không có kỹ năng không gian hữu dụng lên con số nổi bật 0,20. Các cách diễn đạt liên-tệp ghép biến thể cơ sở (trung bình 0,1902) với một đường cơ sở không-phân-nhóm lấy từ một tệp khác (trung bình −0,0286) báo cáo một mức +0,219 lớn hơn, và cấu hình oracle-tier mang tên đúng nghĩa cho lên đến +0,260 trên toàn bộ tập bốn-mươi-trạm; nhưng các con số này trộn lẫn các tập trạm và nên được đọc như đầu trên của dải chứ không phải con số nổi bật. Qua tất cả các cách diễn đạt mức tăng trải khoảng +0,17 (trong-cùng-tệp, so với đường cơ sở không-phân-nhóm cùng-tệp 0,027) đến +0,26, có thể bảo vệ nhất là khoảng +0,17 — và trên mọi cách diễn đạt nó lớn hơn bất kỳ nhóm đặc trưng nào, bất kỳ lựa chọn siêu tham số nào, hay bất kỳ chiến lược tổ hợp nào được khảo sát, xác nhận khẳng định ở Chương 3 rằng T4F là kỹ thuật có tác động lớn nhất trong nghiên cứu. Cái giá của nó, phụ thuộc vòng tròn của việc cần biết mục tiêu để gán tầng, là chủ đề của Mục 5.5.

*Bảng 5.10: Mức tăng do phân tầng qua các biến thể và tập trạm, mỗi mục lấy hiệu số một trung bình true-tier so với một đường cơ sở không-phân-nhóm.*

| So sánh | Trung bình T4F | Trung bình không phân nhóm | Mức tăng |
|---|---:|---:|---:|
| dart_base (37) vs no_t4f (37) | 0,1902 | −0,0286 | +0,219 |
| dart_ensemble (37) vs no_t4f (37) | 0,2048 | −0,0286 | +0,233 |
| oracle_t4f (37) vs no_t4f (37) | 0,2067 | −0,0286 | +0,235 |
| oracle_t4f (40) vs no_t4f (40) | 0,1980 | −0,0615 | +0,260 |


## 5.5 So sánh các mô hình triển khai được

Mức tăng +0,21 của Mục 5.4 được mua bằng thông tin oracle: việc gán một trạm vào tầng thực của nó đòi hỏi biết trung bình PM2.5 của trạm, vốn là đại lượng mà mô hình được kỳ vọng dự đoán. Một mô hình thực sự triển khai được — một mô hình dùng được tại một vị trí không có lịch sử mặt đất — không được dùng tầng thực hay trung bình thực của trạm giữ lại. Mục này so sánh mọi cấu hình triển khai được phát triển trong luận văn với ngưỡng trần oracle và đo khoảng cách giữa chúng, vốn là chi phí thông tin của việc không biết mức ô nhiễm của một trạm.

Hai tiền đề trong dàn ý ban đầu phải được hiệu chỉnh trước khi so sánh có thể được đọc một cách trung thực, cả hai đều xoay quanh điều gì được tính là triển khai được. Thứ nhất, cấu hình mà dàn ý gán nhãn là phương pháp triển khai tốt nhất — một mô hình cổng-mềm được tăng cường với một biên cơ sở theo trạm thực — thực ra là một oracle: việc kiểm tra mã tạo ra nó cho thấy nó thêm trung bình PM2.5 theo tháng thực riêng của trạm giữ lại làm biên cơ sở, vốn chính là rò rỉ mà một mô hình triển khai được phải tránh. Do đó các con số mạnh của nó (trung bình theo trạm 0,270, 92,5% dương) là ngưỡng trần oracle, không phải một kết quả triển khai được, và chúng được báo cáo dưới đây như vậy. Thứ hai, dàn ý gọi biến thể cơ sở definitive_v3 là ngưỡng trần "oracle base-margin"; thực ra cấu hình đó chạy ở chế độ không-oracle tường minh với một biên cơ sở toàn cục duy nhất và chỉ dùng tầng thực của trạm giữ lại (T4F), khiến nó là một cấu hình oracle-tier, không phải một cấu hình oracle trung-bình-theo-trạm. Các con số của nó (trung bình 0,190, 70,3% dương) nằm khá dưới ngưỡng trần trung-bình-theo-trạm thực, đúng như người ta kỳ vọng từ oracle cấp-tầng nhẹ hơn. Với các hiệu chỉnh này bức tranh trở nên mạch lạc.

Các cấu hình thực sự triển khai được — những cấu hình chỉ dùng các đặc trưng vệ tinh, khí tượng, và sử dụng đất quan sát được, với nhiều nhất là một biên cơ sở toàn cục — hội tụ trong một dải hẹp ngay trên không. Các trung vị theo trạm của chúng dao động khoảng +0,01 đến +0,04 và các trung bình theo trạm của chúng lượn quanh không, giữa −0,06 và +0,04, với chỉ khoảng 47–57% số trạm dương. Con số "+0,04 trung vị" đã nêu là đầu lạc quan của dải này và được đạt bởi các biến thể tốt hơn: mô hình tổn-thất-cân-bằng bất biến đạt một trung vị 0,0415 và mô hình vệ tinh LOSO trực tiếp với các trạm lân cận không gian đạt 0,0428, trong khi các biến thể cluster-base-margin và không-phân-nhóm nằm thấp hơn ở +0,01 đến +0,02. Tổ hợp chuyên gia cổng-mềm đạt một trung bình theo trạm 0,0447 — con số được trích ở Chương 3 — nhưng trung vị của nó thực ra âm nhẹ ở −0,0187, nên đóng góp của nó cho "sự hội tụ +0,04" là qua trung bình của nó chứ không phải trung vị. Do đó phát biểu trung thực là ba hướng tiếp cận triển khai được độc lập hội tụ trong một dải dương gần-không mà mép trên của nó là khoảng +0,04, không phải rằng tất cả chúng đáp xuống chính xác một giá trị duy nhất.

Ngưỡng trần oracle, ngược lại, cao hơn nhiều. Việc tiêm trung bình theo trạm thực của trạm giữ lại như một biên cơ sở cộng tính nâng R² trung bình theo trạm lên giữa +0,25 và +0,31 — +0,306 trên tập ba-mươi-bảy-trạm đã-loại-cảm-biến-hỏng, +0,254 đến +0,271 trên các tập bốn-mươi-trạm — với 85% đến 92% số trạm dương. Điều này xác nhận ngưỡng trần oracle đã nêu khoảng +0,27 trung bình theo trạm với khoảng 92% dương. Cấu hình cổng-mềm-cộng-trung-bình-thực bị gán nhãn sai, nay được phân loại đúng là oracle, tái lập ngưỡng trần một cách độc lập ở đúng +0,270 trung bình và 92,5% dương.

*Bảng 5.11: Các cấu hình triển khai được so với các ngưỡng trần oracle — R² trung bình, trung vị, tỷ lệ dương, và trung bình t3, với mỗi cấu hình được gắn cờ là triển khai được hay oracle.*

| Cấu hình | Nguồn | Triển khai được | Trung bình R² | Trung vị R² | % > 0 | trung bình t3 | n |
|---|---|---|---:|---:|---:|---:|---:|
| soft_gate_moe | soft_gate_moe | có | +0,0447 | −0,0187 | 47,5% | +0,4129 | 40 |
| no_t4f | irm_invariant | có | −0,0615 | +0,0142 | 52,5% | +0,2552 | 40 |
| balanced | irm_invariant | có | −0,0508 | +0,0415 | 57,5% | +0,2445 | 40 |
| direct | satellite_v5b | có | −0,0178 | +0,0428 | 55,0% | +0,3094 | 40 |
| clust_bm_rfsi | satellite_v5g | có | +0,0292 | +0,0189 | 54,1% | +0,4249 | 37 |
| clust_bm | satellite_v5g | có | −0,0067 | +0,0117 | 51,4% | +0,2413 | 37 |
| dart_base (T4F) | definitive_v3 | không (oracle tầng) | +0,1902 | +0,1819 | 70,3% | +0,5548 | 37 |
| oracle_real_bm | irm_invariant | không (oracle trung bình) | +0,2429 | +0,1850 | 85,0% | +0,6031 | 40 |
| sgm_real_bm | irm_invariant | không (oracle trung bình) | +0,2701 | +0,2377 | 92,5% | +0,6038 | 40 |
| oracle_bm | satellite_v5g | không (oracle trung bình, ngưỡng trần) | +0,3063 | +0,2910 | 91,9% | +0,5694 | 37 |

Khoảng cách giữa dải triển khai được và ngưỡng trần oracle là chi phí thông tin của việc không biết trung bình trạm, và nó lớn. Được đo trên các fold tương ứng trong một tệp duy nhất — cách duy nhất sạch về phương pháp luận để lấy hiệu số hai, vì các cấu hình chia sẻ một tập trạm ở đó — biên cơ sở oracle vượt mô hình cluster-base-margin triển khai được tốt nhất +0,277 về R² trung bình theo trạm, +0,272 về trung vị, và +37,8 điểm phần trăm về tỷ lệ trạm dương. Cùng so sánh trong tệp invariant, lấy hiệu số cấu hình oracle trung-bình-thực so với đường cơ sở không-phân-nhóm triển khai được, cho +0,332 về trung bình, +0,224 về trung vị, và +40,0 điểm phần trăm dương. Theo cách làm tròn, việc biết trung bình thực của mỗi trạm trị giá khoảng +0,28 đến +0,33 về R² trung bình theo trạm và khoảng bốn mươi điểm phần trăm trạm dùng được bổ sung.

*Bảng 5.12: Chi phí thông tin của việc không biết trung bình trạm — các khoảng cách oracle-trừ-triển-khai về R² trung bình, R² trung vị, và tỷ lệ dương, được đo trên các tập trạm tương ứng.*

| Khoảng cách (oracle − triển khai) | Tập tương ứng | Khoảng cách trung bình R² | Khoảng cách trung vị R² | Khoảng cách % > 0 |
|---|---|---:|---:|---:|
| oracle_bm vs clust_bm_rfsi | satellite_v5g, 37 trạm | +0,2771 | +0,2721 | +37,8 điểm |
| sgm_real_bm vs no_t4f | irm_invariant, 40 trạm | +0,3316 | +0,2235 | +40,0 điểm |

Toàn bộ chuỗi ba-con-số của luận văn giờ đã hiện ra. Mô hình đạt R² ≈ 0,81 dưới đánh giá chéo ngẫu nhiên, R² ≈ 0,20 dưới LOSO true-tier trung thực, và một R² trung vị theo trạm triển khai được ≈ 0,04 một khi tầng oracle bị rút đi. Bước thứ nhất, từ 0,81 xuống 0,20, là chi phí của rò rỉ danh-tính-trạm (Mục 5.3); bước thứ hai, từ 0,20 xuống 0,04, là chi phí của phụ thuộc vòng tròn — khoảng cách giữa việc biết và không biết chế độ ô nhiễm của mỗi trạm. Khoảng cách triển-khai-đến-oracle khoảng +0,27 trung bình theo trạm là cái giá chính xác của thông tin bị thiếu đó, và sự hội tụ của ba hướng tiếp cận triển khai được độc lập ở đáy của nó là bằng chứng mạnh rằng khoảng +0,04 trung vị là ngưỡng trần triển khai thực sự chỉ từ các biến quan sát được từ vệ tinh, khi không có dữ liệu hiệu chuẩn cục bộ. Đáng chú ý là ngay cả các mô hình triển khai được cũng hoạt động tốt hơn nhiều tại tầng ô nhiễm t3 (trung bình t3 từ +0,24 đến +0,42) so với tổng thể, phản chiếu gradient tầng của Mục 5.4; các trung vị tổng thể gần-không bị kéo xuống bởi các trạm tầng-thấp sạch, không phải bởi thất bại đồng đều ở khắp nơi.

Tuy nhiên, ngưỡng trần +0,04 đó cụ thể là ngưỡng trần cho việc phục hồi đường cơ sở trạm *từ các biến quan sát được từ vệ tinh*. Nó không phải ngưỡng trần cho một mô hình được phép nội suy đường cơ sở từ mạng lưới quan trắc xung quanh. Quy trình định tuyến tiên nghiệm không gian của Mục 3.6.4 làm chính xác điều đó, và nó thay đổi bức tranh một cách đáng kể. Thay vì ước tính tầng từ các biến quan sát được — chiến lược tạo ra dải +0,04 và thất bại với bảy họ của Mục 5.7 — nó ước tính mức nền của mỗi trạm giữ lại như một trung bình có trọng số khoảng cách của các trung bình *quan trắc* của các trạm huấn luyện xung quanh (bản thân trạm luôn bị loại trừ), rồi dịch chuyển và định tuyến các luồng dự đoán không-mục-tiêu về phía ước tính đó. Được đánh giá dưới cùng giao thức bỏ-một-trạm-ra trên bốn mươi trạm, với GHAP được loại bỏ nên kết quả không phụ thuộc vào sản phẩm PM2.5 khí hậu nào, quy trình triển khai được này đạt R² trung bình theo trạm 0,197 và trung vị 0,117 — trong vòng 0,01 và 0,03 của ngưỡng trần T4F oracle 0,203 và 0,147 — với R² gộp 0,569 so với 0,572 của oracle, ba mươi trong bốn mươi trạm dương, và, quan trọng là, không có chuyển trạng thái lớp cao-thành-không-cao và không có chuyển trạng thái thấp-thành-cao nguy hiểm nào, an toàn hơn một cách cận biên so với chính oracle. Tóm lại, một mô hình hoàn toàn triển khai được phục hồi gần như toàn bộ mức tăng tầng oracle mà các đại diện quan sát được từ vệ tinh không thể.

*Bảng 5.13: Mô hình định tuyến tiên nghiệm không gian so với ngưỡng trần oracle và dải triển khai dựa-trên-biến-quan-sát-từ-vệ-tinh, tất cả dưới bỏ-một-trạm-ra không-GHAP. Bộ ước tính đường-cơ-sở-không-gian triển khai được khép gần như toàn bộ khoảng cách oracle mà ước tính tầng từ biến quan sát không thể.*

| Cấu hình | Triển khai được | R² gộp | R² trạm trung bình | R² trạm trung vị | Cao→không-cao | Nguy hiểm thấp→cao |
|---|---|---:|---:|---:|---:|---:|
| oracle_t4f (tầng thực, ngưỡng trần) | không | 0,572 | 0,203 | 0,147 | 1 | 0 |
| chuyên gia MoE theo tầng (ý tưởng đúng nghĩa) | có | 0,434 | 0,083 | 0,081 | 6 | 1 |
| dải biến-quan-sát-từ-vệ-tinh (§3.6.1–3.6.3) | có | — | ≈ +0,00 đến +0,04 | ≈ +0,01 đến +0,04 | — | — |
| **định tuyến tiên nghiệm không gian (§3.6.4, triển khai)** | **có** | **0,569** | **0,197** | **0,117** | **0** | **0** |

Cơ chế của sự khác biệt đáng được nêu chính xác, bởi vì nó dễ bị đọc nhầm như một mâu thuẫn với ngưỡng trần +0,04 nhưng thực ra không phải. Các cấu hình triển khai được của dải ở trên đã sử dụng các đặc trưng trạm-lân-cận không gian (RFSI): sự khác biệt không phải là mô hình mới có quyền truy cập các trạm lân cận và các mô hình khác thì không. Sự khác biệt là *thông tin trạm lân cận được dùng để làm gì*. Các cấu hình trước đó dùng các trạm lân cận như các đặc trưng đồng-biến-thiên theo giờ, vốn cố định *hình dạng* thời gian của chuỗi nhưng để *mức đường cơ sở* tuyệt đối không được neo — thất bại kinh điển trong đó chuỗi dự đoán của một trạm giữ lại có mẫu đúng nhưng trung bình sai. Tiên nghiệm không gian dùng các trạm lân cận để neo mức đường cơ sở đó trực tiếp, như một vật thay thế triển khai được cho biên cơ sở oracle, cung cấp đại lượng duy nhất mà các biến quan sát được từ vệ tinh không thể. Thiết kế hai-pha trước đó (Mục 3.6.2) đã thử cùng việc neo nhưng ước tính trung bình từ các đặc trưng vệ tinh (R² bỏ-một-ra ≈ 0,59) và tiêm nó như một biên cơ sở cứng, vốn vượt quá tại các vị trí sạch; ước tính không gian vừa chính xác hơn vừa được dùng thận trọng hơn, đó là lý do nó thành công ở nơi thiết kế hai-pha thất bại. Lưu ý thiết yếu theo sau từ cơ chế: bởi vì ước tính là một nội suy không gian, kỹ năng của nó phụ thuộc vào sự gần kề của các trạm nội suy, và nó suy giảm khi mục tiêu di chuyển ra xa mạng lưới — chính là ràng buộc mật độ của Chương 4 được nêu lại, và được kiểm tra trực tiếp trên các trạm chưa thấy ở Mục 5.6.


## 5.6 Đánh giá độc lập (mạng lưới LCS)

Các khung đánh giá của Mục 5.2 đến 5.5 đều là nội bộ, dùng ba mươi bảy trạm quan trắc tự động vừa để huấn luyện vừa, dưới LOSO, để kiểm tra. Phép kiểm tra nghiêm ngặt nhất là độc lập: dự đoán tại các vị trí thực sự độc lập chưa bao giờ tham gia vào quá trình phát triển mô hình. Mục này báo cáo hai phép kiểm tra như vậy của mô hình vùng đồng bằng sông Hồng — bỏ-một-trạm-ra trong mười hai trạm đồng bằng dày đặc, và đánh giá độc lập so với ba mươi chín cảm biến chi phí thấp và trạm tham chiếu Đại sứ quán Hoa Kỳ.

LOSO đồng bằng xác nhận hai điều đã được xác lập ở quy mô quốc gia: nội suy không gian giúp ích, và một đường cơ sở oracle để lại khoảng dư rõ ràng. Trong số ba cấu hình, biến thể nội-suy-không-gian (delta_rfsi) rõ ràng vượt mô hình base-margin thuần (delta_bm), nâng R² trung bình từ 0,2021 lên 0,3022 và trung vị từ 0,2944 lên 0,4330, với cả hai đạt 75,0% số trạm dương. Cấu hình oracle base-margin đặt ngưỡng trần ở trung bình 0,4109, trung vị 0,4728, và 91,7% số trạm dương — xác nhận rằng ngay cả trong mạng lưới con dày đặc, thuận lợi này, việc biết trung bình trạm thực vẫn gia tăng kỹ năng và các mô hình triển khai được chưa làm cạn kiệt hiệu suất có thể đạt được. Trung bình và trung vị delta_rfsi (0,302 và 0,433) là các con số vùng đã nêu ở Chương 3, và chúng vượt đáng kể trung bình 0,255 của cấu hình toàn quốc tương đương trên cùng các trạm, biện minh cho chiến lược mô-hình-vùng đối với các cụm dày đặc.

*Bảng 5.14: LOSO của mô hình đồng bằng trên mười hai trạm đồng bằng sông Hồng dày đặc — các cấu hình base-margin, nội-suy-không-gian, và oracle.*

| Cấu hình | Trung bình R² | Trung vị R² | % dương | Trung bình RMSE | Trung bình MAE |
|---|---:|---:|---:|---:|---:|
| delta_bm | 0,2021 | 0,2944 | 75,0% | 25,93 | 17,05 |
| delta_rfsi | 0,3022 | 0,4330 | 75,0% | 23,34 | 15,25 |
| oracle_bm (ngưỡng trần) | 0,4109 | 0,4728 | 91,7% | 21,30 | 13,92 |

Đánh giá độc lập cảm-biến-chi-phí-thấp là phép kiểm tra độc lập nhất sẵn có, nhưng nó phải được mô tả đặc trưng chính xác, bởi vì nó không phải dự đoán không gian thực sự tại một vị trí chưa thấy. Các cảm biến chi phí thấp giữ lại chỉ cung cấp PM2.5 mục tiêu; các đặc trưng vệ tinh, AOD, TROPOMI, và khí tượng ERA5 cấp cho mô hình được lấy từ bản ghi theo giờ của trạm quan trắc (KK) gần nhất chứ không phải đo tại chính cảm biến, và các điểm neo không gian RFSI được định vị đúng tại tọa độ của cảm biến. Do đó phép kiểm tra là một kiểm tra chuyển đặc trưng từ trạm gần nhất cộng với sự phù hợp cảm biến: nó hỏi mô hình dự đoán tốt đến đâu các nồng độ của một cảm biến độc lập khi các tác nhân khí tượng và vệ tinh của nó được mượn từ trạm quan trắc gần nhất, và thiết bị chi phí thấp đồng thuận tốt đến đâu với dự đoán đó. Cách diễn đạt này quan trọng để diễn giải cả kỹ năng nổi bật lẫn sự phụ thuộc khoảng cách của nó. Tệp đánh giá chứa bốn mươi vị trí giữ lại — ba mươi chín cảm biến chi phí thấp và một trạm tham chiếu Đại sứ quán Hoa Kỳ — và các con số nổi bật đã nêu ở Chương 1 là tập con cảm-biến-chi-phí-thấp: trên ba mươi chín cảm biến R² trung vị là 0,5293, R² trung bình là 0,0310, và 84,6% số vị trí dương. (Việc bao gồm Đại sứ quán dịch các con số này nhẹ thành trung vị 0,5416, trung bình 0,0473, và 85,0% dương; các con số đã nêu là giá trị chỉ-cảm-biến.) Do đó các con số này một phần là kết quả chuyển đặc trưng chứ không phải một thước đo kỹ năng không gian độc lập. Một điều kiện thứ hai áp dụng cụ thể cho mẫu LCS: các bản ghi chi phí thấp chỉ bao phủ cửa sổ tháng-Mười–tháng-Tư (mỗi cảm biến đóng góp khoảng 1.400 giờ trong mùa khô), nên trung vị R² = 0,53 là một con số mùa khô và kỹ năng quanh năm của mô hình tại các vị trí này chưa được kiểm tra. Khoảng cách rộng giữa trung vị mạnh 0,53 và trung bình yếu 0,03 là chữ ký giờ-đã-quen của một số ít vị trí xa, bị suy giảm: trung bình bị kéo xuống gần như hoàn toàn bởi một giá trị ngoại lai duy nhất cách 43 km trạm quan trắc gần nhất với R² −14,75, trong khi trung vị, bền vững trước giá trị ngoại lai đó, trung thực báo cáo rằng vị trí chi phí thấp điển hình được dự đoán tốt.

Đại sứ quán Hoa Kỳ tại Hà Nội, điểm độc lập chất lượng cao nhất — một thiết bị suy giảm beta BAM-1022 chuẩn-tham-chiếu cách trạm huấn luyện gần nhất 3,0 km — được kiểm chứng cao hơn và được báo cáo riêng: R² = 0,6842, RMSE = 17,51 µg/m³, trên 9.729 giờ. Đây là điểm độc lập đáng tin cậy nhất duy nhất trong luận văn, bởi vì thiết bị Đại sứ quán là cấp-tham-chiếu chứ không phải một cảm biến chi phí thấp tán xạ laser; chịu cùng lưu ý chuyển đặc trưng như các vị trí LCS — các tác nhân khí tượng và vệ tinh của nó cũng được mượn từ trạm quan trắc gần nhất — nó cho thấy rằng trong vùng phủ mạng lưới dày đặc các dự đoán của mô hình đồng thuận chặt chẽ với một trạm tham chiếu độc lập. Hai lưu ý làm dịu một so sánh trực tiếp với trung vị LCS. Thứ nhất, bản ghi Đại sứ quán (9.729 giờ) trải dài nhiều mùa, trong khi các bản ghi chi phí thấp bị giới hạn ở mùa khô (~1.400 giờ mỗi cái); do đó hai cái không cùng nền tảng mùa, và R² Đại sứ quán không nên được đọc như một sự nâng cấp như-thể-tương-đương của con số LCS mùa khô. Thứ hai, bản ghi dài hơn bản thân nó không giải thích R² cao hơn: độ dài bản ghi ảnh hưởng đến độ chính xác của ước tính, không phải giá trị kỳ vọng của nó, và lợi thế của Đại sứ quán hợp lý hơn là khoảng cách neo ngắn của nó (3,0 km) và thiết bị cấp-tham-chiếu chứ không phải số giờ của nó. Do đó kết quả Đại sứ quán tốt nhất được đọc như một kiểm tra sự phù hợp chất-lượng-cao, cự-ly-gần, đa-mùa chứ không phải như bằng chứng về kỹ năng không gian độc lập.

*Bảng 5.15: Tóm tắt đánh giá độc lập — tập con cảm-biến-chi-phí-thấp, tất cả các vị trí bao gồm Đại sứ quán, và riêng trạm tham chiếu Đại sứ quán Hoa Kỳ.*

| Tập con | n | Trung vị R² | Trung bình R² | % dương |
|---|---:|---:|---:|---:|
| Chỉ các vị trí LCS | 39 | 0,5293 | 0,0310 | 84,6% |
| Tất cả các vị trí (gồm Đại sứ quán) | 40 | 0,5416 | 0,0473 | 85,0% |
| Đại sứ quán Hoa Kỳ Hà Nội (đơn lẻ) | 1 | 0,6842 | — | RMSE 17,51, n = 9.729 h |

Chi tiết cấp-vị-trí làm cho sự phụ thuộc mật độ cụ thể. Các vị trí chi phí thấp tốt nhất được dự đoán gần tốt bằng Đại sứ quán, với các giá trị R² từ 0,65 đến 0,76, và chúng hầu hết đều gần một trạm neo quan trắc — trong vòng vài kilômét ở trung tâm Hà Nội hoặc ở các thị trấn đồng bằng liền kề. Các vị trí tệ nhất được dự đoán thảm họa, với các giá trị R² từ −0,16 đến −14,75, và chúng chủ yếu ở xa bất kỳ trạm neo nào, ở Hải Phòng và các vị trí Hưng Yên ngoại vi cách trạm quan trắc gần nhất 15 đến 43 km.

*Bảng 5.16: Mười vị trí chi phí thấp được dự đoán tốt nhất theo R², với khoảng cách đến trạm quan trắc gần nhất, độ dài bản ghi, và RMSE.*

| Hạng | R² | Khoảng cách KK (km) | n giờ | RMSE | Trạm |
|---|---:|---:|---:|---:|---|
| 1 | 0,7638 | 15,5 | 1082 | 20,26 | Ninh Bình trạm bơm Hoành Uyển, P. Hà Nam |
| 2 | 0,7468 | 3,7 | 1376 | 18,52 | Hà Nội UBND phường Định Công |
| 3 | 0,7463 | 2,7 | 1094 | 17,52 | Ninh Bình Đảng ủy – HĐND phường Hà Nam |
| 4 | 0,7363 | 1,1 | 1345 | 23,43 | Hà Nội Trường tiểu học Minh Khai, P. Thanh Nhàn |
| 5 | 0,6986 | 2,7 | 1482 | 16,72 | Hà Nội 83 Nguyễn Chí Thanh |
| 6 | 0,6865 | 3,0 | 1504 | 20,86 | Hà Nội Bộ TNMT, 10 Tôn Thất Thuyết |
| 7 | 0,6599 | 3,5 | 1507 | 23,08 | Hà Nội UBND phường Quan Hoa (cũ) |
| 8 | 0,6518 | 2,0 | 1346 | 19,94 | Hà Nội Trường tiểu học Thịnh Hào, P. Ô Chợ Dừa |
| 9 | 0,6488 | 17,1 | 1507 | 20,92 | Hà Nội UBND xã Chuyên Mỹ |
| 10 | 0,6467 | 9,0 | 1366 | 31,61 | Hà Nội Trường mầm non B, Ngọc Hồi |

*Bảng 5.17: Các vị trí chi phí thấp được dự đoán kém nhất, tất cả đều ở xa bất kỳ trạm neo quan trắc nào, với khoảng cách, độ dài bản ghi, và RMSE.*

| R² | Khoảng cách KK (km) | n giờ | RMSE | Trạm |
|---:|---:|---:|---:|---|
| −14,7549 | 43,0 | 1443 | 33,89 | Hải Phòng cột điện P. Bạch Đằng |
| −0,5981 | 15,2 | 1428 | 53,56 | Hưng Yên thôn Lương xã Thượng Hồng |
| −0,3160 | 18,4 | 1462 | 34,92 | Hưng Yên xã Phạm Ngũ Lão |
| −0,2177 | 4,9 | 1465 | 26,50 | Hà Nội 18 Hoàng Quốc Việt |
| −0,1596 | 21,8 | 1508 | 20,05 | Hải Phòng UB phường Phạm Sư Mạnh |

Mối quan hệ khoảng-cách–kỹ-năng rõ ràng về mặt thống kê, và được tóm tắt bền vững nhất bởi tương quan hạng, vốn không nhạy với giá trị ngoại lai âm cực đoan. Trên ba mươi chín vị trí chi phí thấp, tương quan hạng Spearman giữa R² theo vị trí và khoảng cách đến trạm quan trắc gần nhất là ρ = −0,62 (p < 0,001). Tương quan Pearson là −0,43 (p = 0,007) khi bao gồm tất cả các vị trí, nhưng nó nhạy với giá trị ngoại lai −14,75 duy nhất: việc loại trừ một vị trí đó làm yếu nó xuống −0,34 (p = 0,034), nên liên kết tuyến tính, dù vẫn có ý nghĩa, được mang một phần bởi điểm cực đoan đó, trong khi tương quan hạng thì không. R² trung vị rơi từ 0,6396 cho mười sáu vị trí trong vòng 10 km xuống 0,4487 cho hai mươi ba vị trí ngoài 10 km. Đây là ràng buộc suy-giảm-theo-khoảng-cách của Mục 4.5 được biểu đạt ở cấp vị trí: kỹ năng suy giảm đơn điệu với sự cô lập khỏi trạm neo gần nhất. Dưới cách diễn đạt chuyển đặc trưng ở trên, một phần của sự suy giảm này mang tính cơ học — một vị trí chi phí thấp càng nằm xa trạm quan trắc gần nhất, sai số càng lớn trong việc thay thế khí tượng và đặc trưng vệ tinh của trạm đó cho điều kiện riêng của vị trí — nên đường cong suy-giảm-theo-khoảng-cách phản ánh cả sai số thay-thế-đặc-trưng tăng dần lẫn sự mất mát thực sự của kỹ năng nội-suy-không-gian.

*Bảng 5.18: Thống kê khoảng-cách–kỹ-năng trên ba mươi chín vị trí chi phí thấp — các tương quan Pearson và Spearman và R² trung vị gần-so-với-xa.*

| Tham số | Giá trị |
|---|---|
| Pearson r (khoảng cách, R²) | −0,4255 |
| Spearman ρ (khoảng cách, R²) | −0,6188 |
| Trung vị R², khoảng cách ≤ 10 km (n = 16) | 0,6396 |
| Trung vị R², khoảng cách > 10 km (n = 23) | 0,4487 |
| Dải khoảng cách | 1,1 – 44,4 km |

Đọc đối chiếu với tài liệu khoa học quốc tế, các kết quả độc lập này cạnh tranh chính xác ở nơi mạng lưới dày đặc, với lưu ý rằng các con số Việt Nam là một phép kiểm tra chuyển đặc trưng từ trạm gần nhất chứ không phải đánh giá chéo không gian hoàn toàn độc lập được báo cáo trong các nghiên cứu đó, nên so sánh mang tính chỉ dấu chứ không phải nghiêm ngặt như-thể-tương-đương. Trung vị cảm-biến-chi-phí-thấp 0,529 thấp hơn một cách vừa phải so với R² đánh giá chéo không gian Ấn Độ 0,67 của Kawano et al. (2025) — không bất ngờ, vì kết quả của họ dựa trên khoảng một nghìn trạm huấn luyện so với ba mươi bảy của Việt Nam — nhưng khoảng cách thu hẹp ở cự ly gần: R² trung vị của các vị trí trong vòng 10 km (0,64) và vị trí tham chiếu Đại sứ quán Hoa Kỳ (0,684) ngang bằng hoặc vượt mốc chuẩn Ấn Độ, và mười vị trí chi phí thấp tốt nhất trải từ 0,647 đến 0,764. Do đó cách diễn đạt trung thực là các dự đoán dựa-trên-chuyển-giao của mô hình tiệm cận các kết quả đánh giá chéo không gian tốt nhất được công bố trong vùng phủ dày đặc và rơi xuống dưới chúng khi mạng lưới mỏng đi — chính là câu chuyện ràng-buộc-mật-độ của Chương 4 được nêu lại như một so sánh mốc chuẩn độc lập. Hai ghi chú về tính so sánh được là cần thiết. Thứ nhất, các bản ghi chi phí thấp ngắn (phần lớn khoảng 1.000–1.500 giờ), chỉ mùa khô, và các cảm biến là tán xạ laser chứ không phải cấp-tham-chiếu, nên các giá trị R² chi phí thấp riêng lẻ mang nhiều nhiễu lấy mẫu hơn các con số quan trắc và Đại sứ quán, điều này giải thích một phần độ trải rộng bao gồm đuôi âm. Thứ hai, bởi vì các đặc trưng vệ tinh và khí tượng được chuyển từ trạm quan trắc gần nhất, một phần của sự ngang bằng cự-ly-gần phản ánh sai số thay-thế-đặc-trưng nhỏ ở các khoảng cách ngắn chứ không phải kỹ năng nội suy độc lập.

Một phép kiểm tra độc lập thứ hai, nghiêm ngặt hơn đánh giá quy trình định tuyến tiên nghiệm không gian của Mục 3.6.4 — mô hình quốc gia triển khai được giải quyết phụ thuộc vòng tròn — trực tiếp trên các trạm chưa thấy. Quy trình được huấn luyện trên bốn mươi trạm của luận văn và được dùng để dự đoán tại bốn mươi sáu cảm biến chi phí thấp độc lập không tham gia gì vào quá trình phát triển mô hình; khác với phép kiểm tra chuyển giao đồng bằng ở trên, mỗi cảm biến giữ lại mang các đặc trưng vệ tinh, AOD, và khí tượng riêng của nó được tính tại tọa độ riêng của nó, và tiên nghiệm đường-cơ-sở-không-gian được neo vào bốn mươi trạm huấn luyện với cảm biến giữ lại bị loại trừ. Do đó đây là một bài kiểm tra sức chịu đựng huấn-luyện-trên-bốn-mươi, dự-đoán-phần-chưa-thấy thực sự của toàn bộ chuỗi chứ không phải một kiểm tra chuyển đặc trưng. Quy trình thuần đạt R² gộp 0,332, R² trung bình theo trạm 0,198 và trung vị 0,216, với ba mươi lăm trong bốn mươi sáu cảm biến dương — một sự phục hồi rõ rệt so với chỉ riêng luồng biên-độ-cao thô (R² gộp 0,127) và một kết quả mà, đáng chú ý, tái lập trung bình theo trạm nội bộ khoảng 0,20 trên các trạm chưa bao giờ thấy trong huấn luyện. Thuộc tính an-toàn-bản-đồ quyết định là nó tạo ra **không có phân loại sai thấp-thành-cao nguy hiểm nào**: mô hình không ánh xạ bất kỳ cảm biến sạch nào thành ô nhiễm. Các sai số phần dư của nó gần như hoàn toàn là các nhầm lẫn ranh giới trung-bình/cao trong dữ liệu chi phí thấp nhiễu — mười cảm biến cao-thực được dự đoán trung bình và tám cảm biến trung-bình-thực được dự đoán cao — chứ không phải các sai số chế độ thô. Việc áp dụng bộ bảo vệ độ tin cậy duy nhất của Mục 3.6.4 như một lớp hiển-thị-bản-đồ để các dự đoán số về cơ bản không đổi trên bốn mươi trạm nội bộ và, trên các cảm biến độc lập, gắn cờ sáu cảnh báo cao-ẩn và một cao-sai rõ ràng; việc tính các vị trí cao-ẩn được gắn cờ như không-bỏ-sót giảm số bỏ-sót cao-thành-không-cao hiệu dụng từ mười xuống năm trong khi vẫn không đưa vào nguy hiểm thấp-thành-cao nào. Các con số số học độc lập nên được đọc như một bài kiểm tra sức chịu đựng chứ không phải một sự lặp lại của điểm số nội bộ — các cảm biến chi phí thấp nhiễu hơn, chỉ mùa khô, và bao gồm các vi môi trường cục bộ vắng mặt khỏi tập huấn luyện bốn-mươi-trạm — nhưng kết luận định tính bền vững: quy trình triển khai được tổng quát hóa đến các trạm thực sự chưa thấy như một bản đồ sàng lọc ô-nhiễm-cao, với độ thiên lệch thận trọng phù hợp với một sản phẩm y tế công cộng.

*Bảng 5.19: Quy trình định tuyến tiên nghiệm không gian trên bốn mươi sáu cảm biến chi phí thấp chưa thấy (huấn-luyện-trên-bốn-mươi, dự-đoán-phần-chưa-thấy). Các tham số số học là một bài kiểm tra sức chịu đựng miền-nhiễu; kết quả vận hành là không có phân loại sai thấp-thành-cao nguy hiểm nào, với bộ bảo vệ độ tin cậy cắt giảm một nửa số bỏ-sót-cao hiệu dụng.*

| Biến thể | R² gộp | R² trạm trung bình | R² trạm trung vị | % dương | Cao→không-cao | Nguy hiểm thấp→cao |
|---|---:|---:|---:|---:|---:|---:|
| chỉ luồng biên-độ-cao thô | 0,127 | 0,005 | 0,083 | — | 19 | 0 |
| định tuyến tiên nghiệm không gian (thuần) | 0,332 | 0,198 | 0,216 | 35/46 | 10 | 0 |
| + bộ bảo vệ độ tin cậy (số học) | 0,335 | 0,205 | 0,264 | 35/46 | 10 | 0 |
| + bộ bảo vệ độ tin cậy (bản đồ hiệu dụng) | 0,335 | 0,205 | 0,264 | 35/46 | 5 | 0 |


## 5.7 Tổng hợp các thí nghiệm không thành công

Ngưỡng trần triển khai khoảng +0,04 trung vị không được chấp nhận trước khi nó được kiểm tra. Bảy họ phương pháp riêng biệt được phát triển cụ thể để phá vỡ phụ thuộc vòng tròn — để phục hồi một phần mức tăng tầng +0,21 mà không dùng tầng thực hay trung bình thực của trạm giữ lại — và cả bảy đều thất bại. Việc ghi nhận chúng không phải một suy nghĩ thêm: các kết quả âm giới hạn không gian giải pháp và, gộp lại, cấu thành luận điểm thực nghiệm rằng không có đại diện quan sát được từ vệ tinh nào giải quyết đáng tin cậy phụ thuộc, vốn là đóng góp thứ ba đã nêu ở Chương 1. Mỗi họ được tóm tắt dưới đây, với kết quả tốt nhất nó đạt được và lý do nó thất bại; điểm tham chiếu xuyên suốt là trung bình theo trạm true-tier (hard-T4F) 0,198, mà mọi biến thể triển khai được của các phương pháp này không đạt tới.

Họ soft-T4F thay các ranh giới tầng cứng bằng các ranh giới mềm có-trọng-số-Gauss, dựa trên giả thuyết rằng thành viên tầng mượt sẽ tổng quát hóa tốt hơn gán rời rạc. Nó không: cấu hình mềm tốt nhất đạt trung bình theo trạm chỉ 0,072, và hiệu suất giảm đơn điệu khi ranh giới được làm mềm hơn (0,072, rồi 0,037, rồi 0,021, rồi 0,002), xác nhận rằng các ranh giới cứng luôn thắng — tầng rời rạc mang thông tin mà việc làm mượt phá hủy. Họ GHAP-lai nhân khí hậu vệ tinh GHAP với các đặc trưng nội-suy-không-gian như một số hạng tương tác; nó đạt 0,166, thấp nhất trong tệp bốn-cấu-hình của nó và dưới cơ sở 0,196, bởi vì GHAP cạnh tranh chứ không bổ sung cho tín hiệu RFSI dưới LOSO. Họ đặc-trưng-bất-biến IRM loại bỏ các đặc trưng phụ-thuộc-chế-độ ("không ổn định") với hy vọng cô lập một biến dự đoán bất biến chuyển giao được; mọi biến thể loại bỏ đều âm (loại bỏ lỏng nhất tệ nhất ở −0,169), chứng minh rằng các đặc trưng được cho là không ổn định mang tín hiệu thực, không thể thay thế. Họ dự-đoán-dị-thường tái khung mục tiêu thành một độ lệch so với một đường cơ sở trượt chứ không phải một nồng độ tuyệt đối; cấu hình độ lệch tốt nhất chỉ đạt 0,065, và biến thể đường-cơ-sở-một-tuần thảm họa ở −0,78, bởi vì một đường cơ sở nhiễu khiến độ lệch khó dự đoán hơn mức tuyệt đối — mục tiêu dị-thường-theo-ngày hầu như không khác đối ứng tuyệt đối của nó (0,145 so với 0,142).

Ba họ còn lại đáng được hiệu chỉnh các con số được đề xuất trong dàn ý ban đầu, bởi vì các con số của dàn ý cho chúng là các ngưỡng trần oracle hoặc các giá trị đơn-trạm chứ không phải điểm số triển khai được riêng của các phương pháp. Họ tổ-hợp-tầng huấn luyện cả bốn mô hình tầng và lấy trung bình các dự đoán của chúng; con số "0,246" của dàn ý không phải điểm số của phương pháp này mà là ngưỡng trần oracle được đóng gói trong cùng tệp thí nghiệm, trong khi mọi cấu hình tầng-trung-bình thực tế đều âm về trung bình (tốt nhất, lấy trung bình đơn giản, ở −0,068), bởi vì ba mô hình tầng-sai tiêm nhiều nhiễu hơn một mô hình tầng-đúng loại bỏ. Họ hai-pha dự đoán tầng của mỗi trạm từ các biến quan sát được rồi dùng mô hình của tầng dự đoán; con số "0,249" của dàn ý lại là ngưỡng trần oracle base-margin trong tệp đó, không phải điểm số triển khai được — cấu hình dự-đoán-rồi-dùng thực tế sụp xuống −0,063, bởi vì tầng dự đoán đúng quá hiếm (cổng chỉ đạt khoảng 40% độ chính xác tầng) và việc định tuyến sai phá hủy mức tăng. Họ KNN/cụm-phát-thải gán tầng hoặc cụm từ các đặc trưng quan sát được bằng các sơ đồ trạm-lân-cận-gần-nhất và phân cụm; dải "0,274–0,544" của dàn ý không tương ứng với trung bình cấp-cấu-hình nào — đó là các giá trị R² đơn-trạm rải rác — và mọi trung bình cấu hình cụm hoặc KNN thực ra đều âm (tốt nhất, phân nhóm phát thải bốn-cụm, ở −0,030), với họ cấu hình đặc-trưng-phát-thải rộng hơn tất cả đều rơi dưới tham chiếu hard-T4F 0,198. Do đó khẳng định có thể bảo vệ cho họ này là khẳng định định tính nêu trong dàn ý: định tuyến cụm và KNN dựa-trên-biến-quan-sát vẫn tệ hơn hard T4F.

*Bảng 5.20: Bảy họ thí nghiệm không thành công, mỗi họ với R² trung-bình-theo-trạm tốt nhất của nó và lý do nó không đạt tham chiếu hard-T4F 0,198.*

| Thí nghiệm | Phương pháp | R² tốt nhất (trung bình theo trạm) | Vì sao thất bại |
|---|---|---|---|
| Soft T4F | Ranh giới tầng có-trọng-số-Gauss | 0,072 | Ranh giới cứng luôn thắng; kỹ năng giảm đơn điệu khi ranh giới mềm hơn |
| GHAP lai | Tương tác GHAP × RFSI | 0,166 (dưới cơ sở 0,196) | GHAP cạnh tranh với RFSI dưới LOSO; xếp cuối trong các cấu hình của nó |
| Tổ hợp tầng | Huấn luyện cả bốn tầng, lấy trung bình | −0,068 (ngưỡng trần oracle 0,198) | Các mô hình tầng-sai thêm nhiều nhiễu hơn mô hình đúng loại bỏ |
| Hai-pha tầng | Dự đoán tầng, rồi dùng mô hình tầng | −0,063 (ngưỡng trần oracle 0,249) | Tầng dự đoán chỉ đúng ~40% thời gian; định tuyến sai phá hủy mức tăng |
| KNN / cụm phát thải | Gán tầng/cụm dựa-trên-biến-quan-sát | −0,030 (mọi cấu hình âm) | Định tuyến cụm/KNN vẫn tệ hơn hard T4F (0,198) |
| Đặc trưng bất biến IRM | Loại bỏ các đặc trưng phụ-thuộc-chế-độ | −0,051 (mọi cấu hình âm) | Các đặc trưng "không ổn định" mang tín hiệu thực; loại bỏ thành âm |
| Dự đoán dị thường | Dự đoán độ lệch so với đường cơ sở | 0,065 | Đường cơ sở nhiễu khiến độ lệch khó hơn tuyệt đối; đường cơ sở một-tuần thảm họa (−0,78) |

Bài học của bảy thất bại này nhất quán và, theo cách của nó, giàu thông tin hơn một thành công. Nhãn tầng mã hóa mức ô nhiễm trung bình của trạm, và trung bình đó chính xác là đại lượng không thể quan sát được tại một vị trí không có quan trắc mà không có phép đo mặt đất. Mọi nỗ lực phục hồi nó từ các biến quan sát được — bằng cách làm mềm nó, phân cụm về phía nó, dự đoán nó, loại bỏ sự phụ thuộc vào nó, hoặc né tránh nó qua các dị thường — hoặc chỉ tái tạo một phần của tín hiệu hoặc phá hủy nó hoàn toàn. Do đó các thí nghiệm không chỉ thất bại; chúng tam giác đạc ranh giới của điều gì có thể đạt được, và chúng xác lập rằng trung vị +0,04 triển khai được là một ngưỡng trần thông tin thực sự *cho ước tính dựa-trên-biến-quan-sát-từ-vệ-tinh về đường cơ sở*, không phải một sự thiếu sót về nỗ lực. Phụ thuộc vòng tròn là thực và ràng buộc đối với lớp phương pháp đó: không có đại diện quan sát được từ vệ tinh nào được kiểm tra trong luận văn này phục hồi được trung bình bị thiếu.

Tuy nhiên, ranh giới mà bảy thất bại này tam giác đạc cụ thể là ranh giới của ước tính *quan sát được*, và việc định vị nó chính xác chỉ ra nơi thông tin bị thiếu thực sự nằm. Mức ô nhiễm trung bình không thể được đọc ra từ các biến quan sát được từ vệ tinh hay sử dụng đất — nhưng nó có thể được nội suy từ mạng lưới quan trắc xung quanh, bởi vì đường cơ sở của một trạm được dự đoán mạnh bởi các đường cơ sở quan trắc của các trạm lân cận của nó. Đây là đòn bẩy mà bảy họ chưa bao giờ kéo: mỗi họ trong số chúng, bao gồm bộ dự đoán hai-pha và các sơ đồ phân cụm, tìm tầng trong không gian đặc trưng quan sát được, trong khi lời giải nằm trong không gian vật lý. Quy trình định tuyến tiên nghiệm không gian của Mục 3.6.4 và 5.5 quả thực kéo nó, phục hồi một trung bình theo trạm 0,197 so với 0,203 của oracle bằng cách neo đường cơ sở vào các trạm quan trắc lân cận thay vì đoán nó từ vệ tinh. Điều này không làm yếu bảy kết quả âm — nó làm sắc nét diễn giải của chúng. Phụ thuộc vòng tròn không thể phá vỡ từ riêng các biến quan sát được, và có thể phá vỡ bằng nội suy không gian từ mạng lưới; cái giá của lời giải là nó kế thừa tầm với của mạng lưới, hoạt động ở nơi mục tiêu có các trạm lân cận khả dụng và suy giảm ở nơi không có. Do đó ngưỡng trần triển khai không phải một con số duy nhất mà là một hàm của sự gần kề trạm, và ràng buộc then chốt, một lần nữa, là mật độ của mạng lưới mặt đất chứ không phải thuật toán.


## 5.8 Một góc nhìn thống nhất: nội suy và ngoại suy

Các mục trước đã đánh giá ba khung đánh giá mà cho đến nay được xử lý như những khung riêng biệt. Đọc cùng nhau, chúng phân giải thành một nguyên lý tổ chức duy nhất giải thích vì sao các con số nổi bật biến thiên rộng đến vậy — từ trên 0,8 đến dưới 0,1 — cho cái về danh nghĩa là một mô hình trên một bộ dữ liệu. Mọi dự đoán mà mô hình có thể được yêu cầu thực hiện hoặc là một *nội suy* giữa các quan trắc đã biết hoặc là một *ngoại suy* đến một điểm thực sự chưa được quan trắc, và kỹ năng có thể đạt được được chi phối gần như hoàn toàn bởi việc một bài toán cho trước đòi hỏi cái nào trong hai. Các con số cao trong tài liệu khoa học, và trong luận văn này, là nội suy; mục tiêu vận hành — một bản đồ ô nhiễm cho các vị trí không có quan trắc — là ngoại suy; và việc gộp lẫn hai cái là chính xác sai lầm mà phê phán đánh giá của Chương 2 đã xác định.

Trước tiên hữu ích là tách hai bài toán mà thuật ngữ "dự đoán thời gian" lặng lẽ gộp lại, bởi vì chúng nằm ở hai phía đối lập của ranh giới nội-suy–ngoại-suy. Thứ nhất là *lấp khoảng trống thời gian*: bù các giờ thiếu tại một trạm có lịch sử đo đạc, trường hợp vận hành của cảm biến ngừng hoạt động hoặc khoảng trống bảo trì. Thứ hai là *dự báo thời gian*: dự đoán các giờ nằm trong tương lai, ngoài cuối bản ghi huấn luyện. Để đo từng cái một cách sạch sẽ, mô hình được huấn luyện lại theo từng trạm chỉ dùng các đặc trưng ngoại sinh (khí tượng, quan trắc vệ tinh, và các mã hóa thời gian — không có lịch sử PM), một lần dưới một phép chia năm-fold ngẫu nhiên trên các giờ riêng của trạm (lấp khoảng trống) và một lần dưới một phép chia theo trình tự thời gian huấn luyện trên 70% đầu của bản ghi và dự đoán 30% cuối (dự báo). Sự tương phản là quyết định và được báo cáo trong Bảng 5.21.

*Bảng 5.21: Cùng một mô hình đặc-trưng-ngoại-sinh theo trạm được đánh giá như bốn bài toán khác nhau, theo tầng ô nhiễm. Lấp khoảng trống và dự báo dùng dữ liệu riêng của trạm trong huấn luyện; bản đồ không gian (LOSO) thì không. Dự báo được phân tách thành R² thô và kỹ năng hình dạng (bình phương tương quan Pearson của chuỗi tương lai dự đoán và quan trắc), vốn tách sự bám mẫu khỏi trôi đường cơ sở.*

| Tầng | Lấp khoảng trống thời gian (R² KFold trong-trạm) | Dự báo thời gian — R² thô (trung vị) | Dự báo thời gian — r² hình dạng (trung vị) | Bản đồ không gian (R² LOSO) |
|---|---:|---:|---:|---:|
| t0 (sạch) | +0,73 | −0,57 | +0,03 | −0,14 |
| t1 | +0,60 | +0,11 | +0,19 | −0,02 |
| t2 | +0,66 | −0,11 | +0,25 | +0,33 |
| t3 (ô nhiễm) | +0,76 | +0,01 | +0,35 | +0,56 |
| Tất cả | +0,68 | −0,07 | +0,20 | +0,20 |

Ba cách đọc theo trực tiếp từ bảng. Thứ nhất, *lấp khoảng trống thời gian là nội suy thực sự và nó hoạt động ở khắp nơi*, đạt một R² trong-trạm 0,68 trung bình và 0,73 ngay cả ở tầng sạch nhất — chính tầng mà bản đồ không gian thất bại. Lấp một giờ thiếu là dễ bởi vì mô hình nội suy giữa các giờ quan trắc xung quanh của cùng trạm, và đây là một năng lực thực sự, triển khai được cho bốn mươi trạm được trang bị bất kể mức ô nhiễm của chúng. Đây là cùng một nội suy mà, dưới đánh giá chéo ngẫu nhiên của Mục 5.2, đã tạo ra R² nổi bật khoảng 0,80; con số đó không sai, nó chỉ đơn giản đo lấp khoảng trống chứ không phải lập bản đồ.

Thứ hai, *dự báo thời gian là ngoại suy trong thời gian, và nó suy giảm tương ứng* — nhưng không thảm họa như chỉ riêng R² thô gợi ý, và sự phân tách quan trọng. R² dự báo thô gần không hoặc âm ở mọi tầng, thế nhưng kỹ năng hình dạng, bình phương tương quan giữa chuỗi tương lai dự đoán và quan trắc, tăng từ 0,03 tại các vị trí sạch lên 0,35 tại các vị trí ô nhiễm, tương ứng với một tương quan Pearson gần 0,5–0,6 tại các tầng trung bình và ô nhiễm. Do đó mô hình *quả thực* bám theo mẫu tương lai tại các trạm có biến thiên ô nhiễm đáng kể; R² thô âm chủ yếu do một *sự trôi đường cơ sở* giữa giai đoạn huấn luyện và tương lai giữ lại — một dịch chuyển mức trung vị khoảng 8 đến 13 µg/m³ tại các tầng ô nhiễm — chứ không phải do một thất bại trong việc theo mẫu thời gian. Việc hiệu chỉnh độ dời đó nâng R² dự báo lên giữa +0,11 và +0,32 tại t1–t3. Kết luận trung thực là dự báo tương lai từ các đặc trưng ngoại sinh là khiêm tốn và bị-giới-hạn-bởi-trôi, chỉ hữu dụng với việc tái hiệu chuẩn định kỳ, và vắng mặt tại các vị trí sạch nơi ngay cả mẫu cũng không tương quan; nó không phải, và không nhằm là, mục tiêu của luận văn này.

Thứ ba, và quan trọng nhất, *bản đồ không gian là nội suy giữa các trạm trong không gian*, và nó thành công và thất bại theo đúng cùng một logic như các bài toán thời gian. Nơi mạng lưới đủ dày để một vị trí mục tiêu có thể được nội suy từ các trạm lân cận gần, bản đồ hoạt động tốt: đánh giá độc lập của Mục 5.6 đạt một R² trung vị 0,53 trên các vị trí chi phí thấp, 0,64 trong vòng 10 km của một trạm neo, và 0,68 tại trạm tham chiếu Đại sứ quán Hoa Kỳ. Nơi mục tiêu phải được ngoại suy ngoài tầm với của mạng lưới — xa bất kỳ trạm neo nào, hoặc trong các chế độ sạch và nông thôn được trang bị thưa thớt — nó sụp đổ về phía không, như cả mối quan hệ suy-giảm-theo-khoảng-cách của Mục 4.5 lẫn phân tách LOSO theo tầng của Mục 5.4 đều cho thấy. Đây là lý do phân tích tầm quan trọng đặc trưng của Chương 4 phát hiện nội suy không gian từ các trạm lân cận mặt đất (RFSI) là tác nhân đóng góp đơn lẻ lớn nhất cho độ tăng của mô hình: bản đồ, về mặt cơ học, là một bộ nội suy không gian, và kỹ năng của nó bị chặn bởi mật độ của các điểm mà nó nội suy giữa.

Do đó bức tranh thống nhất là một trục duy nhất. Các bài toán nội suy giữa các quan trắc đã biết — lấp các giờ thiếu của một trạm, hoặc lập bản đồ một vị trí được bao quanh bởi các trạm gần — là giải được, với R² từ khoảng 0,5 đến 0,8. Các bài toán ngoại suy đến các điểm thực sự chưa được quan trắc — dự báo tương lai của một trạm, hoặc lập bản đồ một vị trí xa bất kỳ trạm neo nào hoặc trong một chế độ sạch được lấy mẫu thiếu — thì không, với R² gần không. Cùng một mô hình trải dài toàn bộ dải này, và mâu thuẫn biểu kiến giữa "năng lực 0,8" và "năng lực 0,2" của nó tan biến một khi mỗi con số được gắn nhãn bởi bài toán mà nó đo. Đóng góp của luận văn này không phải một con số độ chính xác duy nhất mà là phân loại học này: một phát biểu chính xác về việc những bài toán ước tính chất lượng không khí nào có thể đạt được từ dữ liệu vệ tinh trên một mạng lưới nhiệt đới thưa thớt, và những bài toán nào bị giới hạn bởi tầm với của mạng lưới mặt đất chứ không phải bởi mô hình. Lấp khoảng trống tại các vị trí được trang bị và lập bản đồ trong vùng phủ dày đặc là khả thi hôm nay; dự báo và lập bản đồ vào các vùng sạch, thưa thớt chờ một mạng lưới dày hơn.


Chương 5 đã trình bày đánh giá định lượng đầy đủ trên ba khung đánh giá. Đánh giá chéo ngẫu nhiên xác nhận rằng mô hình là một bộ nội suy thời gian có năng lực ở R² 0,81, trong khi đánh giá bỏ-một-trạm-ra trung thực cho thấy khoảng ba phần tư con số đó là rò rỉ danh-tính-trạm, để lại một R² dự-đoán-không-gian thực sự gần 0,20 một khi mô hình được cho tầng thực của mỗi trạm. Phân tầng được chứng minh là đòn bẩy đơn lẻ lớn nhất — một mức tăng khoảng +0,17 trong-cùng-tệp, tăng lên khoảng +0,26 qua các cách diễn đạt — và phân tách theo tầng xác lập một gradient đơn điệu, bất biến-mô-hình mà theo đó khả năng dự đoán tỷ lệ với mức ô nhiễm; việc liệu mức chứ không phải vùng là biến chi phối vẫn là một giả thuyết mà mạng lưới không thể giải quyết, vì tất cả các trạm tầng-đỉnh đều ở phía Bắc và sự tương phản Bắc–Nam duy nhất, ở t2, không có ý nghĩa thống kê. So sánh triển khai được chứng minh rằng việc rút tầng oracle làm sụp đổ hiệu suất xuống một trung vị theo trạm gần 0,04, với khoảng cách triển-khai-đến-oracle +0,27 định lượng chi phí thông tin của việc không biết trung bình của một trạm, và bảy họ thí nghiệm không thành công xác nhận rằng không có đại diện quan sát được từ vệ tinh nào phục hồi được nó. Đánh giá độc lập so với mạng lưới cảm-biến-chi-phí-thấp và trạm tham chiếu Đại sứ quán Hoa Kỳ — một phép kiểm tra chuyển đặc trưng từ trạm gần nhất cộng với sự phù hợp cảm biến chứ không phải dự đoán không gian hoàn toàn độc lập — cho thấy mô hình đạt một R² trung vị mùa khô 0,53 và một R² cấp-tham-chiếu 0,68 trong vùng phủ dày đặc, ngang bằng với các mốc chuẩn quốc tế tốt nhất ở cự ly gần, trong khi suy giảm mạnh theo khoảng cách đến trạm neo gần nhất. Sợi chỉ thống nhất là kỹ năng dự đoán không gian ở Việt Nam bị chặn trên bởi tầm với của mạng lưới mặt đất chứ không phải bởi thuật toán. Chương 6 gắn kết các kết quả này thành các kết luận và chuyển ràng buộc mật độ then chốt thành một khuyến nghị cụ thể: rằng Việt Nam theo đuổi một chiến lược quan trắc lai gồm các trạm chuẩn-tham-chiếu thưa thớt được làm dày bằng các cảm biến chi phí thấp đã hiệu chuẩn, sự can thiệp mà bằng chứng suy-giảm-theo-khoảng-cách xác định là đòn bẩy trực tiếp và khả thi nhất lên độ chính xác dự đoán PM2.5 quốc gia.
