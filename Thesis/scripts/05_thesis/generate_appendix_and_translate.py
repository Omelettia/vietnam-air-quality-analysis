"""Generate final thesis appendices.

This script intentionally writes only the final appendices.  Older versions also
rewrote translated chapters and listed exploratory experiment branches; that is
not appropriate for the final Red River Delta thesis package.
"""
from pathlib import Path
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TEX = ROOT / "Thesis" / "latex"
CHUONG = TEX / "Chuong"


def esc_tex(value):
    text = unicodedata.normalize("NFC", str(value))
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        text = text.replace(a, b)
    return text


def read_csv(path):
    return pd.read_csv(ROOT / path)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


def appendix_a():
    stn = read_csv("Thesis/results/01_stations/station_selection_final.csv")
    stn = stn.sort_values(["region", "tier", "station_name"])

    feat = read_csv("Thesis/results/01_stations/feature_list.csv")

    stn_rows = []
    for _, r in stn.iterrows():
        name = esc_tex(str(r["station_name"]))
        stn_rows.append(
            f"  {name} & {r['lat']:.2f} & {r['lon']:.3f} & "
            f"{esc_tex(r['region'])} & t{int(r['tier'])} & "
            f"{float(r['pm25_coverage']) * 100:.0f}\\% \\\\"
        )

    feat_rows = []
    for _, r in feat.iterrows():
        feat_rows.append(
            f"  {esc_tex(r['feature'])} & {esc_tex(r['type'])} & "
            f"{esc_tex(r['source'])} & {float(r['coverage_pct']):.0f}\\% \\\\"
        )

    rfsi_rows = [
        ("PM25_nn_idw", "Nội suy PM2.5 theo nghịch đảo khoảng cách từ các trạm neo cùng thời điểm"),
        ("PM25_nn1", "PM2.5 của trạm neo gần nhất"),
        ("PM25_nn2", "PM2.5 của trạm neo gần thứ hai"),
        ("PM25_nn3", "PM2.5 của trạm neo gần thứ ba"),
        ("PM25_upwind_idw", "Nội suy PM2.5 từ các trạm nằm gần hướng đầu gió"),
        ("PM25_downwind_idw", "Nội suy PM2.5 từ các trạm nằm gần hướng cuối gió"),
        ("PM25_wind_spread", "Chênh lệch giữa PM2.5 đầu gió và cuối gió"),
        ("PM25_neighbor_spread", "Độ phân tán PM2.5 giữa các trạm neo lân cận"),
    ]
    rfsi_table = "\n".join(
        f"  \\path{{{name}}} & {esc_tex(desc)} \\\\"
        for name, desc in rfsi_rows
    )

    content = r"""\documentclass[../DoAn.tex]{subfiles}
\begin{document}

Phụ lục này cung cấp các bảng tham chiếu bổ sung cho bộ dữ liệu và đặc trưng được sử dụng trong đồ án.

\section{Danh sách trạm quan trắc}

Bảng dưới đây liệt kê """ + str(len(stn)) + r""" trạm quan trắc chất lượng không khí (KK) trong tập phát triển. Các thí nghiệm quốc gia dùng đủ 40 trạm theo giao thức bỏ một trạm ra; kiểm soát chất lượng PM2.5 được áp dụng ở cấp từng dòng dữ liệu.

{\small
\begin{longtable}{p{5.5cm}rrllr}
\caption{Danh sách trạm KK trong tập phát triển}\\
\toprule
\textbf{Tên trạm} & \textbf{Vĩ độ} & \textbf{Kinh độ} & \textbf{Vùng} & \textbf{Tier} & \textbf{Phủ PM2.5} \\
\midrule\endfirsthead
\toprule
\textbf{Tên trạm} & \textbf{Vĩ độ} & \textbf{Kinh độ} & \textbf{Vùng} & \textbf{Tier} & \textbf{Phủ PM2.5} \\
\midrule\endhead
""" + "\n".join(stn_rows) + r"""
\bottomrule
\end{longtable}
}

\section{Các cột nền trong bảng hợp nhất}

Bảng dưới đây liệt kê """ + str(len(feat)) + r""" cột nền trong \path{unified_thesis.csv}, phân loại theo nguồn dữ liệu. PM2.5 trong bảng này là biến mục tiêu dùng để huấn luyện và đánh giá, không phải đặc trưng đầu vào của mô hình. Tỷ lệ phủ trong bảng được tính trên 40 trạm KK của tập phát triển; Bảng 3.1 trong thân đồ án tính trên toàn bộ bảng hợp nhất 121 trạm, nên tỷ lệ ở đó thấp hơn (ví dụ AOT trung bình quanh trạm 51\% so với 36\%).

{\small
\begin{longtable}{p{4cm}lp{3cm}r}
\caption{Các cột nền trong bảng hợp nhất}\\
\toprule
\textbf{Cột} & \textbf{Loại} & \textbf{Nguồn} & \textbf{Phủ (\%)} \\
\midrule\endfirsthead
\toprule
\textbf{Cột} & \textbf{Loại} & \textbf{Nguồn} & \textbf{Phủ (\%)} \\
\midrule\endhead
""" + "\n".join(feat_rows) + r"""
\bottomrule
\end{longtable}
}

\section{Đặc trưng RFSI sinh trong từng fold}

Các cột RFSI không nằm sẵn trong \path{feature_list.csv}. Chúng được tính trong script thí nghiệm vùng từ PM2.5 đồng thời của các trạm neo lân cận. Trong đánh giá bỏ một trạm ra, trạm đang được kiểm tra luôn bị loại khỏi nhóm neo trước khi tính các đặc trưng này.

{\small
\begin{longtable}{p{5.8cm}p{7.9cm}}
\caption{Đặc trưng RFSI dùng trong mô hình vùng}\\
\toprule
\textbf{Đặc trưng} & \textbf{Ý nghĩa} \\
\midrule\endfirsthead
\toprule
\textbf{Đặc trưng} & \textbf{Ý nghĩa} \\
\midrule\endhead
""" + rfsi_table + r"""
\bottomrule
\end{longtable}
}

\section{Google Earth Engine scripts}

Các script Google Earth Engine được dùng để trích xuất dữ liệu vệ tinh và khí vết tại tọa độ trạm:

\begin{enumerate}
\item \textbf{MAIAC AOD}: AOD MODIS tại pixel trung tâm và thống kê vùng lân cận.
\item \textbf{TROPOMI}: NO$_2$, SO$_2$, CO và HCHO theo ngày.
\item \textbf{MODIS LST}: nhiệt độ bề mặt ngày và đêm.
\item \textbf{GEOS-CF / MERRA-2}: sản phẩm mô hình và tái phân tích khí quyển dùng cho đánh giá tham chiếu, không dùng làm mô hình cuối.
\item \textbf{Himawari AOD}: nguồn AOD tần suất cao có trong bảng hợp nhất và được dùng để kiểm tra, so sánh với MODIS trong các nhánh phân tích nguồn AOD. Mô hình vùng cuối vẫn ưu tiên các đặc trưng AOD đã được chuẩn hóa và thống nhất trong pipeline cuối.
\end{enumerate}

Các biến HCHO trong mô hình vùng được tạo trong \path{exp_red_river_delta.py} từ bảng GEE theo ngày, gồm anomaly, thống kê lăn 30 ngày và một số tương tác với điều kiện mặt đất. Vì vậy HCHO có thể không xuất hiện như một cột nền riêng trong \path{feature_list.csv}.

\end{document}
"""
    write(CHUONG / "Phu_luc_A.tex", content)


def appendix_b():
    rows = [
        ("1", "Thu thập dữ liệu trạm và vệ tinh", "01_collection/", "dữ liệu thô theo trạm"),
        ("2", "Dựng dữ liệu hợp nhất", "build_unified.py", "unified_thesis.csv"),
        ("3", "Lọc chất lượng PM2.5", "pm25_qc.py", "mặt nạ QC theo từng dòng"),
        ("4", "Tương quan AOD và PM2.5", "aod_pm25_correlation_paper.py", "phân tích Mục 4.3"),
        ("5", "Trần dự đoán trong từng trạm", "within_station_predictability.py", "kết quả Mục 4.5"),
        ("6", "Đánh giá sản phẩm PM2.5 toàn cầu", "exp_satellite_products.py", "so sánh Mục 4.4"),
        ("7", "Chẩn đoán LOSO toàn quốc", "exp_national_loso_diagnostic.py", "Bảng 4.2"),
        ("8", "Mô hình vùng ĐBSH và kiểm định ngoài", "exp_red_river_delta.py", "kết quả Chương 5"),
    ]
    table = "\n".join(
        f"  {idx} & {esc_tex(task)} & \\path{{{script}}} & {esc_tex(out)} \\\\"
        for idx, task, script, out in rows
    )
    content = r"""\documentclass[../DoAn.tex]{subfiles}
\begin{document}

Phụ lục này tóm tắt chuỗi script chính trong gói mã nguồn nộp kèm và môi trường để chạy lại. Mã nguồn được tổ chức theo bốn giai đoạn: thu thập dữ liệu (\path{01_collection}), xử lý và kiểm soát chất lượng (\path{02_processing}), xây dựng đặc trưng (\path{03_features}) và thí nghiệm mô hình (\path{04_experiments}). Cột Script ghi tên tệp chính hoặc thư mục chứa nhóm script tương ứng.

{\footnotesize
\begin{longtable}{r p{3.4cm} p{4.6cm} p{3.6cm}}
\caption{Chuỗi script chính của đồ án}\\
\toprule
\textbf{Bước} & \textbf{Mục đích} & \textbf{Script} & \textbf{Kết quả chính} \\
\midrule
\endfirsthead
\multicolumn{4}{l}{\textit{Bảng B.1 (tiếp)}}\\
\toprule
\textbf{Bước} & \textbf{Mục đích} & \textbf{Script} & \textbf{Kết quả chính} \\
\midrule
\endhead
""" + table + r"""
\bottomrule
\end{longtable}
}

Bước 1 và Bước 2 tạo dữ liệu đầu vào cho pipeline. Các script phân tích và thí nghiệm ở các bước sau sử dụng các bảng hợp nhất hoặc tệp xuất phù hợp với từng mục đích. Bộ lọc chất lượng PM2.5 trong \path{pm25_qc.py} được dùng chung cho các phân tích có dữ liệu PM2.5 mặt đất. Kết quả số và hình do mỗi script sinh ra là cơ sở cho các bảng và hình tương ứng trong Chương 4 và Chương 5; ví dụ Bước 7 tạo chẩn đoán toàn quốc ở Bảng 4.2, còn Bước 8 tạo kết quả mô hình vùng và kiểm định ngoài ở Chương 5.

\section{Môi trường tái lập}

Các kết quả được tạo bằng Python trên Windows, với các phiên bản chính:

\begin{itemize}
\item Python 3.10;
\item pandas 2.3.3;
\item numpy 2.2.6;
\item scikit-learn 1.7.2;
\item XGBoost 3.2.0;
\item matplotlib 3.10.6.
\end{itemize}

Tệp \path{requirements.txt} kèm theo gói mã nguồn liệt kê các gói Python cần thiết. Để chạy lại, cài môi trường bằng \verb|pip install -r requirements.txt| rồi chạy các script theo thứ tự trong Bảng B.1.

\end{document}
"""
    write(CHUONG / "Phu_luc_B.tex", content)


def main():
    appendix_a()
    appendix_b()


if __name__ == "__main__":
    main()
