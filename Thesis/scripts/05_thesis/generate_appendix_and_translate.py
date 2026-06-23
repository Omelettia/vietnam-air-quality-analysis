"""
Generate appendix tables from CSVs, write Vietnamese abstract,
and translate all chapter .tex files to Vietnamese.
"""
import os, re
import pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

TEX = "Thesis/latex"
EXP = "analysis/thesis_experiments"

def esc_tex(s):
    s = str(s)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s

# ============================================================
# 1. Generate Appendix A: station table + feature table
# ============================================================
print("=== Generating Appendix A ===")
stn = pd.read_csv("analysis/thesis_audit/station_selection_final.csv", dtype={"stationId": str})
# Filter to the 37 retained (quality_flag == 'pass' and not in EXCLUDE)
EXCLUDE = {"31616865099255512061948816121", "30991938797551443885460120607", "29098319146067624969113973428"}
stn_kept = stn[(stn["quality_flag"] == "pass") & (~stn["stationId"].isin(EXCLUDE))].copy()
stn_kept = stn_kept.sort_values(["region", "tier", "station_name"])

# Build station longtable
stn_rows = []
for _, r in stn_kept.iterrows():
    name = esc_tex(r["station_name"][:50])
    stn_rows.append(f"  {name} & {r['lat']:.2f} & {r['lon']:.3f} & {r['region']} & t{r['tier']} & {r['pm25_coverage']*100:.0f}\\% \\\\")

feat = pd.read_csv("analysis/thesis_audit/feature_list.csv")
feat_rows = []
for _, r in feat.iterrows():
    feat_rows.append(f"  {esc_tex(r['feature'])} & {esc_tex(r['type'])} & {esc_tex(r['source'])} & {r['coverage_pct']:.0f}\\% \\\\")

app_a = r"""\documentclass[../DoAn.tex]{subfiles}
\begin{document}

Phụ lục này cung cấp các bảng tham chiếu bổ sung cho bộ dữ liệu và đặc trưng được sử dụng trong luận văn.

\section{Danh sách trạm quan trắc}

Bảng dưới đây liệt kê """ + str(len(stn_kept)) + r""" trạm quan trắc chất lượng không khí (KK) được giữ lại sau kiểm soát chất lượng, bao gồm tọa độ, vùng, mức độ ô nhiễm và tỷ lệ phủ dữ liệu PM2.5.

{\small
\begin{longtable}{p{5.5cm}rrllr}
\caption{Danh sách 37 trạm KK được sử dụng trong mô hình}\\
\toprule
\textbf{Tên trạm} & \textbf{Vĩ độ} & \textbf{Kinh độ} & \textbf{Vùng} & \textbf{Tier} & \textbf{Phủ PM2.5} \\
\midrule\endhead
""" + "\n".join(stn_rows) + r"""
\bottomrule
\end{longtable}
}

\section{Danh sách đặc trưng}

Bảng dưới đây liệt kê """ + str(len(feat)) + r""" đặc trưng được sử dụng trong mô hình, phân loại theo nguồn dữ liệu.

{\small
\begin{longtable}{p{4cm}lp{3cm}r}
\caption{Danh sách đặc trưng với nguồn và tỷ lệ phủ}\\
\toprule
\textbf{Đặc trưng} & \textbf{Loại} & \textbf{Nguồn} & \textbf{Phủ (\%)} \\
\midrule\endhead
""" + "\n".join(feat_rows) + r"""
\bottomrule
\end{longtable}
}

\section{Google Earth Engine Scripts}

Năm script GEE JavaScript tùy chỉnh đã được phát triển để trích xuất dữ liệu vệ tinh tại tọa độ mỗi trạm:

\begin{enumerate}
\item \textbf{MAIAC AOD} (MCD19A2 Collection 6.1): pixel trung tâm + thống kê lưới 5$\times$5, hàng ngày.
\item \textbf{TROPOMI} (Sentinel-5P): NO$_2$, SO$_2$, CO, HCHO cột tầng đối lưu, hàng ngày.
\item \textbf{MODIS LST} (MOD11A1/MYD11A1): nhiệt độ bề mặt ngày/đêm, lưới 5$\times$5, hàng ngày.
\item \textbf{GEOS-CF / MERRA-2}: PM2.5 và chẩn đoán sol khí theo giờ ở độ phân giải 0.25$^\circ$.
\item \textbf{GHAP}: khí hậu PM2.5 hàng năm và hàng tháng ở độ phân giải 1~km (Wei et al., 2023).
\end{enumerate}

Các script có sẵn trong kho mã nguồn tại \texttt{scripts/data/}.

\end{document}
"""

with open(os.path.join(TEX, "Chuong/Phu_luc_A.tex"), "w", encoding="utf-8") as f:
    f.write(app_a)
print(f"  Appendix A: {len(stn_kept)} stations, {len(feat)} features")

# ============================================================
# 2. Generate Appendix B: experiment config summary
# ============================================================
print("=== Generating Appendix B ===")
import glob
exp_files = sorted(glob.glob(os.path.join(EXP, "*.csv")))
exp_rows = []
for fp in exp_files:
    try:
        d = pd.read_csv(fp, nrows=3)
        if "config" not in d.columns:
            continue
        r2col = next((c for c in d.columns if "r2" in c.lower() and "hour" in c.lower()), None)
        if not r2col:
            r2col = next((c for c in d.columns if c.lower() == "r2"), None)
        if not r2col:
            continue
        d = pd.read_csv(fp)
        for cfg, g in d.groupby("config"):
            vals = g[r2col].dropna()
            if len(vals) == 0:
                continue
            exp_rows.append({
                "file": os.path.basename(fp),
                "config": str(cfg)[:25],
                "n": len(vals),
                "mean_r2": vals.mean(),
                "median_r2": vals.median(),
            })
    except Exception:
        continue

exp_df = pd.DataFrame(exp_rows).sort_values("mean_r2", ascending=False)
# Top configs
top = exp_df.head(40)

table_rows = []
for _, r in top.iterrows():
    table_rows.append(
        f"  {esc_tex(r['config'])} & {esc_tex(r['file'][:30])} & {r['n']} & "
        f"{r['mean_r2']:+.3f} & {r['median_r2']:+.3f} \\\\"
    )

app_b = r"""\documentclass[../DoAn.tex]{subfiles}
\begin{document}

Phụ lục này liệt kê các cấu hình thí nghiệm và kết quả tương ứng. Bảng dưới đây tóm tắt 40 cấu hình có R$^2$ trung bình cao nhất trong tất cả các thí nghiệm LOSO.

{\small
\begin{longtable}{p{3.5cm}p{3.5cm}rr r}
\caption{Tóm tắt cấu hình thí nghiệm (top 40 theo R$^2$ trung bình)}\\
\toprule
\textbf{Cấu hình} & \textbf{Tệp nguồn} & \textbf{n} & \textbf{R$^2$ TB} & \textbf{R$^2$ Median} \\
\midrule\endhead
""" + "\n".join(table_rows) + r"""
\bottomrule
\end{longtable}
}

Bảng kết quả đầy đủ theo từng trạm có sẵn trong kho mã nguồn tại \texttt{analysis/thesis\_experiments/}.

\end{document}
"""

with open(os.path.join(TEX, "Chuong/Phu_luc_B.tex"), "w", encoding="utf-8") as f:
    f.write(app_b)
print(f"  Appendix B: {len(top)} config rows from {len(exp_df)} total")

# ============================================================
# 3. Write Vietnamese abstract (0_3_Tom_tat_noi_dung.tex)
# ============================================================
print("=== Writing Vietnamese abstract ===")
vn_abstract = r"""\documentclass[../DoAn.tex]{subfiles}
\begin{document}

\begin{center}
    \Large{\textbf{TÓM TẮT ĐỒ ÁN TỐT NGHIỆP}}\\
\end{center}
\vspace{0.5cm}

Ô nhiễm không khí là một trong những thách thức nghiêm trọng nhất về môi trường và sức khỏe cộng đồng tại Việt Nam, tuy nhiên mạng lưới quan trắc PM2.5 mặt đất của quốc gia vẫn còn thưa thớt --- chỉ khoảng 40 trạm quy chuẩn trên diện tích 331.000 km\textsuperscript{2} --- khiến phần lớn dân số thiếu thông tin về chất lượng không khí địa phương. Học máy kết hợp dữ liệu vệ tinh đã được đề xuất như một giải pháp, với các nghiên cứu tại các khu vực có mạng lưới dày đặc thường báo cáo hệ số xác định (R\textsuperscript{2}) trên 0,8. Luận văn này đặt câu hỏi liệu hiệu suất tương đương có thể đạt được cho mạng lưới thưa thớt, nhiệt đới của Việt Nam hay không, đồng thời đối mặt với một vấn đề phương pháp hệ thống: sự khác biệt giữa đánh giá chéo ngẫu nhiên được sử dụng trong hầu hết các nghiên cứu và đánh giá chéo không gian phản ánh nhiệm vụ thực tế của dự đoán tại các vị trí chưa được quan trắc.

Mô hình XGBoost-DART với 66 đặc trưng được huấn luyện trên 37 trạm quy chuẩn, sử dụng độ sâu quang học sol khí từ MODIS và Himawari, các khí vết từ TROPOMI, khí tượng ERA5, sản phẩm mô hình vận chuyển hóa học (CTM) và các yếu tố sử dụng đất. Kết quả trung tâm là chuỗi ba con số: R\textsuperscript{2} xấp xỉ 0,80 với đánh giá chéo ngẫu nhiên giảm xuống xấp xỉ 0,20 với đánh giá LOSO (bỏ-một-trạm-ra), và xuống trung vị khoảng 0,04 cho mỗi trạm khi thông tin oracle bị rút lại. Phân tầng theo mức độ ô nhiễm (T4F) là đòn bẩy hiệu suất lớn nhất, nhưng mang theo sự phụ thuộc vòng tròn --- gán tầng cho trạm đòi hỏi biết trung bình PM2.5, chính là đại lượng cần dự đoán --- và bảy họ phương pháp đại diện đều thất bại trong việc giải quyết nó. Các sản phẩm CTM và vệ tinh toàn cầu thất bại theo các cách riêng biệt: GEOS-CF có chu kỳ ngày đêm đảo ngược và sai lệch trên 200\%, MERRA-2 đạt chỉ số IOA chỉ 0,39, và GHAP chỉ đạt khoảng 50\% độ phù hợp phân tầng. Kiểm chứng ngoại tại cho kết quả tốt trong phạm vi phủ sóng dày đặc, với trung vị R\textsuperscript{2} = 0,53 cho cảm biến chi phí thấp và R\textsuperscript{2} = 0,68 tại Đại sứ quán Hoa Kỳ, giảm dần theo khoảng cách. Ràng buộc chính là mật độ trạm chứ không phải năng lực thuật toán.

\vspace{0.5cm}
\textbf{Từ khóa:} PM2.5, viễn thám, độ sâu quang học sol khí, học máy, đánh giá chéo không gian, Việt Nam, cảm biến chi phí thấp

\vspace{1cm}
Sinh viên thực hiện\\
\vspace{1cm}
[Họ và tên sinh viên]

\end{document}
"""

with open(os.path.join(TEX, "Chuong/0_3_Tom_tat_noi_dung.tex"), "w", encoding="utf-8") as f:
    f.write(vn_abstract)
print("  Vietnamese abstract written")

# ============================================================
# 4. Fix acknowledgements (remove placeholder markers)
# ============================================================
print("=== Fixing acknowledgements ===")
ack = r"""\documentclass[../DoAn.tex]{subfiles}
\begin{document}

Tác giả xin gửi lời cảm ơn chân thành đến giáo viên hướng dẫn và hội đồng chấm luận văn vì sự hướng dẫn tận tình, đội ngũ cán bộ trường và khoa vì sự hỗ trợ trong suốt quá trình thực hiện công trình này, Trung tâm Quan trắc Môi trường Việt Nam và các đơn vị vận hành mạng lưới cảm biến chi phí thấp vì dữ liệu quan trắc mặt đất làm nền tảng cho nghiên cứu, cùng gia đình, bạn bè và đồng nghiệp đã động viên trong suốt quá trình hoàn thành.

\vspace{1cm}
Sinh viên thực hiện\\
\vspace{1cm}
[Họ và tên sinh viên]

\end{document}
"""

with open(os.path.join(TEX, "Chuong/0_2_Loi_cam_on.tex"), "w", encoding="utf-8") as f:
    f.write(ack)
print("  Acknowledgements written in Vietnamese")

print("\nDone. Placeholders replaced with real content + Vietnamese.")
