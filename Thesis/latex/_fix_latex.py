"""Fix the three LaTeX issues: double chapters, template bib entries, appendix."""
import re, os
os.chdir("d:/Geo/vn-air-pollution-analysis-main/Air_quality/Thesis/latex")

# Fix 1: Remove \chapter{} and \label{chuongN} from subfiles
# (DoAn.tex already provides Vietnamese chapter titles)
names = {1: "1_Gioi_thieu", 2: "2_Co_so_ly_thuyet", 3: "3_De_xuat",
         4: "4_Phan_tich_ly_thuyet", 5: "5_Thuc_nghiem", 6: "6_Ket_luan"}
chap_re = re.compile(r"\\chapter\{[^}]*\}\s*\n?")
label_re = re.compile(r"\\label\{chuong\d+\}\s*\n?")
for n in range(1, 7):
    f = "Chuong/" + names[n] + ".tex"
    t = open(f, encoding="utf-8").read()
    t = chap_re.sub("", t)
    t = label_re.sub("", t)
    open(f, "w", encoding="utf-8").write(t)
    print(f"  {f}: removed \\chapter + \\label")

# Also fix front-matter subfiles
chaps_re = re.compile(r"\\chapter\*\{[^}]*\}\s*\n?")
addtoc_re = re.compile(r"\\addcontentsline\{toc\}\{chapter\}\{[^}]*\}\s*\n?")
for fname in ["Chuong/0_4_Tom_tat_noi_dung_eng.tex", "Chuong/0_2_Loi_cam_on.tex"]:
    t = open(fname, encoding="utf-8").read()
    t = chaps_re.sub("", t)
    t = addtoc_re.sub("", t)
    open(fname, "w", encoding="utf-8").write(t)
    print(f"  {fname}: removed \\chapter*")

# Fix 2: Remove template example bib entries
TEMPLATE_KEYS = {"harris2009cloud", "ashton2009internet", "scott2013sdn",
                 "hovy1993automated", "peterson2007computer", "NguyenThucHai",
                 "poesio2001discourse", "knott1996data", "BernersTim", "LectureA"}
bib = open("Danh_sach_tai_lieu_tham_khao.bib", encoding="utf-8").read()
entries = re.split(r"\n(?=@)", bib)
kept, removed = [], []
for e in entries:
    e = e.strip()
    if not e:
        continue
    m = re.match(r"@\w+\{([^,]+),", e)
    if m and m.group(1).strip() in TEMPLATE_KEYS:
        removed.append(m.group(1).strip())
    else:
        kept.append(e)
open("Danh_sach_tai_lieu_tham_khao.bib", "w", encoding="utf-8").write(
    "\n\n".join(kept) + "\n")
print(f"\n  bib: removed {len(removed)} template entries ({', '.join(removed)})")
print(f"  bib: kept {len(kept)} thesis entries")

# Fix 3: Replace appendix template content with thesis-relevant content
# Appendix A: station list + feature list
# Appendix B: removed (template use-case spec not relevant)
app_a = r"""\documentclass[../DoAn.tex]{subfiles}
\begin{document}

This appendix provides supplementary reference tables for the datasets and features used in this thesis.

\section{Station List}

The 37 regulatory (KK) stations retained after quality control are listed below with their coordinates, region, pollution tier, and mean PM2.5 concentration. The three removed stations (two Mekong Delta, one central) are excluded.

\begin{center}
\emph{[Table: see analysis/thesis\_audit/station\_selection\_final.csv --- to be inserted as a formatted longtable.]}
\end{center}

\section{Feature List}

The 66 features used in the definitive model (dart\_nn23) are listed below, grouped by source category, with their coverage percentage and a brief description.

\begin{center}
\emph{[Table: see analysis/thesis\_audit/feature\_list.csv --- to be inserted as a formatted longtable.]}
\end{center}

\section{Google Earth Engine Scripts}

Five custom GEE JavaScript scripts were developed to extract satellite data at each station's coordinates:

\begin{enumerate}
\item \textbf{MAIAC AOD} (MCD19A2 Collection 6.1): center pixel + 5$\times$5 grid statistics, daily.
\item \textbf{TROPOMI} (Sentinel-5P): NO$_2$, SO$_2$, CO, HCHO tropospheric columns, daily.
\item \textbf{MODIS LST} (MOD11A1/MYD11A1): day/night surface temperature, 5$\times$5 grid, daily.
\item \textbf{GEOS-CF / MERRA-2}: hourly PM2.5 and aerosol diagnostics at 0.25$^\circ$.
\item \textbf{GHAP / ACAG}: annual and monthly PM2.5 climatology at 1~km.
\end{enumerate}

The scripts are available in the project repository under \texttt{scripts/data/}.

\end{document}
"""

app_b = r"""\documentclass[../DoAn.tex]{subfiles}
\begin{document}

This appendix lists the experiment configurations and their results referenced throughout the thesis. The complete per-station result tables are available in the project repository under \texttt{analysis/thesis\_experiments/}.

\section{Experiment Configuration Summary}

\begin{center}
\emph{[Table: summary of all LOSO configurations tested --- config name, feature count, mean R$^2$, median R$^2$, source file. To be generated from the experiment catalog.]}
\end{center}

\end{document}
"""

open("Chuong/Phu_luc_A.tex", "w", encoding="utf-8").write(app_a)
open("Chuong/Phu_luc_B.tex", "w", encoding="utf-8").write(app_b)
print("\n  Appendix A+B: replaced with thesis content")

# Fix DoAn.tex appendix titles
doan = open("DoAn.tex", encoding="utf-8").read()
doan = doan.replace(r"\chapter{HƯỚNG DẪN VIẾT ĐATN}",
                    r"\chapter{DỮ LIỆU VÀ ĐẶC TRƯNG}")  # Data & Features
doan = doan.replace(r"\chapter{ĐẶC TẢ USE CASE}",
                    r"\chapter{CẤU HÌNH THÍ NGHIỆM}")  # Experiment Configs
open("DoAn.tex", "w", encoding="utf-8").write(doan)
print("  DoAn.tex: updated appendix titles")

# Also remove the template guidance text from Chapter 7 (reference notes)
ch7 = open("Chuong/7_Tai_lieu_tham_khao.tex", encoding="utf-8").read()
ch7_clean = r"""\documentclass[../DoAn.tex]{subfiles}
\begin{document}

% References are generated automatically from the .bib file via biblatex/biber.
% See \texttt{Danh\_sach\_tai\_lieu\_tham\_khao.bib} for the full reference list.

\end{document}
"""
open("Chuong/7_Tai_lieu_tham_khao.tex", "w", encoding="utf-8").write(ch7_clean)
print("  Ch7 reference notes: cleaned")

print("\nAll 3 fixes applied.")
