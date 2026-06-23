"""
Convert the markdown thesis into the SOICT/HUST LaTeX template (DoAn.tex).

Outputs into Thesis/latex/ following the REAL template structure:
  Chapter1.md -> Chuong/1_Gioi_thieu.tex          (\chapter{...})
  Chapter2.md -> Chuong/2_Co_so_ly_thuyet.tex
  Chapter3.md -> Chuong/3_De_xuat.tex
  Chapter4.md -> Chuong/4_Phan_tich_ly_thuyet.tex
  Chapter5.md -> Chuong/5_Thuc_nghiem.tex
  Chapter6.md -> Chuong/6_Ket_luan.tex
  FrontMatter abstract -> Chuong/0_4_Tom_tat_noi_dung_eng.tex
  FrontMatter acks     -> Chuong/0_2_Loi_cam_on.tex
  References.md -> Danh_sach_tai_lieu_tham_khao.bib  (+ in-text \cite{})
  figures copied into Hinh_ve/

Conventions taken from the template: \chapter/\section/\subsection, figures via
\includegraphics from Hinh_ve/ with \caption+\label, longtable+booktabs, biblatex.
"""
import os, re, shutil
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

SRC = "Thesis/latex"
TEX = "Thesis/latex"
CHUONG = os.path.join(TEX, "Chuong")
FIG = os.path.join(TEX, "Hinh_ve")
os.makedirs(FIG, exist_ok=True)

UNI = [("²", r"\textsuperscript{2}"), ("³", r"\textsuperscript{3}"),
       ("₂", r"\textsubscript{2}"), ("₀", r"\textsubscript{0}"),
       ("₁", r"\textsubscript{1}"), ("₃", r"\textsubscript{3}"),
       ("₅", r"\textsubscript{5}"), ("₄", r"\textsubscript{4}"),
       ("µ", r"$\mu$"), ("×", r"$\times$"), ("≈", r"$\approx$"), ("≥", r"$\geq$"),
       ("≤", r"$\leq$"), ("→", r"$\rightarrow$"), ("↑", r"$\uparrow$"),
       ("↓", r"$\downarrow$"), ("∼", r"$\sim$"), ("−", "-"), ("–", "--"),
       ("—", "---"), ("'", "'"), ("'", "'"), (""", "``"), (""", "''"),
       ("…", r"\ldots{}"), ("°", r"$^\circ$"), ("ρ", r"$\rho$"), ("σ", r"$\sigma$"),
       ("α", r"$\alpha$"), ("β", r"$\beta$"), ("≪", r"$\ll$"), ("≫", r"$\gg$")]

CITEKEYS = {}   # filled from References.md: "surname_year" -> bibkey


_LATEX_FRAGS = []  # accumulator for inline(); reset per call

def _protect(latex_str):
    """Register a raw LaTeX fragment and return a placeholder that esc() won't touch."""
    idx = len(_LATEX_FRAGS)
    _LATEX_FRAGS.append(latex_str)
    return f"\x02FRAG{idx}END\x03"

def _restore(t):
    """Replace all placeholders with their LaTeX fragments."""
    def _repl(m):
        idx = int(m.group(1))
        return _LATEX_FRAGS[idx] if idx < len(_LATEX_FRAGS) else m.group(0)
    t = re.sub(r"\x02FRAG(\d+)END\x03", _repl, t)
    return t

def esc(t):
    """Escape LaTeX special chars in plain markdown text.
    Assumes all LaTeX fragments have already been replaced with placeholders."""
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("$", r"\$"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        t = t.replace(a, b)
    return t


# cite_subst is now integrated into inline() via the protect/restore pattern


def inline(t):
    global _LATEX_FRAGS
    _LATEX_FRAGS = []  # reset per call

    # 1. Protect inline code spans (backtick)
    def _code(m):
        return _protect(r"\texttt{" + esc(m.group(1)) + "}")
    t = re.sub(r"`([^`]+)`", _code, t)

    # 2. Protect markdown links
    def _link(m):
        txt, url = m.group(1), m.group(2)
        if url.startswith("http"):
            safe_url = url.replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
            return _protect(r"\href{" + safe_url + "}{" + esc(txt) + "}")
        return esc(txt)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link, t)

    # 3. Protect unicode -> LaTeX substitutions
    for a, b in UNI:
        if a in t:
            t = t.replace(a, _protect(b))

    # 4. Protect author-year citations -> \cite{}
    def _cite_paren(m):
        body = m.group(1)
        parts = re.split(r";\s*", body)
        keys, ok = [], True
        for p in parts:
            mm = re.match(r"([A-Z][A-Za-zÀ-ỿ''\-]+)(?:\s+et\s+al\.?)?,?\s*(\d{4})", p.strip())
            if mm:
                k = CITEKEYS.get(f"{mm.group(1).lower()}_{mm.group(2)}")
                if k:
                    keys.append(k)
                else:
                    ok = False; break
            else:
                ok = False; break
        if ok and keys:
            return _protect(r"\cite{" + ",".join(keys) + "}")
        return m.group(0)
    t = re.sub(r"\(([A-Z][^()]*?\d{4}[^()]*?)\)", _cite_paren, t)

    def _cite_text(m):
        k = CITEKEYS.get(f"{m.group(1).lower()}_{m.group(2)}")
        if k:
            prefix = m.group(1) + (" et al. " if "et al" in m.group(0) else " ")
            return prefix + _protect(r"\cite{" + k + "}")
        return m.group(0)
    t = re.sub(r"([A-Z][A-Za-zÀ-ỿ''\-]+)\s+et\s+al\.?\s*\((\d{4})\)", _cite_text, t)

    # 5. NOW escape remaining markdown text (placeholders are safe)
    t = esc(t)

    # 6. Bold / italic (on escaped text; the ** / * markers survived esc)
    t = re.sub(r"\*\*([^*]+)\*\*", lambda m: _protect(r"\textbf{" + m.group(1) + "}"), t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: _protect(r"\emph{" + m.group(1) + "}"), t)

    # 7. Restore all protected LaTeX fragments
    t = _restore(t)
    return t


def strip_num(s):
    s = re.sub(r"^CHAPTER\s+\d+\s*:\s*", "", s, flags=re.I)
    s = re.sub(r"^\d+(\.\d+)*\s+", "", s)
    return s.strip()


def titlecase_kw(s):
    return s if s.isupper() is False else s.title()


def conv_table(block, caption=None):
    rows = [r for r in block if r.strip().startswith("|")]
    def cells(r):
        r = r.strip().strip("|")
        return [c.strip() for c in r.split("|")]
    header = cells(rows[0]); data = [cells(r) for r in rows[2:]]
    nc = len(header)
    out = [r"\begin{longtable}{" + "l" * nc + "}"]
    if caption:
        out.append(r"\caption{" + inline(caption) + r"}\\")
    out.append(r"\toprule")
    out.append(" & ".join(r"\textbf{" + inline(h) + "}" for h in header) + r" \\")
    out.append(r"\midrule\endhead")
    for d in data:
        d = (d + [""] * nc)[:nc]
        out.append(" & ".join(inline(c) for c in d) + r" \\")
    out += [r"\bottomrule", r"\end{longtable}"]
    return "{\\small\n" + "\n".join(out) + "\n}"


def conv_body(md):
    lines = md.split("\n"); out = []; i = 0; pend = None; fign = 0
    while i < len(lines):
        s = lines[i].strip()
        mcap = re.match(r"^\*(?:Figure|Table)\s*[\d.]*:?\s*(.*?)\*$", s)
        if mcap:
            pend = mcap.group(1).strip(); i += 1; continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|", lines[i+1]):
            blk = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                blk.append(lines[i]); i += 1
            out += [conv_table(blk, pend), ""]; pend = None; continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            fign += 1
            cap = re.sub(r"^Figure\s+[\d.]+:\s*", "", m.group(1)) or (pend or ""); pend = None
            base = os.path.basename(m.group(2))
            out += [r"\begin{figure}[ht]", r"\centering",
                    r"\includegraphics[width=0.85\textwidth]{" + base + "}"]
            if cap:
                out.append(r"\caption{" + inline(cap) + "}")
            out += [r"\end{figure}", ""]; i += 1; continue
        if s.startswith("#"):
            h = len(s) - len(s.lstrip("#")); title = strip_num(s.lstrip("#").strip())
            if h == 1:
                i += 1; continue
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}.get(h, "paragraph")
            out.append("\\" + cmd + "{" + inline(title) + "}"); i += 1; continue
        if s.startswith(">"):
            q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(lines[i].strip()[1:].strip()); i += 1
            out += [r"\begin{quote}", inline(" ".join(q)), r"\end{quote}"]; continue
        if re.match(r"^[-*]\s+", s):
            out.append(r"\begin{itemize}")
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                out.append(r"\item " + inline(re.sub(r"^[-*]\s+", "", lines[i].strip()))); i += 1
            out.append(r"\end{itemize}"); continue
        if re.match(r"^\d+\.\s+", s):
            out.append(r"\begin{enumerate}")
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                out.append(r"\item " + inline(re.sub(r"^\d+\.\s+", "", lines[i].strip()))); i += 1
            out.append(r"\end{enumerate}"); continue
        if s in ("", "---"):
            if s == "" and not (pend):
                out.append("")
            i += 1; continue
        out.append(inline(s)); i += 1
    return "\n".join(out)


def chap_title(md):
    for ln in md.split("\n"):
        if ln.strip().startswith("# "):
            return strip_num(ln.strip()[2:])
    return "Chapter"


def section_of(md, name):
    m = re.search(r"^##\s+" + name + r"\s*$(.*?)(?=^##\s|\Z)", md, flags=re.M | re.S)
    return m.group(1).strip() if m else ""


# ---------- build citation keys from References.md ----------
def build_bib():
    refmd = open(os.path.join(SRC, "References.md"), encoding="utf-8").read()
    refmd = re.sub(r"<!--.*?-->", "", refmd, flags=re.S)
    bib = []
    for para in re.split(r"\n\s*\n", refmd):
        p = " ".join(para.strip().split("\n"))
        if not p or p.startswith("#"):
            continue
        m = re.match(r"^(.+?)\s*\((\d{4}[a-z]?)\)\.\s*(.*)$", p)
        if not m:
            continue
        authors_raw, year, rest = m.group(1).strip(), m.group(2), m.group(3).strip()
        sur = re.match(r"([A-Za-zÀ-ỹ''\-]+)", authors_raw).group(1)
        key = re.sub(r"[^a-z0-9]", "", sur.lower()) + re.sub(r"[^0-9]", "", year)
        base = key; n = 1
        while key in [b[0] for b in bib]:
            n += 1; key = base + chr(ord('a') + n - 1)
        CITEKEYS[f"{sur.lower()}_{re.sub(chr(91)+'^0-9'+chr(93),'',year)}"] = key
        # title = up to first '. ' after rest; remainder = source
        tm = re.match(r"(.*?[.?!])\s+(.*)$", rest)
        title = (tm.group(1) if tm else rest).rstrip(".")
        source = (tm.group(2) if tm else "").strip()
        # authors -> "A and B and C"
        a = authors_raw.replace("&", "").replace(" and ", ", ")
        names = re.findall(r"[A-Z][A-Za-zÀ-ỹ''\-]+,\s*[A-Z]\.(?:\s*[A-Z]\.)*", a)
        author = " and ".join(n.strip() for n in names) if names else authors_raw
        doi = ""
        dm = re.search(r"https?://\S+|doi\.org/\S+|10\.\d{4,}/\S+", source)
        if dm:
            doi = dm.group(0).rstrip(". ")
            source = source.replace(dm.group(0), "").strip().rstrip(". ")
        def b(x):  # brace-safe, markdown-free field value
            x = x.replace("{", "").replace("}", "").replace("*", "")
            x = x.replace("–", "--").replace("—", "--").replace("'", "'")
            return x.strip().rstrip(". ")
        typ = "article" if re.search(r"\d+\(\d+\)|,\s*\d+,|Journal|Science|Nature|Environment|Atmos", source) else "misc"
        e = [f"@{typ}{{{key},",
             f"  author = {{{b(author)}}},",
             f"  title = {{{b(title)}}},",
             f"  year = {{{re.sub(r'[^0-9]','',year)}}},"]
        if source:
            e.append(f"  howpublished = {{{b(source)}}}," if typ == "misc" else f"  journal = {{{b(source)}}},")
        if doi:
            e.append(f"  url = {{{doi}}},")
        e.append("}")
        bib.append((key, "\n".join(e)))
    bibpath = os.path.join(TEX, "Danh_sach_tai_lieu_tham_khao.bib")
    # Write ONLY the thesis references (no appending to avoid duplicates)
    with open(bibpath, "w", encoding="utf-8") as f:
        f.write("% Auto-generated from Thesis/References.md\n"
                + "\n\n".join(b[1] for b in bib) + "\n")
    print(f"  bib: {len(bib)} entries written (clean), {len(CITEKEYS)} citation keys")


def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  wrote", os.path.relpath(path, TEX))


build_bib()   # must run before chapters (fills CITEKEYS for cite_subst)

print("Chapters ->")
CHMAP = {1: "1_Gioi_thieu", 2: "2_Co_so_ly_thuyet", 3: "3_De_xuat",
         4: "4_Phan_tich_ly_thuyet", 5: "5_Thuc_nghiem", 6: "6_Ket_luan"}
for n in range(1, 7):
    md = open(os.path.join(SRC, f"Chapter{n}.md"), encoding="utf-8").read()
    tex = ("\\chapter{" + chap_title(md) + "}\n\\label{chuong" + str(n) + "}\n\n"
           + conv_body(md) + "\n")
    write(os.path.join(CHUONG, CHMAP[n] + ".tex"), tex)

print("Front matter ->")
fm = open(os.path.join(SRC, "FrontMatter.md"), encoding="utf-8").read()
abs_txt = section_of(fm, "Abstract")
ack_txt = section_of(fm, "Acknowledgements")
write(os.path.join(CHUONG, "0_4_Tom_tat_noi_dung_eng.tex"),
      "\\chapter*{Abstract}\n\\addcontentsline{toc}{chapter}{Abstract}\n\n"
      + conv_body(abs_txt) + "\n")
write(os.path.join(CHUONG, "0_2_Loi_cam_on.tex"),
      "\\chapter*{Acknowledgements}\n\\addcontentsline{toc}{chapter}{Acknowledgements}\n\n"
      + conv_body(ack_txt) + "\n")

print("Figures ->")
figs = ["Thesis/figures/fig_5_3_station_r2_map.png", "Thesis/figures/fig_3_pipeline.png"]
sp = "analysis/thesis_experiments/satellite_products"
for fn in ["geoscf_diurnal_cycle.png", "geoscf_bias_pattern.png",
           "geoscf_scatter_representative.png", "merra2_scatter_representative.png",
           "merra2_species_by_region.png", "ghap_station_ranking.png"]:
    figs.append(os.path.join(sp, fn))
for f in figs:
    if os.path.exists(f):
        shutil.copy(f, os.path.join(FIG, os.path.basename(f))); print("  fig", os.path.basename(f))
    else:
        print("  MISSING", f)
print("DONE")
