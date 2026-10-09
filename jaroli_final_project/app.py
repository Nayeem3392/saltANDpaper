"""
MorphoVision Pro - salt & pepper noise removal lab
Run with:  streamlit run app.py
"""
import base64
import datetime
import io
import os
import zipfile

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="MorphoVision Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_SIDE = 1600  # longest image side used for processing (keeps the app fast)

# ---------------------------------------------------------------- design system
# Ink #0D1A1E  Bench #14262B  Rule #2A4547  Bone #E8E4DA
# Mist #8FA7A3  Safelight #FFB347  Lumen #5EE0C8  Coral #FF7A66
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap');

:root{
  --ink:#0D1A1E; --bench:#14262B; --bench2:#1B343A; --rule:#2A4547;
  --bone:#E8E4DA; --mist:#8FA7A3; --safe:#FFB347; --lumen:#5EE0C8; --coral:#FF7A66;
}

#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] {display:none !important;}
[data-testid="stHeader"] {background:transparent !important;}

/* page background: darkroom glow + measurement graticule */
.stApp{
  font-family:'Instrument Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  color:var(--bone);
  background:
    radial-gradient(1100px 560px at 88% -8%, rgba(255,179,71,.11), transparent 60%),
    radial-gradient(900px 600px at -8% 8%, rgba(94,224,200,.08), transparent 60%),
    linear-gradient(rgba(232,228,218,.035) 1px, transparent 1px) 0 0 / 48px 48px,
    linear-gradient(90deg, rgba(232,228,218,.035) 1px, transparent 1px) 0 0 / 48px 48px,
    var(--ink);
  background-attachment:fixed;
}
.stApp p, .stApp label, .stApp li, .stApp input, .stApp button, .stApp textarea{
  font-family:'Instrument Sans',system-ui,sans-serif;
}
.block-container{max-width:1240px; padding:1rem 2rem 4rem;}

/* top bar */
.topbar{display:flex; align-items:center; justify-content:space-between;
  padding:14px 0 16px; border-bottom:1px solid var(--rule); margin-bottom:8px;}
.brand{display:flex; align-items:center; gap:12px;
  font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.25rem; letter-spacing:-.01em;}
.brand svg{width:28px; height:28px;}
.topbar-meta{color:var(--mist); font-size:.9rem;}

/* hero */
.hero-copy{padding-top:34px;}
.hero-title{
  font-family:'Bricolage Grotesque',sans-serif; font-weight:800;
  font-size:clamp(2.2rem,4.4vw,3.5rem); line-height:1.04; letter-spacing:-.03em;
  margin:0 0 18px !important; padding:0 !important; color:var(--bone);
}
.hero-text{color:#B7C6C2; font-size:1.08rem; line-height:1.6; max-width:34rem; margin:0 0 18px;}
.hero-hint{color:var(--mist); font-size:.9rem; border-left:2px solid var(--safe); padding-left:12px; max-width:30rem;}

/* process strip (a real sequence, so it is numbered) */
.steps{display:grid; grid-template-columns:repeat(3,1fr); gap:0; margin:34px 0 10px;
  border:1px solid var(--rule); border-radius:8px; background:var(--bench); overflow:hidden;}
.step{padding:20px 24px; border-right:1px solid var(--rule);}
.step:last-child{border-right:0;}
.step-n{display:inline-flex; width:24px; height:24px; align-items:center; justify-content:center;
  border:1px solid var(--safe); color:var(--safe); border-radius:50%; font-size:.78rem; font-weight:600; margin-bottom:10px;}
.step-t{font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.1rem; margin-bottom:4px;}
.step-d{color:var(--mist); font-size:.93rem; line-height:1.5;}

/* section headings */
.sec{margin:42px 0 14px;}
.sec-t{font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.7rem; letter-spacing:-.02em; margin:0;}
.sec-d{color:var(--mist); margin-top:4px; font-size:.97rem;}
.chip{display:inline-block; margin-left:12px; padding:3px 10px; border:1px solid var(--rule);
  border-radius:20px; color:var(--mist); font-size:.82rem; font-weight:500; vertical-align:middle;}

/* readout panel */
.readout{display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--rule);
  border-radius:8px; background:var(--bench); overflow:hidden; margin:18px 0 6px;}
.cell{padding:18px 22px; border-right:1px solid var(--rule);}
.cell:last-child{border-right:0;}
.val{font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:2rem; line-height:1.1;
  font-variant-numeric:tabular-nums; color:var(--bone);}
.val small{font-size:.9rem; color:var(--mist); font-weight:500; margin-left:4px;}
.lab{color:var(--mist); font-size:.88rem; margin-top:6px;}
.delta{font-size:.84rem; margin-top:8px; font-weight:600; color:var(--mist);}
.delta.up{color:var(--lumen);} .delta.down{color:var(--coral);}
@media (max-width:820px){
  .readout{grid-template-columns:repeat(2,1fr);} .cell:nth-child(2){border-right:0;}
  .cell:nth-child(-n+2){border-bottom:1px solid var(--rule);}
  .steps{grid-template-columns:1fr;} .step{border-right:0; border-bottom:1px solid var(--rule);}
}

/* sidebar */
[data-testid="stSidebar"]{background:var(--bench) !important; border-right:1px solid var(--rule);}
.side-brand{display:flex; align-items:center; gap:10px; font-family:'Bricolage Grotesque',sans-serif;
  font-weight:800; font-size:1.1rem; padding:4px 0 6px;}
.side-brand svg{width:24px; height:24px;}
.side-note{color:var(--mist); font-size:.85rem; margin-bottom:6px;}
.side-h{font-weight:600; color:var(--bone); font-size:.95rem; margin:22px 0 4px;
  padding-bottom:6px; border-bottom:1px solid var(--rule);}

/* widgets */
.stButton>button, .stDownloadButton>button{
  background:var(--bench2); color:var(--bone); border:1px solid var(--rule);
  border-radius:6px; font-weight:600; padding:.62rem 1rem; transition:border-color .15s, color .15s;}
.stButton>button:hover, .stDownloadButton>button:hover{border-color:var(--safe); color:var(--safe); background:var(--bench2);}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"]{
  background:var(--safe) !important; color:#1A1204 !important; border:1px solid var(--safe) !important;}
.stDownloadButton>button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{
  background:#FFC470 !important; color:#1A1204 !important;}
[data-testid="stFileUploaderDropzone"]{
  background:var(--bench) !important; border:1.5px dashed var(--rule) !important; border-radius:8px !important; padding:28px 20px;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--safe) !important;}
.stTabs [data-baseweb="tab-list"]{gap:6px; border-bottom:1px solid var(--rule);}
.stTabs [data-baseweb="tab"]{color:var(--mist); font-weight:600; padding:10px 16px;}
.stTabs [aria-selected="true"]{color:var(--bone) !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--safe) !important; height:3px;}
.stTabs [data-baseweb="tab-border"]{background:transparent !important;}
[data-testid="stImage"] img{border-radius:6px; border:1px solid var(--rule);}
[data-testid="stImageCaption"], .stCaption{color:var(--mist) !important;}
[data-testid="stCameraInput"] > div{border-radius:8px;}
hr{border:none !important; border-top:1px solid var(--rule) !important; margin:28px 0 !important;}

.footer{margin-top:56px; padding-top:18px; border-top:1px solid var(--rule);
  color:var(--mist); font-size:.88rem; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;}

::-webkit-scrollbar{width:8px; height:8px;}
::-webkit-scrollbar-thumb{background:var(--rule); border-radius:4px;}
</style>
"""

LOGO = """
<svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<circle cx="16" cy="16" r="12" stroke="#FFB347" stroke-width="2"/>
<circle cx="16" cy="16" r="5" stroke="#5EE0C8" stroke-width="2"/>
<path d="M16 1v7M16 24v7M1 16h7M24 16h7" stroke="#E8E4DA" stroke-width="1.5" stroke-linecap="round"/>
</svg>
"""


def html(s: str):
    """Render raw HTML through markdown safely (strips indentation and blank lines)."""
    s = "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())
    st.markdown(s, unsafe_allow_html=True)


def stretch(fn, *args, **kwargs):
    """Call a Streamlit widget so it fills its container on old and new versions."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}):
        try:
            return fn(*args, **kwargs, **extra)
        except Exception:
            continue
    return fn(*args, **kwargs)


def embed(page: str, height: int):
    """Embed an HTML/JS snippet. Uses st.iframe on new Streamlit, components.html on old."""
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
    else:
        components.html(page, height=height)


html(CSS)

# ---------------------------------------------------------------- image functions
def load_gray(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("unreadable image")
    h, w = img.shape
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def add_noise(image: np.ndarray, density: float, seed: int) -> np.ndarray:
    """Salt & pepper noise. `density` is the share of corrupted pixels (half salt, half pepper)."""
    rng = np.random.default_rng(int(seed))
    r = rng.random(image.shape)
    noisy = image.copy()
    noisy[r < density / 2] = 255
    noisy[(r >= density / 2) & (r < density)] = 0
    return noisy


SHAPES = {"Rectangle": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}


def morphological_clean(image, ksize, shape_choice):
    kernel = cv2.getStructuringElement(SHAPES[shape_choice], (ksize, ksize))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def apply_filter(image, algo, ksize, shape_choice):
    if algo.startswith("Median"):
        return cv2.medianBlur(image, ksize)
    if algo.startswith("Hybrid"):
        return morphological_clean(cv2.medianBlur(image, ksize), ksize, shape_choice)
    return morphological_clean(image, ksize, shape_choice)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a, b):
    """Standard windowed SSIM (11x11 gaussian, sigma 1.5)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu1, mu2 = blur(a), blur(b)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = blur(a * a) - mu1_sq
    s2 = blur(b * b) - mu2_sq
    s12 = blur(a * b) - mu12
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return float(m.mean())


def edge_match(original, other):
    e1 = cv2.Canny(original, 50, 150) > 0
    e2 = cv2.Canny(other, 50, 150) > 0
    union = np.logical_or(e1, e2).sum()
    return float(np.logical_and(e1, e2).sum() / union) if union else 0.0


def to_data_uri(img, max_w=1000):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def encode(img, fmt):
    if fmt == "JPG":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


@st.cache_data(show_spinner=False, max_entries=6)
def run_pipeline(file_bytes, density, seed, algo, ksize, shape_choice):
    original = load_gray(file_bytes)
    noisy = add_noise(original, density, seed)
    processed = apply_filter(noisy, algo, ksize, shape_choice)

    diff = cv2.absdiff(original, processed)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(cv2.cvtColor(original, cv2.COLOR_GRAY2BGR), 0.55, heat, 0.45, 0)

    corrupted = noisy != original
    n_bad = int(corrupted.sum())
    if n_bad:
        err = np.abs(processed.astype(np.int16) - original.astype(np.int16))
        restored = float((err[corrupted] <= 12).mean() * 100)
    else:
        restored = None

    metrics = {
        "psnr_noisy": psnr(original, noisy),
        "psnr": psnr(original, processed),
        "ssim_noisy": ssim(original, noisy),
        "ssim": ssim(original, processed),
        "edge_noisy": edge_match(original, noisy),
        "edge": edge_match(original, processed),
        "restored": restored,
        "corrupted": n_bad,
    }
    return {
        "original": original, "noisy": noisy, "processed": processed,
        "diff": diff, "heat": heat, "overlay": overlay, "metrics": metrics,
        "uri_original": to_data_uri(original),
        "uri_noisy": to_data_uri(noisy),
        "uri_processed": to_data_uri(processed),
    }


def make_strip(original, noisy, processed, labels):
    h, w = original.shape
    scale = max(0.6, w / 900)
    band_h = int(52 * scale)
    tiles = []
    for label, img in zip(labels, (original, noisy, processed)):
        band = np.full((band_h, w, 3), (30, 26, 13), np.uint8)  # ink, BGR
        cv2.putText(band, label, (int(16 * scale), int(band_h * 0.68)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (218, 228, 232), max(1, int(2 * scale)), cv2.LINE_AA)
        tiles.append(np.vstack([band, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)]))
    gap = np.full((tiles[0].shape[0], int(6 * scale), 3), (30, 26, 13), np.uint8)
    return np.hstack([tiles[0], gap, tiles[1], gap, tiles[2]])


def build_pdf(r, params):
    from fpdf import FPDF  # pip install fpdf2

    m = r["metrics"]
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 14, "MorphoVision Pro - analysis report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90, 100, 100)
    pdf.cell(0, 8, f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}   |   {params['name']}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    def row(k, v, bold=False):
        pdf.set_font("Helvetica", "B" if bold else "", 12)
        pdf.set_text_color(30, 40, 40)
        pdf.cell(100, 9, k, border="B")
        pdf.cell(0, 9, v, border="B", align="R", new_x="LMARGIN", new_y="NEXT")

    row("Setting", "Value", True)
    row("Noise density", f"{params['density'] * 100:.0f}% (seed {params['seed']})")
    row("Algorithm", params["algo"])
    row("Kernel size / element", f"{params['ksize']}x{params['ksize']} / {params['shape']}")
    pdf.ln(4)
    row("Measure", "Noisy -> Filtered", True)
    row("PSNR (dB)", f"{m['psnr_noisy']:.2f} -> {m['psnr']:.2f}")
    row("SSIM", f"{m['ssim_noisy']:.3f} -> {m['ssim']:.3f}")
    row("Edge match (Canny overlap)", f"{m['edge_noisy']:.3f} -> {m['edge']:.3f}")
    row("Corrupted pixels restored", "n/a" if m["restored"] is None else f"{m['restored']:.1f}%")
    pdf.ln(8)

    h, w = r["original"].shape
    tile_w = min(60.0, 75.0 * w / h)
    y = pdf.get_y()
    for i, (label, key) in enumerate((("Original", "original"), ("Noisy", "noisy"), ("Filtered", "processed"))):
        x = 10 + i * (tile_w + 4)
        pdf.set_xy(x, y)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(tile_w, 6, label)
        ok, buf = cv2.imencode(".png", r[key])
        pdf.image(io.BytesIO(buf.tobytes()), x=x, y=y + 7, w=tile_w)
    return bytes(pdf.output())


@st.cache_data(show_spinner=False, max_entries=6)
def build_exports(file_bytes, density, seed, algo, ksize, shape_choice, fmt, stem):
    r = run_pipeline(file_bytes, density, seed, algo, ksize, shape_choice)
    ext = fmt.lower()
    short = f"{algo.split(' (')[0]} {ksize}x{ksize}"
    strip = make_strip(r["original"], r["noisy"], r["processed"],
                       ("Original", f"Noisy {density * 100:.0f}%", f"Filtered: {short}"))
    files = {
        f"{stem}_filtered.{ext}": encode(r["processed"], fmt),
        f"{stem}_noisy.{ext}": encode(r["noisy"], fmt),
        f"{stem}_difference.{ext}": encode(r["heat"], fmt),
        f"{stem}_comparison.{ext}": encode(strip, fmt),
    }
    pdf, pdf_error = None, None
    try:
        pdf = build_pdf(r, {"name": stem, "density": density, "seed": seed, "algo": algo,
                            "ksize": ksize, "shape": shape_choice})
    except Exception as e:  # missing fpdf2, etc.
        pdf_error = str(e)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        if pdf:
            z.writestr(f"{stem}_report.pdf", pdf)
    return {"files": files, "pdf": pdf, "pdf_error": pdf_error, "zip": buf.getvalue()}


# ---------------------------------------------------------------- charts
PLOT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E8E4DA", family="Instrument Sans, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
)


def surface(image, title):
    step = max(1, max(image.shape) // 110)
    small = image[::step, ::step]
    x, y = np.meshgrid(range(small.shape[1]), range(small.shape[0]))
    fig = go.Figure(go.Surface(
        z=small, x=x, y=y, showscale=False,
        colorscale=[[0, "#0D1A1E"], [0.5, "#5EE0C8"], [1, "#FFB347"]],
    ))
    fig.update_layout(
        **PLOT, title=dict(text=title, font=dict(size=15)), height=440,
        scene=dict(
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False, autorange="reversed"),
            zaxis=dict(showgrid=False, title="", color="#8FA7A3"),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.1)),
        ),
    )
    return fig


def compare_slider(uri_left, uri_right, label_left, label_right, w, h):
    ratio = w / h
    height = int(min(720, max(320, 980 / ratio)))
    page = """
<style>
html,body{margin:0;height:100%;background:transparent;font-family:system-ui,sans-serif;overflow:hidden}
#wrap{position:relative;width:100%;max-width:calc(100vh * __RATIO__);aspect-ratio:__RATIO__;margin:0 auto;
  border:1px solid #2A4547;border-radius:8px;overflow:hidden;touch-action:none;cursor:ew-resize;user-select:none;background:#0D1A1E;box-sizing:border-box}
img{position:absolute;inset:0;width:100%;height:100%;display:block;pointer-events:none}
#top{clip-path:inset(0 50% 0 0)}
#bar{position:absolute;top:0;bottom:0;left:50%;width:2px;margin-left:-1px;background:#FFB347;box-shadow:0 0 14px rgba(255,179,71,.65)}
#knob{position:absolute;top:50%;left:50%;width:38px;height:38px;margin:-19px 0 0 -19px;border-radius:50%;
  background:#0D1A1E;border:2px solid #FFB347;display:flex;align-items:center;justify-content:center;color:#FFB347;font-size:15px;font-weight:700}
.tag{position:absolute;top:12px;padding:4px 10px;border-radius:4px;background:rgba(13,26,30,.85);color:#E8E4DA;font-size:12px;font-weight:600}
</style>
<div id="wrap">
  <img src="__RIGHT__" alt="__LR__">
  <img id="top" src="__LEFT__" alt="__LL__">
  <div class="tag" style="left:12px">__LL__</div>
  <div class="tag" style="right:12px">__LR__</div>
  <div id="bar"><div id="knob">&#8596;</div></div>
</div>
<script>
const wrap=document.getElementById('wrap'),top_=document.getElementById('top'),bar=document.getElementById('bar');
let drag=false;
function setPos(p){p=Math.max(0,Math.min(100,p));top_.style.clipPath='inset(0 '+(100-p)+'% 0 0)';bar.style.left=p+'%';}
function fromEvent(e){const r=wrap.getBoundingClientRect();setPos((e.clientX-r.left)/r.width*100);}
wrap.addEventListener('pointerdown',e=>{drag=true;wrap.setPointerCapture(e.pointerId);fromEvent(e);});
wrap.addEventListener('pointermove',e=>{if(drag)fromEvent(e);});
['pointerup','pointercancel'].forEach(n=>wrap.addEventListener(n,()=>{drag=false;}));
</script>
"""
    page = (page.replace("__RATIO__", f"{ratio:.4f}").replace("__LEFT__", uri_left)
            .replace("__RIGHT__", uri_right).replace("__LL__", label_left).replace("__LR__", label_right))
    embed(page, height)


HERO_CANVAS = """
<style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden}
canvas{display:block;width:100%;height:100%;border-radius:10px;border:1px solid #2A4547;box-sizing:border-box;cursor:ew-resize}
</style>
<canvas id="c"></canvas>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let W,H,gw,gh,cleanC,noisyC,scan,targetX=null;
function hash(x,y){let h=(Math.imul(x,374761393)+Math.imul(y,668265263))^0x5bd1e995;h=Math.imul(h^(h>>>13),1274126177);return((h^(h>>>16))>>>0)/4294967295;}
function specimen(u,v){
  let val=0.16+0.05*Math.sin(u*3.1+v*2.3);
  const cells=[[-0.9,-0.2,0.42],[0.1,0.35,0.5],[0.85,-0.3,0.36],[-0.2,-0.75,0.3],[1.45,0.5,0.3],[-1.55,0.4,0.34],[0.5,-0.8,0.22]];
  for(const [cx,cy,r] of cells){
    const d=Math.hypot(u-cx,v-cy);
    val+=0.55*Math.exp(-Math.pow((d-r)/0.03,2));
    if(d<r)val+=0.15*(1-d/r);
    val+=0.38*Math.exp(-Math.pow(d/(r*0.22),2));
  }
  return Math.min(1,Math.max(0,val));
}
function lerp(a,b,t){return a+(b-a)*t;}
function build(){
  const dpr=window.devicePixelRatio||1;
  W=innerWidth;H=innerHeight;cv.width=W*dpr;cv.height=H*dpr;
  gw=200;gh=Math.max(60,Math.round(gw*H/W));
  cleanC=document.createElement('canvas');noisyC=document.createElement('canvas');
  cleanC.width=noisyC.width=gw;cleanC.height=noisyC.height=gh;
  const ci=ctx.createImageData(gw,gh),ni=ctx.createImageData(gw,gh);
  for(let y=0;y<gh;y++)for(let x=0;x<gw;x++){
    const u=(x-gw/2)/gh*2,v=(y-gh/2)/gh*2,val=specimen(u,v);
    const r=lerp(13,232,val),g=lerp(26,228,val),b=lerp(30,218,val);
    const i=(y*gw+x)*4;
    ci.data[i]=r;ci.data[i+1]=g;ci.data[i+2]=b;ci.data[i+3]=255;
    let rr=r,gg=g,bb=b;const n=hash(x,y);
    if(n<0.045){rr=gg=bb=255;}else if(n<0.09){rr=gg=bb=0;}
    ni.data[i]=rr;ni.data[i+1]=gg;ni.data[i+2]=bb;ni.data[i+3]=255;
  }
  cleanC.getContext('2d').putImageData(ci,0,0);noisyC.getContext('2d').putImageData(ni,0,0);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  if(scan===undefined)scan=W*0.5;
}
function label(text,x,align){
  ctx.font='600 12px system-ui,sans-serif';ctx.textAlign=align;
  const w=ctx.measureText(text).width+20;const bx=align==='left'?x:x-w;
  ctx.fillStyle='rgba(13,26,30,.85)';ctx.fillRect(bx,12,w,24);
  ctx.fillStyle='#E8E4DA';ctx.fillText(text,align==='left'?x+10:x-10,28);
}
function frame(t){
  if(targetX!==null){scan+=(targetX-scan)*0.18;}
  else if(!reduce){scan=W*(0.5+0.38*Math.sin(t*0.00055));}
  ctx.imageSmoothingEnabled=false;
  ctx.drawImage(noisyC,0,0,W,H);
  ctx.save();ctx.beginPath();ctx.rect(0,0,scan,H);ctx.clip();ctx.drawImage(cleanC,0,0,W,H);ctx.restore();
  ctx.save();ctx.shadowColor='rgba(255,179,71,.7)';ctx.shadowBlur=14;ctx.fillStyle='#FFB347';ctx.fillRect(scan-1,0,2,H);ctx.restore();
  ctx.beginPath();ctx.arc(scan,H/2,19,0,Math.PI*2);ctx.fillStyle='#0D1A1E';ctx.fill();
  ctx.lineWidth=2;ctx.strokeStyle='#FFB347';ctx.stroke();
  ctx.fillStyle='#FFB347';ctx.font='700 15px system-ui';ctx.textAlign='center';ctx.fillText('\\u2194',scan,H/2+5);
  label('Filtered',12,'left');label('Noisy',W-12,'right');
  requestAnimationFrame(frame);
}
cv.addEventListener('pointermove',e=>{const r=cv.getBoundingClientRect();targetX=e.clientX-r.left;});
cv.addEventListener('pointerleave',()=>{targetX=null;});
addEventListener('resize',build);
build();requestAnimationFrame(frame);
</script>
"""


DANDELION = """
<style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden}
canvas{display:block;position:absolute;inset:0;width:100%;height:100%;
  -webkit-mask-image:linear-gradient(to bottom,transparent 0%,#000 38%);
  mask-image:linear-gradient(to bottom,transparent 0%,#000 38%)}
#themeBtn{position:absolute;top:14px;right:14px;width:34px;height:34px;border-radius:50%;
  background:rgba(20,38,43,.8);border:1px solid #2A4547;cursor:pointer;z-index:5;
  display:flex;align-items:center;justify-content:center;transition:border-color .15s}
#themeBtn:hover,#themeBtn:focus-visible{border-color:#FFB347;outline:none}
#dot{width:12px;height:12px;border-radius:50%;background:#FFB347}
</style>
<button id="themeBtn" title="Change color" aria-label="Change dandelion color"><div id="dot"></div></button>
<canvas id="c"></canvas>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const themes=[
  {line:'rgba(255,179,71,.55)',dot:'#FFB347',glow:'rgba(255,179,71,.28)'},
  {line:'rgba(94,224,200,.55)',dot:'#5EE0C8',glow:'rgba(94,224,200,.26)'},
  {line:'rgba(232,228,218,.5)',dot:'#E8E4DA',glow:'rgba(232,228,218,.2)'}
];
let ti=0,theme=themes[0],W,H,dpr,lines=[],seeds=[],mouse={x:-999,y:-999};
const ox=()=>W/2, oy=()=>H*0.97;
function init(){
  dpr=window.devicePixelRatio||1;W=innerWidth;H=innerHeight;
  cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  lines=[];const n=130,maxL=Math.min(H*0.88,W*0.42);
  for(let i=0;i<n;i++){
    const a=-Math.PI+Math.PI*(i/(n-1))+(Math.random()-0.5)*0.04;
    lines.push({a:a,base:a,len:maxL*(0.45+Math.random()*0.55),sway:Math.random()*6.28,r:1.6+Math.random()*1.8});
  }
  seeds=[];for(let i=0;i<14;i++)seeds.push(newSeed(true));
}
function newSeed(rand){
  return {x:W*0.25+Math.random()*W*0.5,y:rand?Math.random()*H:H*0.8,s:1+Math.random()*1.6,v:0.15+Math.random()*0.3,ph:Math.random()*6.28};
}
function frame(t){
  ctx.clearRect(0,0,W,H);
  const g=ctx.createRadialGradient(ox(),oy(),0,ox(),oy(),H*0.9);
  g.addColorStop(0,theme.glow);g.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
  ctx.lineWidth=1;
  for(const l of lines){
    const dx0=ox()+Math.cos(l.a)*l.len,dy0=oy()+Math.sin(l.a)*l.len;
    const dx=mouse.x-dx0,dy=mouse.y-dy0,d=Math.hypot(dx,dy);
    if(d<90){const f=(90-d)/90;l.a-=Math.cos(Math.atan2(dy,dx)+1.57)*f*0.03*(dx>0?1:-1);}
    else l.a+=(l.base-l.a)*0.03;
    if(!reduce)l.a+=Math.sin(t*0.001+l.sway)*0.0007;
    const x=ox()+Math.cos(l.a)*l.len,y=oy()+Math.sin(l.a)*l.len;
    ctx.strokeStyle=theme.line;ctx.beginPath();ctx.moveTo(ox(),oy());ctx.lineTo(x,y);ctx.stroke();
    ctx.fillStyle=theme.dot;ctx.beginPath();ctx.arc(x,y,l.r,0,6.283);ctx.fill();
  }
  ctx.fillStyle=theme.dot;ctx.beginPath();ctx.arc(ox(),oy(),5,0,6.283);ctx.fill();
  if(!reduce){
    ctx.globalAlpha=0.55;
    for(const s of seeds){
      s.y-=s.v;s.x+=Math.sin(t*0.0008+s.ph)*0.25;
      if(s.y<-10)Object.assign(s,newSeed(false));
      ctx.beginPath();ctx.arc(s.x,s.y,s.s,0,6.283);ctx.fillStyle=theme.dot;ctx.fill();
    }
    ctx.globalAlpha=1;
  }
  requestAnimationFrame(frame);
}
addEventListener('mousemove',e=>{mouse.x=e.clientX;mouse.y=e.clientY;});
addEventListener('mouseleave',()=>{mouse.x=mouse.y=-999;});
addEventListener('resize',init);
document.getElementById('themeBtn').addEventListener('click',()=>{
  ti=(ti+1)%themes.length;theme=themes[ti];document.getElementById('dot').style.background=theme.dot;
});
init();requestAnimationFrame(frame);
</script>
"""

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    html(f'<div class="side-brand">{LOGO}MorphoVision Pro</div>')
    html('<div class="side-note">Settings update the results as you change them.</div>')

    html('<div class="side-h">Noise</div>')
    noise_percent = st.slider("Density", 0, 40, 7, 1, format="%d%%")
    seed = st.number_input("Random seed", 0, 99999, 42, 1,
                           help="The same seed always produces the same noise pattern.")

    html('<div class="side-h">Filter</div>')
    algo = st.radio("Algorithm", ["Open-Close morphology", "Median filter", "Hybrid (median + open-close)"])
    ksize = st.slider("Kernel size", 3, 15, 3, 2)
    shape_choice = st.selectbox("Structuring element", list(SHAPES), disabled=algo.startswith("Median"))

    html('<div class="side-h">Export</div>')
    fmt = st.radio("Image format", ["PNG", "JPG"], horizontal=True)

    html('<div class="side-h">Live camera</div>')
    use_webcam = st.checkbox("Enable webcam capture")

density = noise_percent / 100.0

# ---------------------------------------------------------------- top bar + hero
html(f"""
<div class="topbar">
  <div class="brand">{LOGO}MorphoVision Pro</div>
  <div class="topbar-meta">Image restoration lab</div>
</div>
""")

hero_l, hero_r = st.columns([1, 1.15], gap="large")
with hero_l:
    html("""
    <div class="hero-copy">
      <h1 class="hero-title">Remove salt and pepper noise without softening the edges.</h1>
      <p class="hero-text">Upload an image, corrupt it with a controlled amount of impulse noise, and see how
      morphological and median filters recover it. Every result comes with PSNR, SSIM and edge scores, and
      every output can be downloaded.</p>
      <div class="hero-hint">Move your cursor across the preview to sweep between the noisy and filtered image.</div>
    </div>
    """)
with hero_r:
    embed(HERO_CANVAS, 380)

html("""
<div class="steps">
  <div class="step"><div class="step-n">1</div><div class="step-t">Add impulse noise</div>
    <div class="step-d">Pick a density from 0 to 40%. A seed makes the noise pattern repeatable.</div></div>
  <div class="step"><div class="step-n">2</div><div class="step-t">Filter the image</div>
    <div class="step-d">Choose open-close morphology, a median filter, or both together.</div></div>
  <div class="step"><div class="step-n">3</div><div class="step-t">Measure and export</div>
    <div class="step-d">Compare the scores, inspect the differences, then download images or a PDF report.</div></div>
</div>
""")

# ---------------------------------------------------------------- workspace
html("""
<div class="sec"><div class="sec-t">Workspace</div>
<div class="sec-d">Start by choosing an image. JPG and PNG files work, and large images are scaled to 1600 px on the long side.</div></div>
""")
uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")


def fmt_delta(d, unit="", digits=2, ref="vs noisy"):
    cls = "up" if d >= 0 else "down"
    return f'<div class="delta {cls}">{d:+.{digits}f}{unit} {ref}</div>'


if uploaded is None:
    st.info("Upload an image above to see the original, noisy and filtered versions side by side.")
else:
    file_bytes = uploaded.getvalue()
    stem = os.path.splitext(os.path.basename(uploaded.name))[0] or "image"

    try:
        r = run_pipeline(file_bytes, density, int(seed), algo, ksize, shape_choice)
    except ValueError:
        st.error("That file could not be read as an image. Try a different JPG or PNG.")
        st.stop()

    m = r["metrics"]
    h, w = r["original"].shape
    short_algo = algo.split(" (")[0]

    html(f"""
    <div class="sec" style="margin-top:30px"><div class="sec-t">Results<span class="chip">{stem}, {w} × {h} px</span></div></div>
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        stretch(st.image, r["original"], caption="Original", clamp=True)
    with c2:
        stretch(st.image, r["noisy"], caption=f"Noisy, {noise_percent}% density", clamp=True)
    with c3:
        stretch(st.image, r["processed"], caption=f"Filtered, {short_algo} {ksize}×{ksize}", clamp=True)

    restored_val = "n/a" if m["restored"] is None else f'{m["restored"]:.1f}<small>%</small>'
    restored_sub = ('<div class="delta">No noise added</div>' if m["restored"] is None
                    else f'<div class="delta">of {m["corrupted"]:,} corrupted pixels</div>')
    html(f"""
    <div class="readout">
      <div class="cell"><div class="val">{m["psnr"]:.2f}<small>dB</small></div>
        <div class="lab">PSNR after filtering</div>{fmt_delta(m["psnr"] - m["psnr_noisy"], " dB")}</div>
      <div class="cell"><div class="val">{m["ssim"]:.3f}</div>
        <div class="lab">SSIM (structure match)</div>{fmt_delta(m["ssim"] - m["ssim_noisy"], "", 3)}</div>
      <div class="cell"><div class="val">{m["edge"]:.2f}</div>
        <div class="lab">Edge match (Canny overlap)</div>{fmt_delta(m["edge"] - m["edge_noisy"], "", 2)}</div>
      <div class="cell"><div class="val">{restored_val}</div>
        <div class="lab">Corrupted pixels restored</div>{restored_sub}</div>
    </div>
    """)

    tab_cmp, tab_hist, tab_diff, tab_3d = st.tabs(["Compare", "Histogram", "Difference map", "3D surface"])

    with tab_cmp:
        pair = st.radio("Compare", ["Noisy and filtered", "Original and filtered"], horizontal=True,
                        label_visibility="collapsed")
        if pair.startswith("Noisy"):
            compare_slider(r["uri_noisy"], r["uri_processed"], "Noisy", "Filtered", w, h)
        else:
            compare_slider(r["uri_original"], r["uri_processed"], "Original", "Filtered", w, h)
        st.caption("Drag anywhere on the image to move the divider.")

    with tab_hist:
        log_y = st.checkbox("Log scale (shows the 0 and 255 noise spikes more clearly)", value=True)
        fig = go.Figure()
        for name, key, color in (("Original", "original", "#E8E4DA"),
                                 ("Noisy", "noisy", "#FF7A66"),
                                 ("Filtered", "processed", "#FFB347")):
            fig.add_trace(go.Histogram(x=r[key].ravel(), name=name, marker_color=color,
                                       opacity=0.6, xbins=dict(start=0, end=256, size=4)))
        fig.update_layout(**PLOT, barmode="overlay", height=420,
                          xaxis=dict(title="Pixel value", gridcolor="#2A4547"),
                          yaxis=dict(title="Pixels", gridcolor="#2A4547", type="log" if log_y else "linear"),
                          legend=dict(orientation="h", y=1.1))
        stretch(st.plotly_chart, fig)

    with tab_diff:
        d1, d2 = st.columns(2)
        with d1:
            stretch(st.image, r["heat"], caption="Difference between original and filtered", channels="BGR")
        with d2:
            stretch(st.image, r["overlay"], caption="Difference overlaid on the original", channels="BGR")
        st.caption("Brighter areas show where the filtered image differs most from the original.")

    with tab_3d:
        s1, s2 = st.columns(2)
        with s1:
            stretch(st.plotly_chart, surface(r["original"], "Original"))
        with s2:
            stretch(st.plotly_chart, surface(r["processed"], "Filtered"))
        st.caption("Pixel brightness drawn as height. Drag to rotate and scroll to zoom.")

    # ------------------------------------------------------------ downloads
    html("""
    <div class="sec"><div class="sec-t">Download results</div>
    <div class="sec-d">Choose the image format in the sidebar. Files are named after your upload.</div></div>
    """)
    ex = build_exports(file_bytes, density, int(seed), algo, ksize, shape_choice, fmt, stem)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    names = list(ex["files"])

    b1, b2, b3 = st.columns(3)
    with b1:
        stretch(st.download_button, "Download filtered image", ex["files"][names[0]], file_name=names[0],
                mime=mime, type="primary", key="dl_filtered")
    with b2:
        stretch(st.download_button, "Download noisy image", ex["files"][names[1]], file_name=names[1],
                mime=mime, key="dl_noisy")
    with b3:
        stretch(st.download_button, "Download difference map", ex["files"][names[2]], file_name=names[2],
                mime=mime, key="dl_diff")
    b4, b5, b6 = st.columns(3)
    with b4:
        stretch(st.download_button, "Download comparison strip", ex["files"][names[3]], file_name=names[3],
                mime=mime, key="dl_strip")
    with b5:
        stretch(st.download_button, "Download everything (ZIP)", ex["zip"], file_name=f"{stem}_morphovision.zip",
                mime="application/zip", key="dl_zip")
    with b6:
        if ex["pdf"]:
            stretch(st.download_button, "Download PDF report", ex["pdf"], file_name=f"{stem}_report.pdf",
                    mime="application/pdf", key="dl_pdf")
        else:
            st.warning("PDF report needs the fpdf2 package. Run: pip uninstall fpdf && pip install fpdf2")

# ---------------------------------------------------------------- webcam
if use_webcam:
    html("""
    <div class="sec"><div class="sec-t">Live camera</div>
    <div class="sec-d">Take a photo. Your browser asks for camera permission, then the current filter settings are applied to it.</div></div>
    """)
    cam_l, cam_r = st.columns([1, 1.4], gap="large")
    with cam_l:
        photo = st.camera_input("Take a photo", label_visibility="collapsed")
    with cam_r:
        if photo is not None:
            img = load_gray(photo.getvalue())
            out = apply_filter(img, algo, ksize, shape_choice)
            w1, w2 = st.columns(2)
            with w1:
                stretch(st.image, img, caption="Captured photo", clamp=True)
            with w2:
                stretch(st.image, out, caption=f"Filtered, {algo.split(' (')[0]} {ksize}×{ksize}", clamp=True)
            stretch(st.download_button, "Download filtered photo", encode(out, fmt),
                    file_name=f"camera_filtered.{fmt.lower()}",
                    mime="image/png" if fmt == "PNG" else "image/jpeg", type="primary", key="dl_cam")
        else:
            st.info("Your filtered photo appears here after you capture one.")

# ---------------------------------------------------------------- footer
embed(DANDELION, 340)

html(f"""
<div class="footer"><span>MorphoVision Pro, built with OpenCV, Streamlit and Plotly.</span>
<span>{datetime.date.today():%Y}</span></div>
""")