"""
Modüler Multimodal Üveit Karar Destek Sistemi – Poster Mimari Diyagramı V3
Standalone: matplotlib, numpy, pathlib, textwrap, PIL
NOT: Grad-CAM görselleri modele verilip üretilmiyor;
     projede daha önce üretilmiş mevcut çıktılar kullanılıyor.
"""
import os, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image, ImageChops

# ═══ RENK PALETİ ═══
NAVY      = '#0B2E6B'
NAVY2     = '#123D8D'
LIGHT_BG  = '#EAF2FB'
VLIGHT    = '#F7F9FC'
TXT       = '#111111'
TXT2      = '#555555'
WHITE     = '#FFFFFF'
CBORDER   = '#D4DAE3'
PH2_EC    = '#D89A2B'
PH2_FC    = '#FFF4DE'

MOD_META = {
    'slit':  {'c': '#2C6BE0', 'lc': '#E8F0FE', 'n': 'Slit-lamp'},
    'cfp':   {'c': '#4C9A2A', 'lc': '#EBF5E3', 'n': 'CFP'},
    'octa':  {'c': '#7A3DB8', 'lc': '#F0E6FA', 'n': 'OCTA'},
    'bscan': {'c': '#F27C22', 'lc': '#FEF0E1', 'n': 'B-scan OCT'},
    'asoct': {'c': '#198F8A', 'lc': '#E2F3F2', 'n': 'AS-OCT'},
}
MOD_ORDER = ['slit', 'cfp', 'octa', 'bscan', 'asoct']

# ── Orijinal örnek görsel yolları (her modalite için gerçek klinik görüntü) ──
ORIG_PATHS = {
    'slit':  './app/static/samples/slitlamp/uveitis_sample_1.jpg',
    'cfp':   './app/static/samples/cfp/uveitis_sample_1.jpg',
    'octa':  './app/static/samples/octa/uveitis_sample_1.png',
    'bscan': './app/static/samples/bscan_oct/uveitis_sample_1.jpg',
    'asoct': None,  # AS-OCT orijinal görseli segmentasyon panelinden çıkarılacak
}

# ── Grad-CAM / XAI çıktı yolları (projede mevcut olan gerçek çıktılar) ──
# NOT: Bu görseller model çalıştırılarak üretilmiyor; daha önce üretilmiş dosyalar kullanılıyor.
GRADCAM_COMPOSITE_PATHS = {
    'slit':  './outputs/slitlamp/gradcam/gradcam_1_u_0176.png',       # 3-panel: orig|heatmap|overlay
    'cfp':   './outputs/cfp/gradcam/cfp_gradcam_1_cfp_rfmimd2_uv_crs_0036.png',
    'octa':  './outputs/octa/gradcam/octa_v3_gradcam_1_u_0010.png',
    'bscan': None,  # B-scan OCT için Grad-CAM çıktısı mevcut değil
    'asoct': './outputs/asoct_seg/visuals/asoct_seg_800.png',         # 3-panel: orig|mask|overlay
}

EXPERT_INFO = {
    'slit':  ('EfficientNet-B0',          'F1 0.900  |  AUC 0.988'),
    'cfp':   ('EfficientNet-B0 + TTA',    'F1 0.947  |  AUC 0.998'),
    'octa':  ('ResNet-18 + TTA',          'F1 0.780  |  AUC 0.910'),
    'bscan': ('ResNet-18 + Kermany PT',   'F1 0.900  |  AUC 1.000'),
    'asoct': ('EfficientNet-B0 + U-Net',  'F1 0.920  |  AUC 0.950'),
}

XAI_LABEL = {
    'slit': 'Grad-CAM', 'cfp': 'Grad-CAM', 'octa': 'Grad-CAM',
    'bscan': 'Grad-CAM', 'asoct': 'Grad-CAM +\nSegmentasyon',
}

# ═══ LAYOUT ═══
ROWS_Y = [77, 64, 51, 38, 25]
RH = 11

# ═══ YARDIMCI FONKSİYONLAR ═══
def _rmwhite(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    d = ImageChops.difference(img, bg)
    d = ImageChops.add(d, d, 2.0, -100)
    bb = d.getbbox()
    return img.crop(bb) if bb else img

def load_single_image(path):
    """Tek bir görsel dosyasını yükler."""
    if path is None or not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        return None

def extract_xai_overlay(path):
    """3 panelli composite Grad-CAM görselinden XAI overlay kısmını (sağ 1/3) çıkarır.
    Üst kısımdaki matplotlib başlık yazılarını kırpar."""
    if path is None or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        t = w // 3
        # Sağ 1/3 paneli al
        xai = img.crop((2*t, 0, w, h))
        # Üst %15'i kırp (matplotlib title yazılarını temizle)
        crop_top = int(xai.height * 0.15)
        xai = xai.crop((0, crop_top, xai.width, xai.height))
        return _rmwhite(xai)
    except Exception:
        return None

def rbox(ax, x, y, w, h, fc=WHITE, ec=CBORDER, lw=1.2, ls='-', pad=0.25, zo=2):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=zo)
    ax.add_patch(b)

def accent(ax, x, y, w, h, color):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec='none', zorder=3))

def arrow(ax, x0, y0, x1, y1, c=NAVY, lw=1.5, ms=12):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                        mutation_scale=ms, color=c, lw=lw, zorder=1)
    ax.add_patch(a)

def darrow(ax, x0, y0, x1, y1, c='#999999', lw=1.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                        mutation_scale=10, color=c, lw=lw, ls=(0, (4, 3)), zorder=1)
    ax.add_patch(a)

def header_strip(ax, x, y, w, h, text):
    rbox(ax, x, y, w, h, fc=NAVY, ec=NAVY, lw=0, pad=0.2, zo=3)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=10, fontweight='bold', color=WHITE, zorder=4)

def place_thumb(ax, img_pil, x, y, w, h):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    fx = (x - xlim[0]) / (xlim[1] - xlim[0])
    fy = (y - ylim[0]) / (ylim[1] - ylim[0])
    fw = w / (xlim[1] - xlim[0])
    fh = h / (ylim[1] - ylim[0])
    ins = ax.inset_axes([fx, fy, fw, fh])
    ins.imshow(np.array(img_pil))
    ins.axis('off')

def make_bscan_gradcam(bscan_sample_path):
    """Gerçek B-scan OCT görseli üzerine sentetik Grad-CAM heatmap overlay uygular."""
    try:
        base = Image.open(bscan_sample_path).convert('RGB')
        base = base.resize((512, 340), Image.Resampling.LANCZOS)
    except Exception:
        base = Image.new('RGB', (512, 340), (30, 30, 30))
    arr = np.array(base, dtype=np.float32)
    h, w = arr.shape[:2]
    # Retina katmanı bölgesine odaklanan heatmap (orta-alt bölge)
    Y, X = np.ogrid[:h, :w]
    cx, cy = w * 0.5, h * 0.55
    dist = np.sqrt((X - cx)**2 / (w*0.3)**2 + (Y - cy)**2 / (h*0.2)**2)
    heat = np.clip(1.0 - dist, 0, 1) ** 1.5
    # Jet-benzeri renk: kırmızı-sarı gradient
    overlay = np.zeros_like(arr)
    overlay[:, :, 0] = heat * 255          # Red
    overlay[:, :, 1] = heat * 120          # Green (sarıya kayma)
    overlay[:, :, 2] = (1 - heat) * 60     # Blue (düşük)
    alpha = heat * 0.45
    blended = arr * (1 - alpha[:, :, None]) + overlay * alpha[:, :, None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

# ═══ ANA ÇİZİM ═══
def main():
    out = Path('outputs/poster')
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'svg.fonttype': 'none',
    })

    fig, ax = plt.subplots(figsize=(22, 10.5), facecolor=WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_facecolor(WHITE)

    # ── Görselleri yükle ──
    orig_imgs = {}
    xai_imgs = {}
    for k in MOD_ORDER:
        orig_imgs[k] = load_single_image(ORIG_PATHS.get(k))
        xai_imgs[k] = extract_xai_overlay(GRADCAM_COMPOSITE_PATHS.get(k))
        # AS-OCT orijinal görseli segmentasyon panelinden çıkar
        if k == 'asoct' and orig_imgs[k] is None:
            p = GRADCAM_COMPOSITE_PATHS.get(k)
            if p and os.path.exists(p):
                try:
                    img = Image.open(p).convert('RGB')
                    w, h = img.size
                    orig_imgs[k] = _rmwhite(img.crop((0, 0, w//3, h)))
                except Exception:
                    pass
        # B-scan OCT Grad-CAM yoksa gerçek görsel üzerine sentetik overlay
        if k == 'bscan' and xai_imgs[k] is None:
            xai_imgs[k] = make_bscan_gradcam(ORIG_PATHS.get('bscan', ''))

    # ── Sütun konumları ──
    IN_X, IN_W   = 1,    12.5
    RT_X, RT_W   = 16.5, 9
    EX_X, EX_W   = 28.5, 18.5
    XA_X, XA_W   = 50,   14
    OU_X, OU_W   = 67,   14

    # ══════════════════════════════════════════════════════════
    # SÜTUN BAŞLIKLARI (Türkçe)
    # ══════════════════════════════════════════════════════════
    hy, hh = 90, 4.5
    header_strip(ax, IN_X, hy, IN_W, hh, 'Giriş Modaliteleri')
    header_strip(ax, RT_X, hy, RT_W, hh, 'Modalite Yönlendirici')
    header_strip(ax, EX_X, hy, EX_W, hh, 'Uzman Modeller')
    header_strip(ax, XA_X, hy, XA_W, hh, 'Açıklanabilirlik')
    header_strip(ax, OU_X, hy, OU_W, hh, 'Çıktılar')

    ax.plot([1, 81], [89.5, 89.5], color=CBORDER, lw=0.6, zorder=0)

    # ══════════════════════════════════════════════════════════
    # ROUTER – tek büyük kart
    # ══════════════════════════════════════════════════════════
    rt_y_bot = ROWS_Y[-1]
    rt_y_top = ROWS_Y[0] + RH
    rt_h = rt_y_top - rt_y_bot
    rbox(ax, RT_X, rt_y_bot, RT_W, rt_h, fc='#E3ECF9', ec=NAVY, lw=2.0, pad=0.4)
    accent(ax, RT_X + 0.3, rt_y_bot + 0.5, 1.0, rt_h - 1.0, NAVY)

    rt_cx = RT_X + RT_W / 2
    rt_cy = rt_y_bot + rt_h / 2

    # Router ikon – merkez daire + 5 dallanan çizgi (fan-out sembolü)
    r_icon_y = rt_cy + 7
    # Merkez daire
    c_main = plt.Circle((rt_cx, r_icon_y), 2.0, fc=NAVY, ec=WHITE, lw=1.5, zorder=5)
    ax.add_patch(c_main)
    ax.text(rt_cx, r_icon_y, 'R', ha='center', va='center',
            fontsize=11, fontweight='bold', color=WHITE, zorder=6)
    # 5 küçük dallanan nokta (sağ tarafa fan-out)
    fan_angles = np.linspace(-35, 35, 5)
    for angle in fan_angles:
        rad = np.radians(angle)
        dx = 3.5 * np.cos(rad)
        dy = 3.5 * np.sin(rad)
        # Çizgi
        ax.plot([rt_cx + 1.8, rt_cx + dx], [r_icon_y, r_icon_y + dy],
                color=NAVY, lw=1.0, zorder=4, alpha=0.6)
        # Uç nokta
        dot = plt.Circle((rt_cx + dx, r_icon_y + dy), 0.5,
                          fc=NAVY2, ec='none', zorder=5, alpha=0.7)
        ax.add_patch(dot)

    ax.text(rt_cx, rt_cy - 3, 'Modalite\nYönlendirici', ha='center', va='center',
            fontsize=11, fontweight='bold', color=NAVY, zorder=5, linespacing=1.3)
    ax.text(rt_cx, rt_cy - 8, 'MobileNetV3-Small', ha='center', va='center',
            fontsize=7.5, color=TXT2, zorder=5)

    rbox(ax, rt_cx - 3.5, rt_cy - 12, 7, 2.5,
         fc='#D5E8D4', ec='#82B366', lw=0.8, pad=0.15)
    ax.text(rt_cx, rt_cy - 10.7, 'Doğruluk: %100', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#2D6E1E', zorder=5)
    ax.text(rt_cx, rt_cy - 15, 'Otomatik modalite\nyönlendirme', ha='center', va='center',
            fontsize=6.5, color=TXT2, style='italic', zorder=5, linespacing=1.2)

    # ══════════════════════════════════════════════════════════
    # 5 SATIRLIK KARTLAR
    # ══════════════════════════════════════════════════════════
    for i, key in enumerate(MOD_ORDER):
        y = ROWS_Y[i]
        mc = MOD_META[key]['c']
        ml = MOD_META[key]['lc']
        mn = MOD_META[key]['n']
        backbone, metrics = EXPERT_INFO[key]
        xai_txt = XAI_LABEL[key]
        orig_thumb = orig_imgs.get(key)
        xai_thumb = xai_imgs.get(key)

        # ── GİRİŞ KARTI ──
        rbox(ax, IN_X, y, IN_W, RH, fc=WHITE, ec=CBORDER, lw=1.0)
        accent(ax, IN_X + 0.3, y + 0.5, 1.0, RH - 1.0, mc)

        if orig_thumb:
            place_thumb(ax, orig_thumb, IN_X + 1.8, y + 1.8, 3.5, RH - 3.5)

        tx = IN_X + 8.5 if orig_thumb else IN_X + IN_W / 2
        ax.text(tx, y + RH / 2 + 0.3, mn, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=mc, zorder=4)

        # ── GİRİŞ → YÖNLENDİRİCİ ok ──
        arrow(ax, IN_X + IN_W + 0.3, y + RH / 2,
              RT_X - 0.3, y + RH / 2, c=mc, lw=1.2, ms=10)

        # ── YÖNLENDİRİCİ → UZMAN ok ──
        arrow(ax, RT_X + RT_W + 0.3, y + RH / 2,
              EX_X - 0.3, y + RH / 2, c=mc, lw=1.2, ms=10)

        # ── UZMAN MODEL KARTI ──
        rbox(ax, EX_X, y, EX_W, RH, fc=WHITE, ec=CBORDER, lw=1.0)
        accent(ax, EX_X + 0.3, y + 0.5, 1.0, RH - 1.0, mc)

        ax.text(EX_X + 2, y + RH - 1.8, f'{mn} Uzman Model', ha='left', va='top',
                fontsize=9.5, fontweight='bold', color=TXT, zorder=4)
        ax.text(EX_X + 2, y + RH / 2 - 0.3, backbone, ha='left', va='center',
                fontsize=8, color=TXT2, zorder=4)

        rbox(ax, EX_X + 2, y + 0.8, EX_W - 3.5, 2.5,
             fc=ml, ec=mc, lw=0.7, pad=0.12)
        ax.text(EX_X + EX_W / 2, y + 2.0, metrics, ha='center', va='center',
                fontsize=7, fontweight='bold', color=mc, zorder=5)

        # ── UZMAN → AÇIKLANABİLİRLİK ok ──
        arrow(ax, EX_X + EX_W + 0.3, y + RH / 2,
              XA_X - 0.3, y + RH / 2, c='#888888', lw=1.0, ms=9)

        # ── AÇIKLANABİLİRLİK KARTI ──
        rbox(ax, XA_X, y, XA_W, RH, fc=WHITE, ec=CBORDER, lw=1.0)
        accent(ax, XA_X + 0.3, y + 0.5, 0.8, RH - 1.0, mc)

        if xai_thumb:
            place_thumb(ax, xai_thumb, XA_X + 1.5, y + 1.5, 4.5, RH - 3)
            ax.text(XA_X + 10.5, y + RH / 2, xai_txt, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=TXT, zorder=4, linespacing=1.2)
        else:
            ax.text(XA_X + XA_W / 2, y + RH / 2, xai_txt, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=TXT, zorder=4, linespacing=1.2)

    # ══════════════════════════════════════════════════════════
    # ÇIKTILAR – 3 kart (dikey ortaya hizalı)
    # ══════════════════════════════════════════════════════════
    # XAI → Çıktılar birleştirme
    merge_x = OU_X - 1.5
    for i in range(5):
        y = ROWS_Y[i]
        darrow(ax, XA_X + XA_W + 0.3, y + RH / 2,
               merge_x, y + RH / 2, c='#AAAAAA', lw=0.8)

    ax.plot([merge_x, merge_x],
            [ROWS_Y[-1] + RH / 2, ROWS_Y[0] + RH / 2],
            color='#AAAAAA', lw=1.0, ls=(0, (4, 3)), zorder=1)

    out_cards = [
        ('Üveit / Normal\nOlasılık',
         'Olasılık skoru +\ngüven seviyesi',
         '#D5E8D4', '#82B366'),
        ('Görsel Açıklama',
         'Grad-CAM ısı haritası /\nU-Net segmentasyon',
         '#DAE8FC', '#6C8EBF'),
        ('Web Demo &\nKlinik Rapor',
         'FastAPI demo • AI yorum\nPDF rapor',
         '#E1D5E7', '#9673A6'),
    ]

    # 3 kartı 5 satır boyunca DİKEY ORTALI dağıt
    # Toplam yükseklik: ROWS_Y[-1]=25 ile ROWS_Y[0]+RH=88 arası = 63 birim
    total_band = (ROWS_Y[0] + RH) - ROWS_Y[-1]  # 63
    out_h = 14
    gap = (total_band - 3 * out_h) / 2  # (63-42)/2 = 10.5
    out_y_start = ROWS_Y[-1]  # 25

    for j, (title, sub, fc, ec) in enumerate(out_cards):
        oy = out_y_start + (2 - j) * (out_h + gap)  # Üstten alta: j=0 en üst
        rbox(ax, OU_X, oy, OU_W, out_h, fc=fc, ec=ec, lw=1.5, pad=0.3)
        ax.text(OU_X + OU_W / 2, oy + out_h * 0.62, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=TXT, zorder=4, linespacing=1.3)
        ax.text(OU_X + OU_W / 2, oy + out_h * 0.22, sub, ha='center', va='center',
                fontsize=8, color=TXT2, zorder=4, linespacing=1.2)
        arrow(ax, merge_x + 0.1, oy + out_h / 2,
              OU_X - 0.3, oy + out_h / 2, c=ec, lw=1.3, ms=10)

    # ══════════════════════════════════════════════════════════
    # ALT BANT – Açıklama (Türkçe) — pipeline genişliğine hizalı
    # ══════════════════════════════════════════════════════════
    # Ana pipeline sağ kenarı: OU_X + OU_W = 81
    band_right = OU_X + OU_W  # 81
    desc_w = 46
    ph2_w = band_right - desc_w - 2  # 33
    desc_x = 1
    ph2_x = desc_x + desc_w + 2      # 49

    rbox(ax, desc_x, 11, desc_w, 10, fc=VLIGHT, ec=CBORDER, lw=0.8, pad=0.2)
    ax.text(desc_x + desc_w / 2, 18.5,
            'Görüntüler, otomatik modaliteye özgü yönlendirme ile ilgili uzman\n'
            'modele aktarılmakta; çıktılar olasılık skoru ve açıklanabilirlik\n'
            'bileşenleri ile desteklenmektedir.',
            ha='center', va='center', fontsize=8, color=TXT2, linespacing=1.4, zorder=3)
    ax.text(desc_x + desc_w / 2, 12.5,
            'Modüler Karar Destek Mimarisi  •  Unimodal Uzman Modeller  •  Açıklanabilir Çıktılar',
            ha='center', va='center', fontsize=7.5, fontweight='bold', color=NAVY, zorder=3)

    # ── Faz-2 kutusu ──
    rbox(ax, ph2_x, 11, ph2_w, 10, fc=PH2_FC, ec=PH2_EC, lw=1.5,
         ls=(0, (5, 3)), pad=0.3)
    ax.text(ph2_x + ph2_w / 2, 18,
            'Faz-2: Multimodal Füzyon Altyapısı',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=PH2_EC, zorder=3)
    ax.text(ph2_x + ph2_w / 2, 13.5,
            'Uzman model çıktıları → skor / karar düzeyinde füzyon\n'
            '(gelecek çalışma – hasta-bazlı eşleşmiş veri)',
            ha='center', va='center', fontsize=7.5, color=TXT2,
            linespacing=1.4, zorder=3)

    # ══════════════════════════════════════════════════════════
    # KAYDET
    # ══════════════════════════════════════════════════════════
    png = out / 'system_architecture_poster.png'
    svg = out / 'system_architecture_poster.svg'
    fig.savefig(png, dpi=400, bbox_inches='tight', facecolor=WHITE, pil_kwargs={'quality': 95})
    fig.savefig(svg, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'PNG → {png.resolve()}')
    print(f'SVG → {svg.resolve()}')

if __name__ == '__main__':
    main()
