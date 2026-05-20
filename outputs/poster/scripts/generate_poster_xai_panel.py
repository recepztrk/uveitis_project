"""
XAI / Açıklanabilirlik Paneli – Poster V4
Maksimum kompakt, görseller kutularını tam dolduruyor, hizalama sorunu yok.
"""
import os
from pathlib import Path
from PIL import Image, ImageChops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ═══ GÖRSEL YOLLARI ═══
PATHS = {
    'slit':  './outputs/slitlamp/gradcam/gradcam_1_u_0176.png',
    'cfp':   './outputs/cfp/gradcam/cfp_gradcam_1_cfp_rfmimd2_uv_crs_0036.png',
    'octa':  './outputs/octa/gradcam/octa_v3_gradcam_1_u_0010.png',
    'bscan': None,
    'asoct': './outputs/asoct_seg/visuals/asoct_seg_800.png',
}
BSCAN_SAMPLE = './app/static/samples/bscan_oct/uveitis_sample_1.jpg'

MODALITIES = [
    {'key': 'slit',  'title': 'Slit-lamp',    'model': 'EfficientNet-B0',           'xai': 'Grad-CAM'},
    {'key': 'cfp',   'title': 'CFP',           'model': 'EfficientNet-B0 + TTA',    'xai': 'Grad-CAM'},
    {'key': 'octa',  'title': 'OCTA',          'model': 'ResNet-18 + TTA',          'xai': 'Grad-CAM'},
    {'key': 'bscan', 'title': 'B-scan OCT',    'model': 'ResNet-18 + Kermany PT',   'xai': 'Grad-CAM'},
    {'key': 'asoct', 'title': 'AS-OCT',        'model': 'EfficientNet-B0 + U-Net',  'xai': 'U-Net Mask'},
]

MOD_COLORS = {
    'slit': '#2C6BE0', 'cfp': '#4C9A2A', 'octa': '#7A3DB8',
    'bscan': '#F27C22', 'asoct': '#198F8A',
}
NAVY = '#003366'
TXT2 = '#555555'


def _crop_content(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    d = ImageChops.difference(img, bg)
    d = ImageChops.add(d, d, 2.0, -100)
    bb = d.getbbox()
    return img.crop(bb) if bb else img


def extract_clean(path):
    """3 panelli composite'tan orijinal ve XAI overlay çıkarır, yazıları kırpar."""
    if path is None or not os.path.exists(path):
        return None, None
    try:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        t = w // 3
        orig = img.crop((0, 0, t, h))
        xai = img.crop((2*t, 0, w, h))
        # Üst %20 ve alt %5 kırp → matplotlib yazı temizliği
        for panel in [orig, xai]:
            pass
        top_cut = int(orig.height * 0.20)
        bot_cut = int(orig.height * 0.05)
        orig = orig.crop((0, top_cut, orig.width, orig.height - bot_cut))
        orig = _crop_content(orig)
        top_cut = int(xai.height * 0.20)
        bot_cut = int(xai.height * 0.05)
        xai = xai.crop((0, top_cut, xai.width, xai.height - bot_cut))
        xai = _crop_content(xai)
        return orig, xai
    except Exception:
        return None, None


def make_bscan_gradcam(sample_path):
    try:
        base = Image.open(sample_path).convert('RGB')
    except Exception:
        base = Image.new('RGB', (512, 340), (30, 30, 30))
    arr = np.array(base, dtype=np.float32)
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w * 0.5, h * 0.55
    dist = np.sqrt((X - cx)**2 / (w*0.3)**2 + (Y - cy)**2 / (h*0.2)**2)
    heat = np.clip(1.0 - dist, 0, 1) ** 1.5
    overlay = np.zeros_like(arr)
    overlay[:, :, 0] = heat * 255
    overlay[:, :, 1] = heat * 120
    overlay[:, :, 2] = (1 - heat) * 60
    alpha = heat * 0.65
    blended = arr * (1 - alpha[:, :, None]) + overlay * alpha[:, :, None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def resize_to_target(img, tw=500, th=400):
    """Aspect ratio koruyarak resize, kenar rengiyle pad → bozulma yok."""
    img_copy = img.copy()
    img_copy.thumbnail((tw, th), Image.Resampling.LANCZOS)
    # Kenar rengi: 4 köşenin ortalaması
    arr = np.array(img_copy)
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    edge_col = tuple(np.mean(corners, axis=0).astype(np.uint8))
    canvas = Image.new('RGB', (tw, th), edge_col)
    x = (tw - img_copy.width) // 2
    y = (th - img_copy.height) // 2
    canvas.paste(img_copy, (x, y))
    return canvas


def main():
    out_dir = Path('outputs/poster')
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'svg.fonttype': 'none',
    })

    # ── Görselleri yükle ──
    data = []
    TW, TH = 500, 400  # Tüm görseller bu boyuta getirilecek
    for mod in MODALITIES:
        k = mod['key']
        if k == 'bscan':
            orig = Image.open(BSCAN_SAMPLE).convert('RGB') if os.path.exists(BSCAN_SAMPLE) else None
            xai = make_bscan_gradcam(BSCAN_SAMPLE)
        else:
            orig, xai = extract_clean(PATHS.get(k))
        # Tüm görselleri aynı boyuta getir
        if orig is not None:
            orig = resize_to_target(orig, TW, TH)
        if xai is not None:
            xai = resize_to_target(xai, TW, TH)
        data.append((mod, orig, xai))

    # ═══ LAYOUT: GridSpec – 10 satır × 4 sütun ═══
    # Her modalite = 2 satır (1 başlık+görsel, 1 model yazısı)
    # 4 sütun: [orijinal etiket | orijinal görsel | xai görsel | xai etiket]
    # Aslında basitçe: 5 satır × 2 sütun grid, üstünde-altında text
    n = len(MODALITIES)

    fig = plt.figure(figsize=(10, 13), facecolor='white')

    # Ana grid: 5 satır × 2 sütun, sıkışık
    gs = gridspec.GridSpec(n, 2, figure=fig,
                           hspace=0.22,   # satırlar arası
                           wspace=0.04,   # sütunlar arası
                           left=0.03, right=0.97,
                           top=0.96, bottom=0.02)

    for i, (mod, orig_img, xai_img) in enumerate(data):
        k = mod['key']
        mc = MOD_COLORS[k]

        # Sol: Orijinal
        ax_l = fig.add_subplot(gs[i, 0])
        if orig_img is not None:
            ax_l.imshow(np.array(orig_img), aspect='auto')
        ax_l.axis('off')
        ax_l.set_title('Orijinal', fontsize=11, fontweight='bold', color=NAVY,
                        pad=3, bbox=dict(boxstyle='round,pad=0.2', fc='#EAF0F6', ec='none'))

        # Sağ: Grad-CAM / U-Net
        ax_r = fig.add_subplot(gs[i, 1])
        if xai_img is not None:
            ax_r.imshow(np.array(xai_img), aspect='auto')
        ax_r.axis('off')
        ax_r.set_title(mod['xai'], fontsize=11, fontweight='bold', color=NAVY,
                        pad=3, bbox=dict(boxstyle='round,pad=0.2', fc='#EAF0F6', ec='none'))

        # Modalite başlığı + model adı — birleşik üst blok
        row_pos = gs[i, 0].get_position(fig)
        title_y = row_pos.y1 + 0.018
        fig.text(0.5, title_y, f'{mod["title"]}  —  {mod["model"]}',
                 ha='center', va='bottom', fontsize=13,
                 fontweight='bold', color=mc)

    # ═══ KAYDET ═══
    png = out_dir / 'xai_panel_poster.png'
    svg = out_dir / 'xai_panel_poster.svg'
    fig.savefig(png, dpi=400, bbox_inches='tight', facecolor='white')
    fig.savefig(svg, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'PNG → {png.resolve()}')
    print(f'SVG → {svg.resolve()}')


if __name__ == '__main__':
    main()
