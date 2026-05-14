# ==============================================================================
# app/main.py
#
# Üveit Karar Destek Sistemi — FastAPI Demo Web Uygulaması
#
# Endpoint'ler:
#   GET  /              → Ana sayfa (index.html)
#   GET  /api/models    → Mevcut modellerin listesi ve performans bilgileri
#   POST /api/predict   → Görüntü analizi (tahmin + Grad-CAM)
#   GET  /api/samples/{modality} → Hazır örnek vaka dosyaları
#
# Başlatma:
#   cd uveitis_project
#   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# ==============================================================================

import os
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from app.inference import InferenceEngine

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Gemini Başlangıç Ayarları
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class CommentRequest(BaseModel):
    modality: str
    prediction: str
    probability: float
    confidence: str
    model_name: str = ""
    model_backbone: str = ""
    model_f1: float = 0.0
    model_auc: float = 0.0
    clinical_note: str = ""
    training_data: int = 0

app = FastAPI(
    title="Üveit Karar Destek Sistemi",
    description="Multimodal derin öğrenme tabanlı oftalmolojik tanı desteği",
    version="1.0.0",
)

# Statik dosyalar ve şablonlar
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
app.mount("/demo_images", StaticFiles(directory=str(PROJECT_ROOT / "data_work" / "demo_images")), name="demo_images")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Inference motoru — uygulama başlatıldığında tüm modeller yüklenir
engine = InferenceEngine()


@app.get("/")
async def index(request: Request):
    """Ana sayfa — demo arayüzünü döndürür."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/models")
async def get_models():
    """Mevcut modellerin listesini ve performans bilgilerini döndürür."""
    return engine.get_available_modalities()


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    modality: str = Form(...),
):
    """Yüklenen görüntüyü seçilen modalite modeliyle analiz eder.

    Returns:
        JSON: prediction, probability, gradcam görselleri, model bilgileri
    """
    # Dosya tipi kontrolü
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Yalnızca görüntü dosyaları kabul edilir.")

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Boş dosya yüklendi.")

    try:
        result = engine.predict(image_bytes, modality)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz sırasında hata: {str(e)}")

    return JSONResponse(content=result)


@app.get("/api/samples/{modality}")
async def get_samples(modality: str):
    """Belirtilen modalite için hazır örnek vaka dosyalarını data_work/demo_images'dan listeler."""
    samples_dir = PROJECT_ROOT / "data_work" / "demo_images" / modality

    if not samples_dir.exists():
        return []

    samples = []
    # RGLOB ile uveitis/non_uveitis alt klasörlerini tara
    for img_path in sorted(samples_dir.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]:
            
            parent_name = img_path.parent.name.lower()
            if "uveitis" in parent_name or "mcoa" in parent_name:
                label = "uveitis"
                label_display = "Anormal (Üveit / MCOA)"
            else:
                label = "normal"
                label_display = "Normal"

            rel_path = img_path.relative_to(samples_dir)
            samples.append({
                "filename": f"{parent_name}/{img_path.name}",
                "url": f"/demo_images/{modality}/{rel_path}",
                "label": label,
                "label_display": label_display,
            })

    return samples

@app.post("/api/generate_comment")
async def generate_comment(request: CommentRequest):
    """Analiz sonuçlarına göre Gemini API'den klinik yorum üretir."""
    if not GEMINI_API_KEY:
        return {"comment": "⚠️ Gemini API Anahtarı (.env) bulunamadı. Lütfen sistemi yapılandırın."}
        
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Modalite ismini Türkçeleştir
        modality_names = {
            "slitlamp": "Slit-lamp (Biyomikroskop)",
            "octa": "OCTA (Anjiyografi)",
            "cfp": "CFP (Renkli Fundus Fotoğrafı)",
            "bscan_oct": "B-Scan OCT",
            "as_oct": "AS-OCT (Ön Segment OCT)"
        }
        mod_name = modality_names.get(request.modality, request.modality)
        
        # AS-OCT'nin klinik doğası ve XAI yöntemi tamamen farklıdır
        if request.modality == "as_oct":
            pred_tr = "Anormal / Korneal Opasite (MCOA) ile uyumlu bulgular" if request.prediction == "uveitis" else "Normal / Fizyolojik ön segment"
            xai_method = "U-Net Anatomik Segmentasyon Haritası"
            xai_instruction = "Yapay zekanın kullandığı U-Net Anatomik Segmentasyon Haritası bilgisini ve klinik notu kullanarak, modelin ön segment (kornea, iris vb.) yapılarını nasıl başarıyla ayırdığını ve patolojiyi anatomik olarak nasıl tespit ettiğini klinik bir dille izah et."
        else:
            pred_tr = "Anormal / Üveit ile uyumlu bulgular" if request.prediction == "uveitis" else "Normal / Fizyolojik sınırlar içinde"
            xai_method = "Grad-CAM Isı Haritası (Heatmap)"
            xai_instruction = "Yapay zekanın sağlanan 'Anatomik Odak Noktası' bilgisine dayanarak, Grad-CAM ısı haritasının görsel olarak tam olarak hangi patolojik lezyonlara odaklanmış olabileceğini klinik bir dille izah et."
        
        prompt = f"""
        Sen Türkiye'nin en iyi hastanelerinden birinde çalışan, son derece profesyonel, vizyoner ve kıdemli bir Oftalmologsun (Göz Hastalıkları Uzmanı). 
        Asistanın, yeni nesil bir 'Yapay Zeka Tıbbi Görüntüleme ve Karar Destek Sistemi' kullanarak bir hastanın taramasını yaptı ve sana aşağıdaki detaylı analizi sundu:

        --- HASTA VE GÖRÜNTÜLEME BİLGİSİ ---
        - Kullanılan Modalite (Cihaz): {mod_name} ({request.model_name})
        
        --- YAPAY ZEKA TANI VE GÜVEN METRİKLERİ ---
        - AI Tahmini: {pred_tr}
        - AI Hesaplanan Olasılık Oranı: %{request.probability:.1f}
        - Sistemin Kendi Kararına Güven Seviyesi: {request.confidence}
        
        --- YAPAY ZEKA MODELİNİN TEKNİK ALTYAPISI ---
        - Mimari (Backbone): {request.model_backbone}
        - Eğitim Verisi Büyüklüğü: {request.training_data} klinik görüntü
        - Model Başarımı: F1-Score: {request.model_f1:.3f}, AUC: {request.model_auc:.3f}
        - Görsel Açıklanabilirlik (XAI) Yöntemi: {xai_method}
        - AI Anatomik Odak Noktası: {request.clinical_note}
        
        GÖREV:
        Bu verileri harmanlayarak, hastanın elektronik tıbbi kayıt (EHR) dosyasına eklenebilecek kalitede, son derece prestijli, klinik ve akademik bir rapor özeti yaz.
        
        Raporunda şunları mükemmel bir akıcılıkla eriterek anlat:
        1. Elde edilen bulgunun ({pred_tr}) bu görüntüleme yöntemi ({mod_name}) için ne anlama geldiği.
        2. {xai_instruction}
        3. Sistemin teknik başarısına (AUC/F1 skorları ve eğitim verisine) atıfta bulunarak, çıkan analiz sonucunun klinisyen için ne kadar güvenilir bir 'ikinci görüş' (second opinion) sunduğunu belirt.
        
        KURALLAR:
        - ASLA "Ben bir yapay zekayım", "Doktora danışın", "Kesin tanı yerine geçmez" gibi sorumluluk reddi cümleleri kurma! (Bu uyarıyı arayüzde biz zaten veriyoruz, senin rolün uzman profesör olmak).
        - Üslubun çok ağır, güven veren, edebi ama tamamen tıbbi terminolojiye hakim olsun.
        - Paragraf yapısı kullan, maddeleme işareti KULLANMA.
        - Uzunluğu ortalama 4-6 cümle olsun. Ne çok kısa, ne de destan gibi uzun olsun.
        """
        
        response = model.generate_content(prompt)
        return {"comment": response.text.strip()}
        
    except Exception as e:
        return {"comment": f"⚠️ Yapay zeka yorumu üretilirken hata oluştu: {str(e)}"}
