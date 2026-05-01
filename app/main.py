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

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from app.inference import InferenceEngine

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

app = FastAPI(
    title="Üveit Karar Destek Sistemi",
    description="Multimodal derin öğrenme tabanlı oftalmolojik tanı desteği",
    version="1.0.0",
)

# Statik dosyalar ve şablonlar
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
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
    """Belirtilen modalite için hazır örnek vaka dosyalarını listeler."""
    samples_dir = APP_DIR / "static" / "samples" / modality

    if not samples_dir.exists():
        return []

    samples = []
    for img_path in sorted(samples_dir.iterdir()):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
            # Dosya adından etiket tahmin et
            name_lower = img_path.stem.lower()
            if "uveitis" in name_lower or name_lower.startswith("u_"):
                label = "uveitis"
                label_display = "Üveit"
            else:
                label = "normal"
                label_display = "Normal"

            samples.append({
                "filename": img_path.name,
                "url": f"/static/samples/{modality}/{img_path.name}",
                "label": label,
                "label_display": label_display,
            })

    return samples
