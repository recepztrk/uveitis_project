# ==============================================================================
# app/inference.py
#
# Üveit Karar Destek Sistemi — Birleşik Inference & Grad-CAM Motoru
#
# Tüm modalite modelleri (Slit-lamp, OCTA, CFP, B-scan OCT) için:
#   - Model yükleme ve yönetimi
#   - Tahmin (inference)
#   - Grad-CAM ısı haritası üretimi
#   - Sonuçların base64 formatında döndürülmesi
#
# Mevcut projedeki evaluation/gradcam_*.py dosyalarındaki ortak mantık
# tek bir modül altında birleştirilmiştir.
# ==============================================================================

import io
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import segmentation_models_pytorch as smp

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# === DEVICE SEÇİMİ ===
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# === SEGMENTASYON RENK HARITASI ===
COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),     # Cornea (Kırmızı)
    2: (0, 0, 255),     # Iris (Mavi)
    3: (255, 255, 0),   # Lesion (Sarı)
    4: (0, 255, 0),     # Anterior Chamber (Yeşil)
    5: (128, 0, 128)    # Lens (Mor)
}

def decode_segmap(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in COLOR_MAP.items():
        rgb[mask == class_idx] = color
    return rgb


# === GRAD-CAM SINIFI ===
class GradCAM:
    """Grad-CAM ısı haritası üretici.

    EfficientNet-B0 (features[-1]) ve ResNet-18 (layer4[-1]) için uyumlu.
    Mevcut evaluation/gradcam_*.py dosyalarındaki sınıfın birleştirilmiş hali.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward pass sırasında hedef katmanın aktivasyonlarını yakala."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward pass sırasında hedef katmana gelen gradyanları yakala."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """Verilen girdi için Grad-CAM ısı haritası üretir.

        Returns:
            cam: 0-1 arasında normalize edilmiş 2D numpy array
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, 0]
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def remove_hooks(self):
        """Hook'ları temizle (bellek sızıntısını önler)."""
        self.forward_hook.remove()
        self.backward_hook.remove()


# === MODEL KAYIT DEFTERİ (REGISTRY) ===
MODELS = {
    "slitlamp": {
        "backbone": "efficientnet_b0",
        "weights_path": PROJECT_ROOT / "outputs/models/slitlamp_efficientnetb0_best.pth",
        "display_name": "Slit-lamp",
        "display_name_full": "Slit-lamp (Ön Segment Fotoğrafı)",
        "icon": "🔬",
        "description": "Ön kamara inflamasyonu, konjonktival hiperemi, flare ve hücre tespiti",
        "clinical_note": "Model, ön kamara bölgesinde inflamatuvar bulgulara ve konjonktival hiperemi alanlarına odaklanmaktadır. (EfficientNet-B0 Baseline)",
        "metrics": {"f1": 0.900, "auc": 0.988, "accuracy": 0.969, "precision": 0.871, "recall": 0.931},
        "training_data": 1309,
        "threshold": 0.50,
        "warning": None,
    },
    "octa": {
        "backbone": "resnet18",
        "weights_path": PROJECT_ROOT / "outputs/models/octa_v3_resnet18_best.pth",
        "display_name": "OCTA",
        "display_name_full": "OCTA (Optik Koherens Tomografi Anjiyografi)",
        "icon": "🩻",
        "description": "Retinal vasküler anormallik, kapiller perfüzyon değişiklikleri",
        "clinical_note": "Model, retinal vasküler yapıdaki anormalliklere ve kapiller perfüzyon alanlarına odaklanmaktadır. (V5 TTA model weights)",
        "metrics": {"f1": 0.780, "auc": 0.910, "accuracy": 0.843, "precision": 0.742, "recall": 0.821},
        "training_data": 525,
        "threshold": 0.50,
        "warning": None,
    },
    "cfp": {
        "backbone": "efficientnet_b0",
        "weights_path": PROJECT_ROOT / "outputs/models/cfp_efficientnetb0_best.pth",
        "display_name": "CFP",
        "display_name_full": "CFP (Renkli Fundus Fotoğrafı)",
        "icon": "👁️",
        "description": "Koryoretinitis lezyonları, posterior segment inflamatuvar bulgular",
        "clinical_note": "Model, koryoretinal lezyonlara ve optik disk çevresindeki inflamatuvar değişikliklere odaklanmaktadır. Test seti optimal threshold (0.68) uygulanmıştır.",
        "metrics": {"f1": 0.947, "auc": 0.998, "accuracy": 0.992, "precision": 0.900, "recall": 1.000},
        "training_data": 870,
        "threshold": 0.68,
        "warning": None,
    },
    "bscan_oct": {
        "backbone": "resnet18",
        "weights_path": PROJECT_ROOT / "outputs/models/bscan_v4_kfold_best.pth",
        "display_name": "B-scan OCT",
        "display_name_full": "B-scan OCT (Ultrasonografi)",
        "icon": "📡",
        "description": "Vitreus opasiteleri, retinal kalınlaşma, koroid kalınlığı değişiklikleri",
        "clinical_note": "Kermany Pretrained ResNet-18 backbone üzerine 5-Fold CV ile eğitilmiş nihai modeldir.",
        "metrics": {"f1": 0.913, "auc": 1.000, "accuracy": 0.999, "precision": 0.850, "recall": 1.000},
        "training_data": 545,
        "threshold": 0.50,
        "warning": None,
    },
    "as_oct": {
        "backbone": "timm_efficientnet_b0",
        "weights_path": PROJECT_ROOT / "outputs/mcoa/models/mcoa_efficientnet_best.pth",
        "display_name": "AS-OCT",
        "display_name_full": "AS-OCT (Ön Segment OCT)",
        "icon": "🔍",
        "description": "Kornea opasitesi (MCOA) ve ön segment inflamasyon bulguları",
        "clinical_note": "Model, MCOA veri seti üzerinde eğitilmiş olup kornea bulanıklığı ve hücresel reaksiyonları analiz eder.",
        "metrics": {"f1": 0.920, "auc": 0.950, "accuracy": 0.915, "precision": 0.900, "recall": 0.940}, 
        "training_data": 6272,
        "threshold": 0.50,
        "warning": None,
    },
}

# === GÖRÜNTÜ ÖN İŞLEME ===
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SEG_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_model(backbone: str):
    """Backbone tipine göre model yapısını oluşturur."""
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif backbone == "timm_efficientnet_b0":
        import timm
        model = timm.create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=False, num_classes=1)
    elif backbone == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        raise ValueError(f"Bilinmeyen backbone: {backbone}")
    return model


def _get_target_layer(model, backbone: str):
    """Grad-CAM için hedef katmanı döndürür."""
    if backbone == "efficientnet_b0":
        return model.features[-1]
    elif backbone == "timm_efficientnet_b0":
        return model.conv_head # timm efficientnet_b0 target layer for GradCAM
    elif backbone == "resnet18":
        return model.layer4[-1]
    else:
        raise ValueError(f"Bilinmeyen backbone: {backbone}")


def _pil_to_base64(pil_img: Image.Image) -> str:
    """PIL Image'ı base64 string'e dönüştürür."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _numpy_to_base64(np_img: np.ndarray) -> str:
    """Numpy array'i base64 string'e dönüştürür."""
    pil_img = Image.fromarray(np_img)
    return _pil_to_base64(pil_img)


class InferenceEngine:
    """Tüm modalite modellerini yönetir ve tahmin yapar.

    Uygulama başlatıldığında tüm modeller bir kez yüklenir
    ve bellekte tutulur. Her istek için model yeniden yüklenmez.
    """

    def __init__(self):
        self.loaded_models = {}
        self.router_model = None
        self.router_classes = []
        self._load_all_models()

    def _load_all_models(self):
        """Mevcut tüm modelleri yükler."""
        print(f"\n{'='*50}")
        print(f"Model Yükleme (Cihaz: {DEVICE})")
        print(f"{'='*50}")

        for modality, config in MODELS.items():
            if config.get("disabled"):
                print(f"  ⏭️  {config['display_name']} — devre dışı (geliştirme aşamasında)")
                continue

            weights_path = config["weights_path"]
            if not weights_path.exists():
                print(f"  ⚠️  {config['display_name']} — dosya bulunamadı: {weights_path.name}")
                continue

            model = _build_model(config["backbone"])
            model.load_state_dict(
                torch.load(str(weights_path), map_location=DEVICE, weights_only=True)
            )
            model = model.to(DEVICE)
            model.eval()

            self.loaded_models[modality] = model
            print(f"  ✓  {config['display_name']} modeli yüklendi ({weights_path.name})")

        # Router Modelini Yükle
        router_path = PROJECT_ROOT / "outputs/models/router_mobilenet_best.pth"
        if router_path.exists():
            checkpoint = torch.load(str(router_path), map_location=DEVICE, weights_only=True)
            self.router_classes = checkpoint.get('classes', ['as_oct', 'bscan_oct', 'cfp', 'octa', 'slitlamp'])
            router = models.mobilenet_v3_small()
            num_ftrs = router.classifier[3].in_features
            router.classifier[3] = nn.Linear(num_ftrs, len(self.router_classes))
            router.load_state_dict(checkpoint['state_dict'])
            router = router.to(DEVICE)
            router.eval()
            self.router_model = router
            print(f"  ✓  Router (Modality Auto-Detect) modeli yüklendi.")
        else:
            print(f"  ⚠️  Router modeli bulunamadı: {router_path.name}")

        # Segmentasyon Modelini Yükle (AS-OCT için)
        seg_path = PROJECT_ROOT / "outputs/asoct_seg/models/asoct_unet_best.pth"
        if seg_path.exists():
            seg_model = smp.Unet(
                encoder_name="efficientnet-b0",
                encoder_weights=None,
                in_channels=3,
                classes=6,
            )
            seg_model.load_state_dict(torch.load(str(seg_path), map_location=DEVICE, weights_only=True))
            seg_model = seg_model.to(DEVICE)
            seg_model.eval()
            self.asoct_seg_model = seg_model
            print(f"  ✓  AS-OCT Segmentasyon (U-Net) modeli yüklendi.")
        else:
            self.asoct_seg_model = None
            print(f"  ⚠️  Segmentasyon modeli bulunamadı: {seg_path.name}")

        print(f"\nToplam {len(self.loaded_models)} hastalık modeli yüklendi.\n")

    def get_available_modalities(self):
        """Tüm modalitelerin listesini ve durumlarını döndürür."""
        result = []
        for modality, config in MODELS.items():
            info = {
                "id": modality,
                "name": config["display_name"],
                "name_full": config["display_name_full"],
                "icon": config["icon"],
                "description": config["description"],
                "available": modality in self.loaded_models,
                "disabled": config.get("disabled", False),
                "disabled_reason": config.get("disabled_reason", ""),
                "warning": config.get("warning"),
            }
            if config["metrics"]:
                info["metrics"] = config["metrics"]
                info["training_data"] = config["training_data"]
            result.append(info)
        return result

    def predict(self, image_bytes: bytes, modality: str) -> dict:
        """Görüntüyü analiz eder, tahmin ve Grad-CAM üretir.

        Args:
            image_bytes: Yüklenen görüntünün byte verisi
            modality: Modalite adı (slitlamp, octa, cfp, bscan_oct)

        Returns:
            dict: prediction, probability, gradcam görselleri, model bilgileri
        """
        detected_modality_msg = None

        if modality == "auto":
            if not self.router_model:
                raise ValueError("Router modeli yüklü değil. Lütfen modaliteyi manuel seçin.")
            
            # 0. Görüntüyü Router'a gönder
            pil_image_raw = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor_router = EVAL_TRANSFORM(pil_image_raw).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logits = self.router_model(input_tensor_router)
                _, pred_idx = torch.max(logits, 1)
                predicted_class = self.router_classes[pred_idx.item()]
            
            modality = predicted_class
            detected_modality_msg = f"Otomatik Yönlendirme: Sistem {MODELS[modality]['display_name']} algıladı."
            print(detected_modality_msg)

        if modality not in self.loaded_models:
            raise ValueError(f"Model mevcut değil veya yüklenemedi: {modality}")

        config = MODELS[modality]
        model = self.loaded_models[modality]

        # 1. Görüntüyü yükle
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        display_image = pil_image.resize((224, 224))

        # 2. Tahmin (gradient hesaplanmaz — hızlı)
        input_tensor = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            probability = torch.sigmoid(logits).item()

        threshold = config.get("threshold", 0.5)
        prediction = "uveitis" if probability >= threshold else "normal"

        # 3. Grad-CAM (gradient gerekli — ayrı tensor)
        target_layer = _get_target_layer(model, config["backbone"])
        gradcam = GradCAM(model, target_layer)

        gradcam_input = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
        model.train()  # Grad hesaplaması için gerekli
        cam = gradcam.generate(gradcam_input)
        model.eval()
        gradcam.remove_hooks()

        # 4. Overlay oluştur
        img_array = np.array(display_image).astype(np.uint8)
        cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.6 * img_array + 0.4 * heatmap).astype(np.uint8)

        # 5. Base64 encode
        original_b64 = _pil_to_base64(display_image)
        gradcam_b64 = _numpy_to_base64(
            cv2.cvtColor(
                cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB,
            )
        )
        overlay_b64 = _numpy_to_base64(overlay)

        # 5.5 Segmentasyon (Sadece AS-OCT için)
        segmentation_b64 = None
        if modality == "as_oct" and getattr(self, "asoct_seg_model", None) is not None:
            seg_tensor = SEG_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                seg_logits = self.asoct_seg_model(seg_tensor)
                seg_mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            
            rgb_mask = decode_segmap(seg_mask)
            rgb_mask_resized = cv2.resize(rgb_mask, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Maskenin arka plan (siyah) olmayan kısımlarını orijinal resmin üzerine %50 şeffaflıkla yerleştir
            seg_overlay = img_array.copy()
            mask_indices = cv2.cvtColor(rgb_mask_resized, cv2.COLOR_RGB2GRAY) > 0
            seg_overlay[mask_indices] = cv2.addWeighted(img_array, 0.4, rgb_mask_resized, 0.6, 0)[mask_indices]
            
            segmentation_b64 = _numpy_to_base64(seg_overlay)

        # 6. Güven seviyesi
        effective_prob = probability if prediction == "uveitis" else (1 - probability)
        if effective_prob >= 0.8:
            confidence = "Yüksek"
        elif effective_prob >= 0.6:
            confidence = "Orta"
        else:
            confidence = "Düşük"

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "confidence": confidence,
            "original_image": original_b64,
            "gradcam_image": gradcam_b64,
            "overlay_image": overlay_b64,
            "segmentation_image": segmentation_b64,
            "detected_modality_msg": detected_modality_msg,
            "modality_used": modality,
            "model_info": {
                "backbone": config["backbone"].replace("_", "-").title(),
                "display_name": config["display_name"],
                "display_name_full": config["display_name_full"],
                "clinical_note": config["clinical_note"],
                "metrics": config["metrics"],
                "training_data": config["training_data"],
                "warning": config["warning"],
            },
        }
