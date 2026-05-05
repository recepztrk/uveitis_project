// ==============================================================================
// app.js — Üveit Karar Destek Sistemi Frontend
//
// Tek sayfa uygulamasının (SPA) tüm istemci tarafı mantığını yönetir:
//   - Görüntü yükleme (dosya seçimi + drag & drop)
//   - Hazır örnek vaka yükleme (API üzerinden)
//   - Modalite seçimi ve adım göstergeleri
//   - FastAPI backend’e analiz isteği gönderme (POST /api/predict)
//   - Sonuç gösterimi (tahmin, olasılık, Grad-CAM görselleri)
//   - Oturum analiz geçmişi (sayfa yenilenene kadar saklanır)
//
// Backend API Endpoint'leri:
//   GET  /api/models           → Mevcut modellerin listesi
//   POST /api/predict           → Görüntü analizi (tahmin + Grad-CAM)
//   GET  /api/samples/{mod}     → Hazır örnek vaka dosyaları
// ==============================================================================

// === STATE (Uygulama Durumu) ===
// Sayfa yenilenene kadar belleğte tutulan global değişkenler
let selectedFile = null;        // Kullanıcının yüklediği File nesnesi
let selectedModality = null;    // Seçili modalite ID'si (slitlamp, octa, cfp, bscan_oct)
let isAnalyzing = false;        // Analiz sürüyorsa true (çift tıklamayı önler)
let sessionHistory = [];        // Bu oturumdaki tüm analiz sonuçları
let modalityData = [];          // /api/models'den yüklenen modalite bilgileri

// === DOM ELEMENTS ===
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const previewName = document.getElementById('preview-name');
const previewMeta = document.getElementById('preview-meta');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const warningBanner = document.getElementById('warning-banner');
const historySection = document.getElementById('history-section');
const historyGrid = document.getElementById('history-grid');

// === INITIALIZATION ===
document.addEventListener('DOMContentLoaded', () => {
    loadModalities();
    setupUploadHandlers();
    updateStepIndicators();
});

// === LOAD MODALITIES (Modalite Verilerini Yükle) ===
// Sayfa açıldığında /api/models endpoint'inden tüm modalitelerin
// ad, ikon, metrik, durum bilgilerini çeker ve kartları oluşturur.
async function loadModalities() {
    try {
        const response = await fetch('/api/models');
        modalityData = await response.json();
        renderModalityCards();
    } catch (error) {
        console.error('Modalite verileri yüklenemedi:', error);
    }
}

// Her modalite için seçilebilir kart HTML'i üretir.
// Devre dışı modeller (AS-OCT) tıklanamaz hale getirilir.
function renderModalityCards() {
    const grid = document.getElementById('modality-grid');
    grid.innerHTML = '';

    modalityData.forEach(mod => {
        const card = document.createElement('div');
        card.className = `modality-card${mod.disabled ? ' disabled' : ''}`;
        card.dataset.modality = mod.id;

        let metricsHTML = '';
        if (mod.metrics) {
            metricsHTML = `
                <div class="card-metrics">
                    <span>F1: <strong>${mod.metrics.f1.toFixed(3)}</strong></span>
                    <span>AUC: <strong>${mod.metrics.auc.toFixed(3)}</strong></span>
                    <br><span>Veri: <strong>${mod.training_data.toLocaleString()}</strong></span>
                </div>`;
        }

        let badgeHTML = '';
        if (mod.disabled) {
            badgeHTML = `<span class="card-badge">Yakında</span>`;
        } else if (mod.warning) {
            badgeHTML = `<span class="card-badge">Sınırlı</span>`;
        }

        card.innerHTML = `
            ${badgeHTML}
            <span class="card-icon">${mod.icon}</span>
            <div class="card-name">${mod.name}</div>
            <div class="card-desc">${mod.description}</div>
            ${metricsHTML}
        `;

        if (!mod.disabled && mod.available) {
            card.addEventListener('click', () => selectModality(mod.id));
        }

        grid.appendChild(card);
    });
}

// === UPLOAD HANDLERS (Görüntü Yükleme İşlemleri) ===
// Tıklama ile dosya seçimi ve sürükle-bırak (drag & drop) destekler.
function setupUploadHandlers() {
    // Click to upload
    uploadArea.addEventListener('click', (e) => {
        if (e.target.closest('.sample-btn') || e.target.closest('.preview-change')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

// Seçilen dosyayı doğrular, önizleme gösterir ve state'i günceller.
// FileReader ile base64 önizleme üretilir (upload alanında görüntülenir).
function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Lütfen bir görüntü dosyası seçin (JPG, PNG, TIFF).');
        return;
    }

    selectedFile = file;

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    previewName.textContent = file.name;
    const sizeMB = (file.size / 1024 / 1024).toFixed(2);
    previewMeta.textContent = `${sizeMB} MB • ${file.type}`;

    uploadArea.classList.add('has-image');
    updateStepIndicators();
    updateAnalyzeButton();
}

// === SAMPLE CASES (Örnek Vaka Yükleme) ===
// static/samples/ klasöründeki hazır test görüntülerini yükler.
// Görüntü fetch edilir, File nesnesine dönüştürülr ve ilgili modalite seçilir.
async function loadSample(modality, filename) {
    try {
        const response = await fetch(`/static/samples/${modality}/${filename}`);
        const blob = await response.blob();
        const file = new File([blob], filename, { type: blob.type });
        handleFile(file);
        selectModality(modality);
    } catch (error) {
        console.error('Örnek vaka yüklenemedi:', error);
    }
}

// İlgili modalitenin örnek vakalarını dropdown olarak gösterir.
// /api/samples/{modality} endpoint'inden dosya listesini çeker.
async function showSamples(modality) {
    // Close any open dropdowns
    document.querySelectorAll('.sample-dropdown').forEach(d => d.classList.remove('show'));

    const dropdown = document.getElementById(`samples-${modality}`);
    if (!dropdown) return;

    try {
        const response = await fetch(`/api/samples/${modality}`);
        const samples = await response.json();

        if (samples.length === 0) {
            dropdown.innerHTML = '<div class="sample-item" style="color: var(--text-muted);">Örnek vaka bulunamadı</div>';
        } else {
            dropdown.innerHTML = samples.map(s => `
                <div class="sample-item" onclick="event.stopPropagation(); loadSample('${modality}', '${s.filename}')">
                    <span>${s.filename}</span>
                    <span class="sample-badge ${s.label}">${s.label_display}</span>
                </div>
            `).join('');
        }

        dropdown.classList.toggle('show');
    } catch (error) {
        console.error('Örnekler yüklenemedi:', error);
    }
}

// Close dropdowns on outside click
document.addEventListener('click', (e) => {
    if (!e.target.closest('.sample-btn')) {
        document.querySelectorAll('.sample-dropdown').forEach(d => d.classList.remove('show'));
    }
});

// === MODALITY SELECTION (Modalite Seçimi) ===
// Seçilen kartı vurgular, varsa uyarı banner'ını gösterir (orn: B-scan sınırlı veri).
function selectModality(modality) {
    selectedModality = modality;

    // Update card selection
    document.querySelectorAll('.modality-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.modality === modality);
    });

    // Show warning if applicable
    const mod = modalityData.find(m => m.id === modality);
    if (mod && mod.warning) {
        warningBanner.textContent = mod.warning;
        warningBanner.classList.add('show');
    } else {
        warningBanner.classList.remove('show');
    }

    updateStepIndicators();
    updateAnalyzeButton();
}

// === STEP INDICATORS (Adım Göstergeleri) ===
// 3 adımlı ilerleme çubuğunu günceller:
//   Adım 1: Görüntü yüklendi mi? → completed (yeşil)
//   Adım 2: Modalite seçildi mi? → completed (yeşil)
//   Adım 3: Her ikisi de tamam mı? → active (mavi, analiz hazır)
function updateStepIndicators() {
    const step1 = document.getElementById('step-1');
    const step2 = document.getElementById('step-2');
    const step3 = document.getElementById('step-3');
    const line1 = document.getElementById('step-line-1');
    const line2 = document.getElementById('step-line-2');

    // Step 1: Upload
    if (selectedFile) {
        step1.classList.add('completed');
        step1.classList.remove('active');
        line1.classList.add('completed');
    } else {
        step1.classList.add('active');
        step1.classList.remove('completed');
        line1.classList.remove('completed');
    }

    // Step 2: Modality
    if (selectedModality) {
        step2.classList.add('completed');
        step2.classList.remove('active');
        line2.classList.add('completed');
    } else if (selectedFile) {
        step2.classList.add('active');
        step2.classList.remove('completed');
        line2.classList.remove('completed');
    } else {
        step2.classList.remove('active', 'completed');
        line2.classList.remove('completed');
    }

    // Step 3: Analyze
    if (selectedFile && selectedModality) {
        step3.classList.add('active');
    } else {
        step3.classList.remove('active', 'completed');
    }
}

// === ANALYZE BUTTON (Analiz Butonu) ===
// Buton sadece dosya + modalite seçildiğinde aktif olur.
function updateAnalyzeButton() {
    analyzeBtn.disabled = !selectedFile || !selectedModality || isAnalyzing;
}

// Ana analiz fonksiyonu: FormData ile görüntüyü POST /api/predict'e gönderir.
// Backend modeli çalıştırır, tahmin + Grad-CAM üretir ve JSON döner.
async function analyze() {
    if (!selectedFile || !selectedModality || isAnalyzing) return;

    isAnalyzing = true;
    analyzeBtn.classList.add('loading');
    updateAnalyzeButton();

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('modality', selectedModality);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analiz hatası');
        }

        const result = await response.json();
        showResults(result);
        addToHistory(result);

    } catch (error) {
        alert(`Analiz hatası: ${error.message}`);
    } finally {
        isAnalyzing = false;
        analyzeBtn.classList.remove('loading');
        updateAnalyzeButton();
    }
}

// === RESULTS DISPLAY (Sonuç Gösterimi) ===
// API'den dönen JSON sonuçlarını arayüze yansıtır:
//   - Tahmin badge'ı (kırmızı: üveit, yeşil: normal)
//   - Olasılık çubuğu (animasyonlu)
//   - Base64 görseller (orijinal, Grad-CAM ısı haritası, overlay)
//   - Model metrikleri ve klinik not
function showResults(result) {
    resultsSection.classList.add('show');

    // Prediction badge
    const badge = document.getElementById('prediction-badge');
    badge.className = `prediction-badge ${result.prediction}`;
    badge.textContent = result.prediction === 'uveitis' ? '🔴 ÜVEİT ŞÜPHESİ' : '🟢 NORMAL';

    // Probability
    const probPercent = (result.probability * 100).toFixed(1);
    document.getElementById('prob-value').textContent = `%${probPercent}`;
    const probFill = document.getElementById('prob-fill');
    probFill.className = `prob-fill ${result.prediction}`;
    // Çift requestAnimationFrame: Tarayıcının önce 0%'yı render etmesini,
    // sonra hedef değere animasyonla geçmesini sağlar (CSS transition ile).
    probFill.style.width = '0%';
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            probFill.style.width = `${probPercent}%`;
        });
    });

    // Confidence
    const confBadge = document.getElementById('confidence-badge');
    const confMap = { 'Yüksek': 'high', 'Orta': 'medium', 'Düşük': 'low' };
    confBadge.className = `confidence-badge ${confMap[result.confidence] || 'medium'}`;
    confBadge.textContent = `Güven: ${result.confidence}`;

    // Images
    document.getElementById('result-img-original').src = `data:image/png;base64,${result.original_image}`;
    document.getElementById('result-img-gradcam').src = `data:image/png;base64,${result.gradcam_image}`;
    document.getElementById('result-img-overlay').src = `data:image/png;base64,${result.overlay_image}`;
    showImageTab('overlay');

    // Model info
    const info = result.model_info;
    document.getElementById('info-backbone').textContent = info.backbone;
    document.getElementById('info-f1').textContent = info.metrics.f1.toFixed(3);
    document.getElementById('info-auc').textContent = info.metrics.auc.toFixed(3);
    document.getElementById('info-data').textContent = info.training_data.toLocaleString();

    // Clinical note
    document.getElementById('clinical-note-text').textContent = info.clinical_note;

    // Model warning
    const warningEl = document.getElementById('model-warning');
    if (info.warning) {
        warningEl.textContent = info.warning;
        warningEl.style.display = 'block';
    } else {
        warningEl.style.display = 'none';
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// === IMAGE TAB SWITCHING (Görüntü Sekme Geçişi) ===
// Sonuç panelinde 3 görüntü arasında geçiş: Overlay, Orijinal, Grad-CAM.
// Overlay = orijinal görüntü + Grad-CAM ısı haritası üst üste bindirilmiş hali.
function showImageTab(tab) {
    document.querySelectorAll('.image-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tab);
    });

    document.getElementById('result-img-original').style.display = tab === 'original' ? 'block' : 'none';
    document.getElementById('result-img-gradcam').style.display = tab === 'gradcam' ? 'block' : 'none';
    document.getElementById('result-img-overlay').style.display = tab === 'overlay' ? 'block' : 'none';
}

// === SESSION HISTORY (Oturum Geçmişi) ===
// Her başarılı analiz sonucunu belleğe kaydeder ve alt kısımda listeler.
// Sayfa yenilendiğinde geçmiş sıfırlanır (localStorage kullanılmaz).
function addToHistory(result) {
    sessionHistory.push({
        modality: result.model_info.display_name,
        prediction: result.prediction,
        probability: result.probability,
        thumbnail: result.original_image,
        timestamp: new Date().toLocaleTimeString('tr-TR'),
    });
    renderHistory();
}

// Geçmiş kartlarını HTML olarak üretir ve görüntüler.
function renderHistory() {
    if (sessionHistory.length === 0) {
        historySection.style.display = 'none';
        return;
    }

    historySection.style.display = 'block';
    historyGrid.innerHTML = sessionHistory.map((h, i) => `
        <div class="history-card">
            <img class="history-thumb" src="data:image/png;base64,${h.thumbnail}" alt="Analiz ${i + 1}">
            <div class="history-info">
                <div class="history-modality">${h.modality}</div>
                <div class="history-result ${h.prediction}">
                    ${h.prediction === 'uveitis' ? 'Üveit Şüphesi' : 'Normal'}
                </div>
                <div class="history-prob">%${(h.probability * 100).toFixed(1)} • ${h.timestamp}</div>
            </div>
        </div>
    `).join('');
}

// Geçmişi sıfırlar ve bölümü gizler.
function clearHistory() {
    sessionHistory = [];
    renderHistory();
}
