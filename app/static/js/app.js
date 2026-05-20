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
let currentResult = null;       // Son yapılan analizin sonuç objesi

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

        // Add Auto Detect Modality
        modalityData.unshift({
            id: 'auto',
            name: 'Otomatik Tespit (Auto)',
            icon: '🤖',
            description: 'Görüntünün cihaz tipini yapay zekaya bırakın. Router modelimiz görüntüyü doğru hastalık modeline yönlendirecektir.',
            available: true,
            metrics: null
        });

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
// Görüntü fetch edilir, File nesnesine dönüştürülür ve ilgili modalite seçilir.
async function loadSample(modality, filename, url) {
    try {
        const response = await fetch(url);
        const blob = await response.blob();

        // Klasör ismi içerebilen filename'den sadece dosya adını ayıkla
        const cleanFilename = filename.split('/').pop();
        const file = new File([blob], cleanFilename, { type: blob.type });

        handleFile(file);
        selectModality(modality);
    } catch (error) {
        console.error('Örnek vaka yüklenemedi:', error);
    }
}

// İlgili modalitenin örnek vakalarını dropdown olarak gösterir.
// /api/samples/{modality} endpoint'inden dosya listesini çeker.
async function showSamples(modality) {
    // Tüm açık dropdown'ları kapat
    document.querySelectorAll('.sample-dropdown').forEach(d => d.classList.remove('show'));

    const dropdown = document.getElementById(`samples-${modality}`);
    if (!dropdown) return;

    try {
        const response = await fetch(`/api/samples/${modality}`);
        const samples = await response.json();

        if (samples.length === 0) {
            dropdown.innerHTML = '<div class="sample-item" style="color: var(--text-muted);">Örnek vaka bulunamadı</div>';
        } else {
            const isASOCT = modality === 'as_oct';
            const abnormalLabel = isASOCT ? 'MCOA (Korneal Opasite)' : 'Üveit (Aktif İnflamasyon)';
            const normalLabel = 'Normal (Fizyolojik)';

            // Tıklanınca rastgele resim seçen global handler ekleyelim (eğer yoksa)
            if (!window.handleRandomSample) {
                window.handleRandomSample = function(mod, label, allSamplesStr) {
                    const allSamples = JSON.parse(decodeURIComponent(allSamplesStr));
                    const filtered = allSamples.filter(s => s.label === label);
                    if (filtered.length > 0) {
                        const randomItem = filtered[Math.floor(Math.random() * filtered.length)];
                        loadSample(mod, randomItem.filename, randomItem.url);
                    }
                    document.querySelectorAll('.sample-dropdown').forEach(d => d.classList.remove('show'));
                };
            }

            const encodedSamples = encodeURIComponent(JSON.stringify(samples));

            dropdown.innerHTML = `
                <div class="sample-item" onclick="event.stopPropagation(); window.handleRandomSample('${modality}', 'uveitis', '${encodedSamples}')">
                    <span style="font-weight:600;">🔴 Anormal Vaka</span>
                    <span class="sample-badge uveitis" style="font-size:10px;">${abnormalLabel}</span>
                </div>
                <div class="sample-item" onclick="event.stopPropagation(); window.handleRandomSample('${modality}', 'normal', '${encodedSamples}')">
                    <span style="font-weight:600;">🟢 Sağlıklı Vaka</span>
                    <span class="sample-badge normal" style="font-size:10px;">${normalLabel}</span>
                </div>
            `;
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
    currentResult = result;
    resultsSection.classList.add('show');

    // Prediction badge ve Label
    const badge = document.getElementById('prediction-badge');
    const probLabel = document.querySelector('.prob-label');

    badge.className = `prediction-badge ${result.prediction}`;

    if (result.model_info.display_name === 'AS-OCT') {
        badge.textContent = result.prediction === 'uveitis' ? '🔴 ANORMAL BULGU' : '🟢 NORMAL';
        probLabel.textContent = 'Patoloji Olasılığı';
    } else {
        badge.textContent = result.prediction === 'uveitis' ? '🔴 ÜVEİT ŞÜPHESİ' : '🟢 NORMAL';
        probLabel.textContent = 'Üveit Olasılığı';
    }

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

    const toggles = document.getElementById('view-toggles');
    const btnGradcam = document.getElementById('btn-gradcam');
    const btnSeg = document.getElementById('btn-segmentation');

    if (result.segmentation_image) {
        toggles.style.display = 'flex';
        // Varsayılan olarak Grad-CAM göster
        switchView('gradcam');
    } else {
        toggles.style.display = 'none';
        document.getElementById('result-img-overlay').src = `data:image/png;base64,${result.overlay_image}`;
        document.getElementById('slider-instructions').textContent = 'Fareyi sürükleyerek YZ odaklanmasını (Grad-CAM) inceleyin';
    }

    // Initialize Slider
    initSlider();

    // Modality Detection Banner
    const detectedBanner = document.getElementById('detected-modality-banner');
    if (result.detected_modality_msg) {
        detectedBanner.textContent = result.detected_modality_msg;
        detectedBanner.style.display = 'block';
    } else {
        detectedBanner.style.display = 'none';
    }

    // Model info
    const info = result.model_info;
    document.getElementById('info-backbone').textContent = info.backbone;
    document.getElementById('info-f1').textContent = info.metrics.f1.toFixed(3);
    document.getElementById('info-auc').textContent = info.metrics.auc.toFixed(3);
    document.getElementById('info-data').textContent = info.training_data.toLocaleString();

    // Clinical note
    document.getElementById('clinical-note-text').textContent = info.clinical_note;

    // === GÖRÜNTÜ KALİTE SKORU ===
    const qualityRow = document.getElementById('quality-badge-row');
    const qualityBadge = document.getElementById('quality-badge');
    const qualityIssues = document.getElementById('quality-issues');
    if (result.quality_info) {
        const q = result.quality_info;
        qualityRow.style.display = 'flex';
        qualityBadge.textContent = q.score_label;
        qualityBadge.className = `quality-badge quality-${q.score}`;
        if (q.issues && q.issues.length > 0) {
            qualityIssues.textContent = '— ' + q.issues.join('; ');
            qualityIssues.style.display = 'inline';
        } else {
            qualityIssues.textContent = '';
        }
    } else {
        qualityRow.style.display = 'none';
    }

    // === BELİRSİZLİK BÖLGESİ UYARISI ===
    const uncertaintyBox = document.getElementById('uncertainty-warning');
    if (result.uncertainty_zone) {
        const probPercent2 = (result.probability * 100).toFixed(1);
        document.getElementById('uncertainty-prob').textContent = `%${probPercent2}`;
        uncertaintyBox.style.display = 'block';
    } else {
        uncertaintyBox.style.display = 'none';
    }

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

    // Show PDF button but keep it disabled until AI comment is ready
    const pdfBtn = document.getElementById('pdf-download-btn');
    pdfBtn.style.display = 'flex';
    pdfBtn.disabled = true;
    document.getElementById('pdf-btn-icon').textContent = '⏳';
    document.getElementById('pdf-btn-text').textContent = 'AI Rapor Bekleniyor...';

    // Call Gemini API asynchronously
    generateAIComment(result);
}

// === GEMINI AI INTEGRATION ===
async function generateAIComment(result) {
    const aiBox = document.getElementById('ai-comment-box');
    const aiText = document.getElementById('ai-comment-text');
    const aiTyping = document.getElementById('ai-typing-indicator');

    // Reset and show box
    aiBox.style.display = 'block';
    aiText.innerHTML = '';
    aiTyping.style.display = 'inline-flex';

    try {
        const response = await fetch('/api/generate_comment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                modality: result.modality_used,
                prediction: result.prediction,
                probability: result.probability,
                confidence: result.confidence,
                model_name: result.model_info.display_name_full,
                model_backbone: result.model_info.backbone,
                model_f1: result.model_info.metrics.f1,
                model_auc: result.model_info.metrics.auc,
                clinical_note: result.model_info.clinical_note,
                training_data: result.model_info.training_data
            })
        });

        const data = await response.json();
        const comment = data.comment || 'Yorum alınamadı.';
        const errorType = data.error_type || null;

        // Hide typing indicator
        aiTyping.style.display = 'none';

        // Kota / auth hatası → özel banner göster
        if (errorType === 'quota_exceeded') {
            aiText.innerHTML = `
                <div style="display:flex;align-items:center;gap:10px;padding:12px 16px;background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);border-radius:8px;">
                    <span style="font-size:22px;">⏳</span>
                    <div>
                        <div style="font-weight:700;color:#f59e0b;margin-bottom:3px;">Günlük API Kotası Doldu</div>
                        <div style="font-size:12px;color:var(--text-secondary);">
                            Gemini ücretsiz planının günlük istek limiti aşıldı. Kota sıfırlanana kadar AI yorum üretilemez.<br>
                            <span style="color:var(--text-muted);font-size:11px;">PDF rapor AI yorumu olmadan da indirilebilir.</span>
                        </div>
                    </div>
                </div>`;
            populatePDFTemplate(currentResult, '⏳ Günlük API kotası doldu — AI yorumu bu raporda mevcut değil.');
        } else if (errorType === 'auth_error') {
            aiText.innerHTML = `
                <div style="display:flex;align-items:center;gap:10px;padding:12px 16px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);border-radius:8px;">
                    <span style="font-size:22px;">🔑</span>
                    <div>
                        <div style="font-weight:700;color:#ef4444;margin-bottom:3px;">API Anahtarı Hatası</div>
                        <div style="font-size:12px;color:var(--text-secondary);">.env dosyasındaki GEMINI_API_KEY geçersiz veya izin yetersiz.</div>
                    </div>
                </div>`;
            populatePDFTemplate(currentResult, '🔑 API anahtarı hatası — AI yorumu bu raporda mevcut değil.');
        } else {
            // Başarılı yorum → typewriter efekti
            let i = 0;
            function typeWriter() {
                if (i < comment.length) {
                    aiText.innerHTML += comment.substring(i, i + 3);
                    i += 3;
                    setTimeout(typeWriter, 5);
                } else {
                    populatePDFTemplate(currentResult, comment);
                    const pdfBtn = document.getElementById('pdf-download-btn');
                    pdfBtn.disabled = false;
                    document.getElementById('pdf-btn-icon').textContent = '📄';
                    document.getElementById('pdf-btn-text').textContent = 'Klinik Rapor PDF İndir';
                }
            }
            typeWriter();
            return; // PDF enable typeWriter içinde yapılıyor
        }

        // Hata durumlarında PDF yine de etkin
        const pdfBtn = document.getElementById('pdf-download-btn');
        pdfBtn.disabled = false;
        document.getElementById('pdf-btn-icon').textContent = '📄';
        document.getElementById('pdf-btn-text').textContent = 'Klinik Rapor PDF İndir';

    } catch (error) {
        aiTyping.style.display = 'none';
        aiText.innerHTML = `<div style="padding:10px 14px;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);border-radius:8px;color:#ef4444;">
            ⚠️ Bağlantı hatası — sunucuya ulaşılamıyor.</div>`;
        populatePDFTemplate(currentResult, 'Bağlantı hatası — AI yorumu bu raporda mevcut değil.');
        const pdfBtn = document.getElementById('pdf-download-btn');
        pdfBtn.disabled = false;
        document.getElementById('pdf-btn-icon').textContent = '📄';
        document.getElementById('pdf-btn-text').textContent = 'Klinik Rapor PDF İndir';
    }
}


// === PDF TEMPLATE POPULATION ===
function populatePDFTemplate(result, aiComment) {
    const info = result.model_info;
    const isASOCT = result.modality_used === 'as_oct';
    const probPercent = (result.probability * 100).toFixed(1);
    const now = new Date();
    const dateStr = now.toLocaleDateString('tr-TR', { day: '2-digit', month: 'long', year: 'numeric' });
    const timeStr = now.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });
    const reportId = 'RPT-' + Math.random().toString(36).substr(2, 8).toUpperCase();

    // Header meta
    document.getElementById('pdf-report-date').textContent = `${dateStr}, ${timeStr}`;
    document.getElementById('pdf-report-id').textContent = reportId;
    document.getElementById('pdf-footer-date').textContent = `${dateStr} ${timeStr}`;

    // Decision banner
    const banner = document.getElementById('pdf-decision-banner');
    if (result.prediction === 'uveitis') {
        banner.textContent = isASOCT ? '🔴 ANORMAL BULGU — Korneal Opasite (MCOA) ile Uyumlu' : '🔴 ÜVEİT ŞÜPHESİ — Anormal Oftalmolojik Bulgu';
        banner.style.background = '#fff1f2';
        banner.style.border = '2px solid #fca5a5';
        banner.style.color = '#991b1b';
    } else {
        banner.textContent = '🟢 NORMAL — Fizyolojik Sınırlar İçinde';
        banner.style.background = '#f0fdf4';
        banner.style.border = '2px solid #86efac';
        banner.style.color = '#14532d';
    }

    // Metrics
    document.getElementById('pdf-modality').textContent = info.display_name_full || info.display_name;
    const probEl = document.getElementById('pdf-probability');
    probEl.textContent = `%${probPercent}`;
    probEl.style.color = result.prediction === 'uveitis' ? '#dc2626' : '#16a34a';
    document.getElementById('pdf-confidence').textContent = result.confidence;
    document.getElementById('pdf-auc').textContent = info.metrics.auc.toFixed(3);

    // Images
    document.getElementById('pdf-img-original').src = `data:image/png;base64,${result.original_image}`;
    const heatmapSrc = result.segmentation_image
        ? `data:image/png;base64,${result.segmentation_image}`
        : `data:image/png;base64,${result.overlay_image}`;
    document.getElementById('pdf-img-heatmap').src = heatmapSrc;
    if (isASOCT) {
        document.getElementById('pdf-heatmap-label').textContent = 'U-Net Anatomik Segmentasyon Haritası';
        document.getElementById('pdf-heatmap-note').textContent = 'Segmentasyon maskesi, modelin ön segment yapılarını (kornea, iris) anatomik olarak ayırdığını göstermektedir.';
    }

    // AI Comment
    document.getElementById('pdf-ai-comment').textContent = aiComment;

    // Technical metrics
    document.getElementById('pdf-backbone').textContent = info.backbone;
    document.getElementById('pdf-f1').textContent = info.metrics.f1.toFixed(3);
    document.getElementById('pdf-training').textContent = info.training_data.toLocaleString() + ' görüntü';
    document.getElementById('pdf-auc2').textContent = info.metrics.auc.toFixed(3);
    document.getElementById('pdf-clinical-note').textContent = info.clinical_note;
}

// === PDF DOWNLOAD (html2canvas onclone + jsPDF) ===
function downloadPDF() {
    const btn = document.getElementById('pdf-download-btn');
    btn.disabled = true;
    document.getElementById('pdf-btn-icon').textContent = '⏳';
    document.getElementById('pdf-btn-text').textContent = 'PDF Oluşturuluyor...';

    const source = document.getElementById('pdf-content');
    const reportId = document.getElementById('pdf-report-id').textContent || 'RPT';
    const filename = `uveit_ai_rapor_${reportId}.pdf`;

    // html2canvas — onclone içinde template görünür yapılıyor
    html2canvas(source, {
        scale: 2,
        useCORS: false,
        allowTaint: true,
        backgroundColor: '#ffffff',
        logging: false,
        onclone: function (clonedDoc) {
            // Klonlanmış belgede pdf-template'i görünür yap
            const tpl = clonedDoc.getElementById('pdf-template');
            tpl.style.display = 'block';
            tpl.style.position = 'absolute';
            tpl.style.top = '0px';
            tpl.style.left = '0px';
            tpl.style.width = '794px';
            tpl.style.zIndex = '0';
            tpl.style.background = '#ffffff';
        }
    }).then(function (canvas) {
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF({ orientation: 'p', unit: 'mm', format: 'a4' });
        const pageW = pdf.internal.pageSize.getWidth();   // 210 mm
        const pageH = pdf.internal.pageSize.getHeight();  // 297 mm

        const imgData = canvas.toDataURL('image/jpeg', 0.93);
        const imgH = (canvas.height * pageW) / canvas.width; // mm cinsinden yükseklik

        if (imgH <= pageH) {
            // Tek sayfaya sığıyor
            pdf.addImage(imgData, 'JPEG', 0, 0, pageW, imgH);
        } else {
            // Çok sayfalı
            let offset = 0;
            while (offset < imgH) {
                if (offset > 0) pdf.addPage();
                pdf.addImage(imgData, 'JPEG', 0, -offset, pageW, imgH);
                offset += pageH;
            }
        }

        pdf.save(filename);

        btn.disabled = false;
        document.getElementById('pdf-btn-icon').textContent = '✅';
        document.getElementById('pdf-btn-text').textContent = 'PDF İndirildi!';
        setTimeout(() => {
            document.getElementById('pdf-btn-icon').textContent = '📄';
            document.getElementById('pdf-btn-text').textContent = 'Klinik Rapor PDF İndir';
        }, 3000);

    }).catch(function (err) {
        console.error('PDF hatası:', err);
        btn.disabled = false;
        document.getElementById('pdf-btn-icon').textContent = '❌';
        document.getElementById('pdf-btn-text').textContent = 'Hata — Tekrar dene';
        setTimeout(() => {
            document.getElementById('pdf-btn-icon').textContent = '📄';
            document.getElementById('pdf-btn-text').textContent = 'Klinik Rapor PDF İndir';
        }, 3000);
    });
}




// === IMAGE SLIDER (Öncesi/Sonrası) ===
// Fare veya dokunmatik ile orijinal ve AI (Grad-CAM) görüntüleri arasında geçiş sağlar.
function initSlider() {
    const container = document.getElementById('img-comp-container');
    const overlay = document.getElementById('result-img-overlay');
    const slider = document.getElementById('img-comp-slider');

    // Her analizde pozisyonu %50'ye sıfırla
    slider.style.left = "50%";
    overlay.style.clipPath = `inset(0% 50% 0% 0%)`;

    let isDragging = false;

    function slide(e) {
        if (!isDragging) return;

        let rect = container.getBoundingClientRect();
        let clientX = e.type.includes('mouse') ? e.clientX : e.touches[0].clientX;
        let x = clientX - rect.left;

        if (x < 0) x = 0;
        if (x > rect.width) x = rect.width;

        let percent = (x / rect.width) * 100;

        slider.style.left = percent + "%";
        overlay.style.clipPath = `inset(0% ${100 - percent}% 0% 0%)`;
    }

    // Sürükleme Olayları (Mouse)
    slider.onmousedown = () => isDragging = true;
    window.addEventListener('mouseup', () => isDragging = false);
    window.addEventListener('mousemove', slide);

    // Sürükleme Olayları (Touch - Mobil Uyumluluk)
    slider.ontouchstart = () => isDragging = true;
    window.addEventListener('touchend', () => isDragging = false);
    window.addEventListener('touchmove', slide, { passive: true });
}

function switchView(viewType) {
    if (!currentResult) return;

    const btnGradcam = document.getElementById('btn-gradcam');
    const btnSeg = document.getElementById('btn-segmentation');
    const overlayImg = document.getElementById('result-img-overlay');
    const instructions = document.getElementById('slider-instructions');

    if (viewType === 'gradcam') {
        btnGradcam.style.background = 'var(--primary)';
        btnGradcam.style.color = 'white';
        btnSeg.style.background = 'rgba(14, 165, 233, 0.1)';
        btnSeg.style.color = 'var(--text-primary)';

        overlayImg.src = `data:image/png;base64,${currentResult.overlay_image}`;
        instructions.textContent = 'Fareyi sürükleyerek YZ odaklanmasını (Grad-CAM) inceleyin';
    } else if (viewType === 'segmentation') {
        btnSeg.style.background = 'var(--primary)';
        btnSeg.style.color = 'white';
        btnGradcam.style.background = 'rgba(14, 165, 233, 0.1)';
        btnGradcam.style.color = 'var(--text-primary)';

        overlayImg.src = `data:image/png;base64,${currentResult.segmentation_image}`;
        instructions.textContent = 'Fareyi sürükleyerek Anatomik Haritayı (Segmentasyon) inceleyin';
    }
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
    historyGrid.innerHTML = sessionHistory.map((h, i) => {
        let resultText = 'Normal';
        if (h.prediction === 'uveitis') {
            resultText = h.modality.includes('AS-OCT') ? 'Anormal Bulgu' : 'Üveit Şüphesi';
        }

        return `
        <div class="history-card">
            <img class="history-thumb" src="data:image/png;base64,${h.thumbnail}" alt="Analiz ${i + 1}">
            <div class="history-info">
                <div class="history-modality">${h.modality}</div>
                <div class="history-result ${h.prediction}">
                    ${resultText}
                </div>
                <div class="history-prob">%${(h.probability * 100).toFixed(1)} • ${h.timestamp}</div>
            </div>
        </div>
        `;
    }).join('');
}

// Geçmişi sıfırlar ve bölümü gizler.
function clearHistory() {
    sessionHistory = [];
    renderHistory();
}

// === MODEL DETAILS MODAL ===
const modelDetails = {
    'slitlamp': {
        title: 'Slit-lamp Biyomikroskopi (Ön Segment) Detayları',
        icon: '🔬',
        content: `
            <h4>1. Veri Seti ve Etiketleme Metodolojisi</h4>
            <p>Eğitim sürecinde toplam <strong>1.309 adet yüksek çözünürlüklü ön segment fotoğrafı</strong> kullanılmıştır. Görüntüler oftalmologlar tarafından ön kamara, iris ve konjonktiva bölgelerine odaklanacak şekilde klinik standartlarda etiketlenmiştir.</p>
            <h4>2. Mimari ve Eğitim Stratejisi (Architecture)</h4>
            <ul>
                <li><strong>Backbone:</strong> ImageNet ağırlıkları üzerinde ön-eğitimli (pre-trained) <code>EfficientNet-B0</code> mimarisi tercih edilmiştir.</li>
                <li><strong>Veri Artırımı (Data Augmentation):</strong> Aşırı aydınlatma farklarını ve cihaz pozisyon varyasyonlarını tolere edebilmek için rastgele döndürme, renk titreşimi (Color Jitter), ve RandomAffine işlemleri uygulanmıştır.</li>
                <li><strong>Optimizasyon:</strong> AdamW optimizer, Cosine Annealing Learning Rate Scheduler ile desteklenmiştir.</li>
            </ul>
            <h4>3. Klinik Performans ve Yorumlanabilirlik (XAI)</h4>
            <p>Model, ön kamara hücreleri ve konjonktival hiperemiyi saptamada <strong>%93.1 Duyarlılık (Recall)</strong> ve <strong>0.988 AUC (Eğri Altında Kalan Alan)</strong> skoruna ulaşmıştır. Grad-CAM analizleri, modelin doğrudan inflamasyon odaklarına, özellikle korneoskleral limbus çevresine başarıyla odaklandığını kantitatif olarak kanıtlamıştır.</p>
        `
    },
    'octa': {
        title: 'OCT Anjiyografi (OCTA) Detayları',
        icon: '🩻',
        content: `
            <h4>1. Veri Seti Spesifikasyonları ve Zorluklar</h4>
            <p>Çalışma kapsamında <strong>525 adet OCTA (Optical Coherence Tomography Angiography)</strong> taraması kullanılmıştır. OCTA görüntüleri, doğası gereği yüksek düzeyde artefakt (hareket artefaktları, projeksiyon artefaktları) içerdiğinden, modelin yüzeysel ve derin kapiller pleksuslardaki perfüzyon kayıplarını ayırması majör bir zorluk teşkil etmiştir.</p>
            <h4>2. Test Time Augmentation (TTA) Entegrasyonu</h4>
            <ul>
                <li><strong>Müdahale:</strong> Test aşamasında modelin güvenilirliğini artırmak için <code>Test Time Augmentation (TTA)</code> entegre edilmiştir.</li>
                <li><strong>Mekanizma:</strong> Inference sırasında her görüntüye 5 farklı varyasyon (çevirme, ufak rotasyon, kontrast değişimi) uygulanıp, elde edilen tahmin olasılıklarının ortalaması (Ensemble Averaging) alınmıştır.</li>
                <li><strong>Sonuç:</strong> Bu strateji, aşırı gürültülü taramalarda varyansı düşürmüş ve modelin F1 skorunu <strong>%78.0</strong> seviyesine tırmandırmıştır.</li>
            </ul>
            <h4>3. Domain Generalization (Cihaz Bağımsızlığı)</h4>
            <p>Hem <em>Heidelberg Engineering</em> hem de <em>OptoVue</em> cihazlarından alınan farklı FOV (Field of View) taramalarında modelin klinik doğruluğunun ve kalibrasyonunun stabil kaldığı (Brier Skoru) gözlemlenmiştir.</p>
        `
    },
    'cfp': {
        title: 'Renkli Fundus Fotoğrafı (CFP) Detayları',
        icon: '👁️',
        content: `
            <h4>1. Aşırı Veri Dengesizliği (Class Imbalance) Problemi</h4>
            <p>870 görüntünün sadece 63'ü aktif üveitli vakalardan oluşmaktaydı (1:12.8 oran). Bu yapısal dengesizlik (imbalance ratio), standart modellerin loss fonksiyonlarını domine ederek sürekli "Normal" (Majority Class) tahmini yapmasına sebep olan büyük bir problemdi.</p>
            <h4>2. Hiperparametre ve Loss Fonksiyonu Çözümleri</h4>
            <ul>
                <li><strong>Sınıf Ağırlıklandırması (Class Weights):</strong> Pozitif (Üveit) sınıfın ağırlığı, negatif sınıfa kıyasla orantısal olarak artırılarak loss fonksiyonunda (CrossEntropy) üveit vakalarına daha fazla penaltı kesilmesi sağlandı.</li>
                <li><strong>Focal Loss (Gelecek Planı):</strong> Hard-to-classify (zor) örnekler için Focal Loss entegrasyon fizibilitesi analiz edildi.</li>
                <li><strong>Youden Endeksi Optimizasyonu:</strong> Standart 0.50 karar eşiği yerine, ROC eğrisi üzerinde Sensitivite ve Spesifisiteyi maksimize eden <strong>0.68 optimal karar eşiği</strong> hesaplandı.</li>
            </ul>
            <h4>3. Klinik Doğrulama: Sıfır Vaka Kaçağı</h4>
            <p>Uygulanan veri düzeltme teknikleri sayesinde model test setinde <strong>hiçbir üveit vakasını kaçırmamış (%100 Sensitivity/Recall)</strong> ve F1 skoru mükemmel bir seviye olan <strong>%94.7</strong> değerine ulaşmıştır. Model özellikle korioretinal lezyonları çok başarılı yakalamaktadır.</p>
        `
    },
    'bscan_oct': {
        title: 'Retina B-scan OCT Detayları',
        icon: '📡',
        content: `
            <h4>1. Medikal Alan Adaptasyonu (Domain-Specific Transfer Learning)</h4>
            <p>Orijinal veri setinin oldukça kısıtlı (75 resim) olması nedeniyle, sadece genel (ImageNet) ağırlıklar kullanılamazdı. Bu sorunu aşmak için model (ResNet-18), dünyaca ünlü <strong>109.000 görüntülük açık kaynak Kermany OCT Veri Seti</strong> üzerinde ön-eğitime (medikal fine-tuning) tabi tutularak, retinadaki katman yapılarını (RPE, fotoreseptör tabakaları) öğrenmesi sağlandı.</p>
            <h4>2. Sentetik Veri Üretimi (Data Synthesis)</h4>
            <ul>
                <li>Eğitim verisi (Train Set), geometrik (affine) ve tıbbi (gaussian noise, blur) transformasyonlar kullanılarak <strong>1080 sentetik görüntüye</strong> artırıldı (Data Augmentation for Scarcity).</li>
                <li><strong>Kritik Kural:</strong> Sentetik görüntüler kesinlikle Test veya Doğrulama (Validation) setlerine karıştırılmayarak <em>Data Leakage</em> engellendi.</li>
            </ul>
            <h4>3. İstatistiksel Kesinlik: K-Fold Cross Validation</h4>
            <p>Sonuçların şans eseri olmadığını ispatlamak için <strong>5-Fold Cross Validation</strong> uygulandı. Her iterasyonda farklı veri kesitleri kullanılarak model doğrulanmış ve tüm katlamaların (fold) ortalamasında yüksek ve kararlı (low variance) bir istikrar grafiği elde edilmiştir.</p>
        `
    },
    'as_oct': {
        title: 'Ön Segment OCT (AS-OCT) Detayları',
        icon: '🔍',
        content: `
            <h4>1. Yüksek Boyutlu Veri ve Modern Mimari</h4>
            <p>Ön segment yapılarını incelemek için literatürdeki en kapsamlı veri tabanlarından olan yüksek çözünürlüklü <strong>MCOA (Multimodal Corneal ve Ocular Anterior) Veri Setindeki 6.664 devasa görüntü</strong> (Corneal opasite, keratit vs.) kullanılmıştır.</p>
            <h4>2. Ağ Mimarisi: Noisy Student</h4>
            <ul>
                <li>Geleneksel ResNet modelleri yerine <code>timm (PyTorch Image Models)</code> kütüphanesinden <strong>Noisy Student</strong> eğitim stratejisiyle ağırlıklandırılmış gelişmiş <strong>EfficientNet-B0</strong> mimarisi entegre edilmiştir.</li>
                <li>Bu mimari, kornea gibi ince ve şeffaf katmanlardaki milimetrik anomalileri çıkarmada (feature extraction) üstün performans sergiler.</li>
            </ul>
            <h4>3. Klinik Odak ve Grad-CAM Segmentasyonu</h4>
            <p>Model özellikle korneal opasite (bulanıklık), ön kamara derinliği asimetrileri ve iridokorneal açı anomalilerini saptamak üzere çok hassas bir şekilde optimize edilmiştir. Grad-CAM sonuçları, yapay zekanın dikkatinin doğrudan ön kamara açılarına ve stromal defektlere mükemmel derecede odaklandığını göstermektedir.</p>
        `
    }
};

function openModelDetails(modalityId) {
    const data = modelDetails[modalityId];
    if (!data) return;

    document.getElementById('modal-title').innerHTML = `<span style="font-size: 1.5rem; margin-right: 10px;">${data.icon}</span> ${data.title}`;
    document.getElementById('modal-body').innerHTML = data.content;
    document.getElementById('detail-modal').classList.add('show');
}

function closeModelDetails() {
    document.getElementById('detail-modal').classList.remove('show');
}

window.addEventListener('click', (e) => {
    const modal = document.getElementById('detail-modal');
    if (e.target === modal) {
        closeModelDetails();
    }
});
