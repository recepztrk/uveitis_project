// === PDF DOWNLOAD (jsPDF programatik — html2canvas yok) ===
function downloadPDF() {
    const btn = document.getElementById('pdf-download-btn');
    btn.disabled = true;
    document.getElementById('pdf-btn-icon').textContent = '⏳';
    document.getElementById('pdf-btn-text').textContent = 'PDF Oluşturuluyor...';

    try {
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF({ orientation: 'p', unit: 'mm', format: 'a4' });
        const W  = 210;
        const M  = 12;
        const CW = W - 2*M;
        let y = 0;

        // ── HEADER ──────────────────────────────────────────────
        pdf.setFillColor(15, 23, 42);
        pdf.rect(0, 0, W, 26, 'F');

        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(15);
        pdf.setTextColor(255, 255, 255);
        pdf.text('Uveit AI Karar Destek Sistemi', M, 12);

        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(6.5);
        pdf.setTextColor(125, 211, 252);
        pdf.text('YAPAY ZEKA DESTEKLI OFTALMOLOJIK ANALIZ RAPORU', M, 19);

        const rId   = document.getElementById('pdf-report-id').textContent   || 'RPT';
        const rDate = document.getElementById('pdf-report-date').textContent  || '';
        pdf.setFillColor(30, 41, 59);
        pdf.roundedRect(W-58, 3, 46, 20, 2, 2, 'F');
        pdf.setFontSize(6); pdf.setTextColor(148, 163, 184);
        pdf.text('RAPOR KİMLİĞİ', W-35, 9, { align:'center' });
        pdf.setFont('helvetica', 'bold'); pdf.setFontSize(7.5); pdf.setTextColor(241, 245, 249);
        pdf.text(rId, W-35, 15, { align:'center' });
        pdf.setFont('helvetica', 'normal'); pdf.setFontSize(6); pdf.setTextColor(100, 116, 139);
        pdf.text(rDate, W-35, 20, { align:'center' });
        y = 26;

        // ── DECISION BANNER ──────────────────────────────────────
        const bannerEl   = document.getElementById('pdf-decision-banner');
        const bannerText = bannerEl.textContent.trim();
        const bannerBg   = bannerEl.style.background || bannerEl.style.backgroundColor || '';
        const isRed = bannerBg.includes('239') || bannerBg.includes('ef4444') || bannerBg.includes('254,226') || bannerBg.includes('254, 226');
        pdf.setFillColor(isRed ? 254 : 220, isRed ? 226 : 252, isRed ? 226 : 231);
        pdf.rect(0, y, W, 13, 'F');
        pdf.setFont('helvetica', 'bold'); pdf.setFontSize(11);
        pdf.setTextColor(isRed ? 185 : 21, isRed ? 28 : 128, isRed ? 28 : 61);
        pdf.text(bannerText, W/2, y+9, { align:'center' });
        y += 13;

        // ── METRİK ŞERIDI ────────────────────────────────────────
        const metricData = [
            { label:'MODALİTE',           val: document.getElementById('pdf-modality').textContent },
            { label:'PATOLOJİ OLASILIGI', val: document.getElementById('pdf-probability').textContent },
            { label:'SİSTEM GÜVENİ',      val: document.getElementById('pdf-confidence').textContent },
            { label:'ROC AUC',            val: document.getElementById('pdf-auc').textContent },
        ];
        const cW4 = W / 4;
        pdf.setFillColor(248, 250, 252);
        pdf.rect(0, y, W, 20, 'F');
        pdf.setDrawColor(226, 232, 240); pdf.setLineWidth(0.3);
        pdf.line(0, y, W, y); pdf.line(0, y+20, W, y+20);
        metricData.forEach((m, i) => {
            const x = i * cW4;
            if (i > 0) pdf.line(x, y, x, y+20);
            pdf.setFontSize(6); pdf.setFont('helvetica', 'normal'); pdf.setTextColor(148, 163, 184);
            pdf.text(m.label, x + cW4/2, y+7, { align:'center' });
            pdf.setFontSize(i===1 ? 12 : 9); pdf.setFont('helvetica', 'bold');
            pdf.setTextColor(i===3 ? 14 : 15, i===3 ? 165 : 23, i===3 ? 233 : 42);
            pdf.text(m.val, x + cW4/2, y+16, { align:'center' });
        });
        y += 22;

        // ── GÖRSEL KANITLAR ──────────────────────────────────────
        pdf.setFillColor(14, 165, 233); pdf.rect(M, y, 2, 5, 'F');
        pdf.setFontSize(7); pdf.setFont('helvetica', 'bold'); pdf.setTextColor(71, 85, 105);
        pdf.text('GÖRSEL KANITLAR', M+5, y+4);
        y += 8;

        const imgW = (CW-5)/2;
        const imgH = 50;
        pdf.setDrawColor(226, 232, 240); pdf.setFillColor(248, 250, 252); pdf.setLineWidth(0.3);
        pdf.roundedRect(M,        y, imgW, imgH+2, 2, 2, 'FD');
        pdf.roundedRect(M+imgW+5, y, imgW, imgH+2, 2, 2, 'FD');

        const origSrc = document.getElementById('pdf-img-original').src;
        const heatSrc = document.getElementById('pdf-img-heatmap').src;
        if (origSrc && origSrc.startsWith('data:')) {
            try { pdf.addImage(origSrc, 'PNG', M+1,        y+1, imgW-2, imgH); } catch(e){ console.warn(e); }
        }
        if (heatSrc && heatSrc.startsWith('data:')) {
            try { pdf.addImage(heatSrc, 'PNG', M+imgW+6,   y+1, imgW-2, imgH); } catch(e){ console.warn(e); }
        }

        y += imgH + 4;
        pdf.setFontSize(7); pdf.setFont('helvetica', 'normal'); pdf.setTextColor(100, 116, 139);
        pdf.text('Orijinal Tıbbi Görüntü',  M + imgW/2,       y, { align:'center' });
        pdf.text(document.getElementById('pdf-heatmap-label').textContent, M+imgW+5+imgW/2, y, { align:'center' });
        y += 7;

        // ── AI KLİNİK DEĞERLENDİRME ─────────────────────────────
        pdf.setFillColor(168, 85, 247); pdf.rect(M, y, 2, 5, 'F');
        pdf.setFontSize(7); pdf.setFont('helvetica', 'bold'); pdf.setTextColor(71, 85, 105);
        pdf.text('YAPAY ZEKA KLİNİK DEĞERLENDİRMESİ', M+5, y+4);
        y += 8;

        const aiText  = document.getElementById('pdf-ai-comment').textContent || '—';
        const aiLines = pdf.setFontSize(8.5).setFont('helvetica','italic').splitTextToSize(aiText, CW-8);
        const aiCount = Math.min(aiLines.length, 12);
        const aiBoxH  = aiCount * 4.5 + 6;
        pdf.setFillColor(250, 245, 255); pdf.setDrawColor(168, 85, 247); pdf.setLineWidth(0.8);
        pdf.rect(M, y, CW, aiBoxH, 'F');
        pdf.line(M, y, M, y+aiBoxH);
        pdf.setTextColor(30, 27, 75);
        pdf.text(aiLines.slice(0, aiCount), M+5, y+5.5);
        y += aiBoxH + 3;

        pdf.setLineWidth(0.3);
        pdf.setFontSize(6); pdf.setFont('helvetica','normal'); pdf.setTextColor(148,163,184);
        pdf.text('Gemini 2.5 Flash · Google DeepMind tarafından üretilmiştir', W-M, y, { align:'right' });
        y += 5;

        // ── TEKNİK METRİKLER ─────────────────────────────────────
        pdf.setFillColor(16, 185, 129); pdf.rect(M, y, 2, 5, 'F');
        pdf.setFontSize(7); pdf.setFont('helvetica','bold'); pdf.setTextColor(71,85,105);
        pdf.text('MODEL TEKNİK PERFORMANS METRİKLERİ', M+5, y+4);
        y += 8;

        const backbone = document.getElementById('pdf-backbone').textContent;
        const f1score  = document.getElementById('pdf-f1').textContent;
        const training = document.getElementById('pdf-training').textContent;
        const auc2     = document.getElementById('pdf-auc2').textContent;
        const clinNote = document.getElementById('pdf-clinical-note').textContent;

        const cL=28, cV=58, cL2=28;
        const cV2 = CW - cL - cV - cL2;

        const drawRow = (label1, val1, label2, val2) => {
            pdf.setDrawColor(226,232,240); pdf.setLineWidth(0.3);
            // label1
            pdf.setFillColor(241,245,249); pdf.rect(M,         y, cL,  8, 'FD');
            pdf.setFont('helvetica','bold'); pdf.setFontSize(7.5); pdf.setTextColor(100,116,139);
            pdf.text(label1, M+2, y+5.5);
            // val1
            pdf.setFillColor(255,255,255); pdf.rect(M+cL,      y, cV,  8, 'FD');
            pdf.setFont('helvetica','bold'); pdf.setTextColor(15,23,42);
            pdf.text(val1,  M+cL+2, y+5.5);
            // label2
            const x2 = M+cL+cV;
            pdf.setFillColor(241,245,249); pdf.rect(x2,        y, cL2, 8, 'FD');
            pdf.setFont('helvetica','bold'); pdf.setTextColor(100,116,139);
            pdf.text(label2, x2+2, y+5.5);
            // val2
            pdf.setFillColor(255,255,255); pdf.rect(x2+cL2,    y, cV2, 8, 'FD');
            pdf.setFont('helvetica','bold'); pdf.setTextColor(14,165,233);
            pdf.text(val2, x2+cL2+2, y+5.5);
            y += 8;
        };

        drawRow('Mimari',       backbone, 'F1 Score',  f1score);
        drawRow('Eğitim Verisi',training, 'AUC Skoru', auc2);

        pdf.setFillColor(248,250,252); pdf.setDrawColor(226,232,240);
        pdf.rect(M, y, CW, 8, 'FD');
        pdf.setFont('helvetica','bold'); pdf.setFontSize(7.5); pdf.setTextColor(15,23,42);
        const lw = pdf.getTextWidth('YZ Anatomik Odak: ');
        pdf.text('YZ Anatomik Odak: ', M+3, y+5.5);
        pdf.setFont('helvetica','normal'); pdf.setTextColor(71,85,105);
        pdf.text(clinNote, M+3+lw, y+5.5);
        y += 10;

        // ── FOOTER ───────────────────────────────────────────────
        const pH = pdf.internal.pageSize.getHeight();
        pdf.setFillColor(241,245,249); pdf.rect(0, pH-18, W, 18, 'F');
        pdf.setDrawColor(226,232,240); pdf.setLineWidth(0.5);
        pdf.line(0, pH-18, W, pH-18);
        pdf.setFontSize(7); pdf.setFont('helvetica','bold'); pdf.setTextColor(100,116,139);
        pdf.text('Sorumluluk Reddi:', M, pH-12);
        pdf.setFont('helvetica','normal'); pdf.setTextColor(148,163,184);
        const disc = pdf.splitTextToSize(
            'Bu rapor Uveitis AI Decision Support System tarafından otomatik üretilmiştir. ' +
            'Yalnızca klinisyen karar desteği amacıyla sunulmaktadır — kesin hekim tanısı yerine geçmez.',
            CW-52);
        pdf.text(disc, M, pH-8);
        const fDate = document.getElementById('pdf-footer-date').textContent;
        pdf.setFont('helvetica','bold'); pdf.setFontSize(7.5); pdf.setTextColor(100,116,139);
        pdf.text(fDate, W-M, pH-8, { align:'right' });

        // ── KAYDET ───────────────────────────────────────────────
        pdf.save(`uveit_ai_rapor_${rId}.pdf`);

        btn.disabled = false;
        document.getElementById('pdf-btn-icon').textContent = '✅';
        document.getElementById('pdf-btn-text').textContent = 'PDF İndirildi!';
        setTimeout(() => {
            document.getElementById('pdf-btn-icon').textContent = '📄';
            document.getElementById('pdf-btn-text').textContent = 'Klinik Rapor PDF İndir';
        }, 3000);

    } catch(err) {
        console.error('PDF hatası:', err);
        btn.disabled = false;
        document.getElementById('pdf-btn-icon').textContent = '❌';
        document.getElementById('pdf-btn-text').textContent = 'Hata — Tekrar dene';
        setTimeout(() => {
            document.getElementById('pdf-btn-icon').textContent = '📄';
            document.getElementById('pdf-btn-text').textContent = 'Klinik Rapor PDF İndir';
        }, 3000);
    }
}
