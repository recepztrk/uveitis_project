# ==============================================================================
# bscan_kfold_split_v4.py
#
# B-scan V4 modeli için 5-Fold Cross Validation ayrıştırma scripti.
# Sentetik veriler (is_synthetic == True) SADECE "train" setlerine atanır,
# hiçbir şekilde "test" setlerine sızmasına izin verilmez!
#
# Çıktı: metadata/bscan_oct_split_v4_kfold.csv
# ==============================================================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = PROJECT_ROOT / "metadata/bscan_oct_labels_v4.csv"
OUTPUT_CSV = PROJECT_ROOT / "metadata/bscan_oct_split_v4_kfold.csv"
N_SPLITS = 5
RANDOM_STATE = 42

def main():
    print(f"K-Fold (K={N_SPLITS}) Split dosyası oluşturuluyor...")
    
    df = pd.read_csv(LABELS_CSV)
    
    # Gerçek ve Sentetik verileri ayır
    df_real = df[df["is_synthetic"] == False].copy().reset_index(drop=True)
    df_synth = df[df["is_synthetic"] == True].copy().reset_index(drop=True)
    
    # K-Fold hazırlığı (Sadece GERÇEK veriler üzerinde)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # Her fold için yeni kolonlar oluştur
    for i in range(N_SPLITS):
        df_real[f"fold_{i}"] = "train"  # Varsayılan olarak hepsi train
        df_synth[f"fold_{i}"] = "train" # Sentetiklerin HEPSİ DAİMA train!

    # Sadece gerçek verileri StratifiedKFold ile böl
    for fold, (train_idx, test_idx) in enumerate(skf.split(df_real, df_real["binary_label"])):
        # Test indexlerine gelen gerçek verileri "test" olarak işaretle
        # Train olanlar zaten başta "train" olarak işaretlenmişti
        df_real.loc[test_idx, f"fold_{fold}"] = "test"
        
    # Tekrar birleştir
    final_df = pd.concat([df_real, df_synth], ignore_index=True)
    
    # Kaydet
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n=== K-FOLD ÖZETİ ===")
    print(f"Toplam Veri: {len(final_df)} (Gerçek: {len(df_real)}, Sentetik: {len(df_synth)})\n")
    
    for fold in range(N_SPLITS):
        print(f"--- FOLD {fold} ---")
        fold_train = final_df[final_df[f"fold_{fold}"] == "train"]
        fold_test = final_df[final_df[f"fold_{fold}"] == "test"]
        
        train_synth = fold_train["is_synthetic"].sum()
        test_synth = fold_test["is_synthetic"].sum()
        
        print(f"  Train: Toplam={len(fold_train)} (Gerçek: {len(fold_train)-train_synth}, Sentetik: {train_synth})")
        print(f"  Test:  Toplam={len(fold_test)} (Gerçek: {len(fold_test)-test_synth}, Sentetik: {test_synth})")
        print(f"         > Uveitis (Test): {len(fold_test[fold_test['binary_label'] == 1])}")
        
        if test_synth > 0:
            print("  🚨 HATA: TEST SETİNE SENTETİK VERİ SIZMIŞ!")
            
    print(f"\n✅ Başarıyla kaydedildi: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
