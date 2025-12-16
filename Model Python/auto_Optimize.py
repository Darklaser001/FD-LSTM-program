import os
import subprocess
import pandas as pd
import sys
import time
from tqdm import tqdm # Upewnij się, że masz to zainstalowane

# === KONFIGURACJA ===
FAMILIES = ["RLM1", "RLM2", "RLR1", "RLV1", "RLV2"]
VI_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2]
EPOCHS = 5 # Testowe 5 epok wystarczy do oceny dynamiki

RESULTS_FILE = "optimization_results.csv"

def parse_loss_from_log(log_path):
    """Szuka najniższego Loss w pliku logu (odporna na błędy kodowania)."""
    best_loss = float('inf')
    if not os.path.exists(log_path):
        return best_loss
    
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            if "Średnia strata (Loss):" in line:
                try:
                    loss_str = line.split(":")[-1].strip()
                    loss = float(loss_str)
                    if loss < best_loss:
                        best_loss = loss
                except: continue
    return best_loss

def main():
    print(f"\n=== START: OPTYMALIZACJA GRID SEARCH ({len(FAMILIES)} rodzin x {len(VI_VALUES)} wariantów) ===\n")

    results = []
    
    # Obliczamy ile łącznie będzie kroków (np. 5 rodzin * 5 vi = 25 treningów)
    total_steps = len(FAMILIES) * len(VI_VALUES)
    
    # Tworzymy jeden główny pasek postępu
    with tqdm(total=total_steps, unit="trening", desc="Inicjalizacja") as pbar:
        
        for family in FAMILIES:
            family_best_loss = float('inf')
            
            for vi in VI_VALUES:
                # Aktualizacja opisu paska (żebyś wiedział co się dzieje)
                pbar.set_description(f"Trenuję: {family} [Vi={vi}]")
                
                # 1. Czyszczenie logu
                log_dir = os.path.join("outputs", family)
                log_file = os.path.join(log_dir, f"train_model_{family}_output.txt")
                if os.path.exists(log_file):
                    os.remove(log_file)
                
                # 2. Uruchomienie train_model.py (tryb cichy)
                cmd = [
                    "python", "train_model.py",
                    "--family", family,
                    "--epochs", str(EPOCHS),
                    "--vi", str(vi)
                ]
                
                start_time = time.time()
                # subprocess.run blokuje skrypt dopóki trening się nie skończy
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                duration = time.time() - start_time
                
                # 3. Odczyt wyniku
                loss = parse_loss_from_log(log_file)
                
                # Zapis wyniku do listy
                results.append({
                    "Family": family,
                    "Vi": vi,
                    "Best_Loss": loss,
                    "Duration_s": round(duration, 1)
                })

                # 4. Kopia modelu
                checkpoint_src = os.path.join("checkpoints", family, f"model_checkpoint_{family}.pth")
                checkpoint_dst = os.path.join("checkpoints", family, f"model_v{vi}_{family}.pth")
                if os.path.exists(checkpoint_src):
                    import shutil
                    shutil.copy(checkpoint_src, checkpoint_dst)

                # Aktualizacja paska postępu (krok do przodu)
                pbar.update(1)
                
                # Dodajemy info o ostatnim wyniku obok paska
                pbar.set_postfix({"Ostatni_Loss": f"{loss:.4f}", "Czas": f"{duration:.0f}s"})

    # --- KONIEC PĘTLI, WYPISANIE WYNIKÓW ---
    
    # Czyścimy ekran konsoli nad paskiem dla czytelności tabeli
    print("\n" + "="*60)
    print("=== PODSUMOWANIE RANKINGU VI ===")
    
    df = pd.DataFrame(results)
    
    # Znajdź najlepsze Vi dla każdej rodziny
    best_configs = df.loc[df.groupby("Family")["Best_Loss"].idxmin()]
    
    print("\nSzczegółowe wyniki (zapisane do CSV):")
    # Sortujemy po rodzinie i Loss
    df_sorted = df.sort_values(by=["Family", "Best_Loss"])
    # Wyświetlamy tylko ładne formatowanie w konsoli
    print(df_sorted.to_string(index=False))
    
    df_sorted.to_csv(RESULTS_FILE, index=False)
    
    print("\n" + "="*60)
    print(" >>> ZWYCIĘZCY (REKOMENDACJA) <<<")
    print("="*60)
    for _, row in best_configs.iterrows():
        print(f" Rodzina {row['Family']:<5} -> Najlepsze Vi: {row['Vi']:<4} (Błąd: {row['Best_Loss']:.4f})")
    print("="*60)

if __name__ == "__main__":
    main()