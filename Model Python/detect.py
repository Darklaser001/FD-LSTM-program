from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import logging
import sys
import numpy as np
import pandas as pd
import glob
import json
from tqdm import tqdm
import subprocess # Do wywoływania programu w find_threshold.py jak nie będzie takowy stworzony.
from fractional_modules import LSTMAutoencoder

# =========================================================================
# --- FUNKCJA RYSUJĄCA (DASHBOARD 4-w-1) ---
# =========================================================================
def generate_dashboard(family, test_filename, vi, threshold, 
                       file_losses, anomaly_indices, 
                       test_tensor, reconstructions, 
                       output_dir, timestamps=None):
    """
    Generuje i zapisuje panel z 4 wykresami w folderze 'plots'.
    """
    # Tworzenie podfolderu na wykresy
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Raport Detekcji: {family} / {test_filename} (Vi={vi})", fontsize=16)

    # 1. Wykres Błędu w Czasie
    ax1 = fig.add_subplot(2, 2, 1)
    
    if timestamps is not None and len(timestamps) == len(file_losses):
        import matplotlib.dates as mdates
        ax1.plot(timestamps, file_losses, label='Błąd MSE', color='steelblue', linewidth=1)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
    else:
        ax1.plot(file_losses, label='Błąd MSE', color='steelblue', linewidth=1)

    ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Próg ({threshold:.4f})')
    
    if len(anomaly_indices) > 0:
        if timestamps is not None and len(timestamps) == len(file_losses):
            # Wyciągamy daty dla indeksów anomalii
            anom_times = timestamps[anomaly_indices] 
            anom_vals = file_losses[anomaly_indices]
            ax1.scatter(anom_times, anom_vals, color='red', s=20, zorder=5, label='Anomalia')
        else:
            ax1.scatter(anomaly_indices, file_losses[anomaly_indices], color='red', s=20, zorder=5, label='Anomalia')

    ax1.set_title("Błąd Rekonstrukcji (MSE) w czasie")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()

    # 2. Histogram Błędu
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(file_losses, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Rozkład błędu')
    ax2.axvline(x=threshold, color='r', linestyle='--', label='Próg')
    ax2.set_title("Histogram Błędu Rekonstrukcji")
    ax2.set_yscale('log') 
    ax2.legend()

    # 3. Przykładowa Rekonstrukcja (Największy Błąd)
    ax3 = fig.add_subplot(2, 2, 3)
    if len(anomaly_indices) > 0:
        # Szukamy indexu z największym błędem wśród anomalii
        # anomaly_indices to np.array indeksów
        losses_at_anomalies = file_losses[anomaly_indices]
        max_loss_arg = np.argmax(losses_at_anomalies)
        max_err_idx = anomaly_indices[max_loss_arg]
        
        label_text = f"Największa Anomalia (idx={max_err_idx})"
        idx_to_plot = max_err_idx
    else:
        idx_to_plot = len(file_losses) // 2
        label_text = f"Przykładowy przebieg (Normalny - idx={idx_to_plot})"

    orig_seq = test_tensor[idx_to_plot].cpu().numpy()
    recon_seq = reconstructions[idx_to_plot] 

    # Rysujemy cechę 0
    ax3.plot(orig_seq[:, 0], label='Oryginał (Cecha 0)', color='black', alpha=0.7)
    ax3.plot(recon_seq[:, 0], label='Rekonstrukcja (Cecha 0)', color='orange', linestyle='--')
    ax3.set_title(label_text)
    ax3.legend()

    # 4. Posortowany Błąd
    ax4 = fig.add_subplot(2, 2, 4)
    sorted_losses = np.sort(file_losses)
    ax4.plot(sorted_losses, color='green', linewidth=2)
    ax4.axhline(y=threshold, color='r', linestyle='--')
    ax4.set_title("Posortowane wartości błędu")
    ax4.set_xlabel("Próbki")
    ax4.set_ylabel("MSE")

    save_path = os.path.join(plots_dir, f"dashboard_{test_filename.replace('.pt', '')}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    
    logging.info(f"Zapisano wykresy w: {plots_dir}")

# =============================================================================
# --- Główna Funkcja Detekcji ---
# =============================================================================
def main(args):
    # --- Konfiguracja Logowania (do pliku outputs/detect_...) ---
    # Tworzy ścieżkę do podfolderu, np. "outputs/RLM1"
    log_dir = os.path.join("outputs", args.family) 
    os.makedirs(log_dir, exist_ok=True) 
    
    # Dynamicznie pobiera nazwę bieżącego skryptu (np. "train_model")
    script_name = os.path.splitext(os.path.basename(__file__))[0] 
    log_file_name = f"{script_name}_{args.family}_output.txt"
    log_file_path = os.path.join(log_dir, log_file_name)
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='w'), # 'w' - nadpisz stary log
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    
    # --- Wypisanie konfiguracji ---
    logging.info("--- Konfiguracja Uruchomienia ---")
    logging.info(f"Rodzina danych: {args.family}")
    logging.info(f"Katalog bazowy danych: {args.base_dir}")
    logging.info(f"Rozmiar partii (Batch Size): {args.batch_size}")
        
    # Specyficzne dla detect.py (jeśli tam wklejasz)
    if 'recalculate_threshold' in args:
        logging.info(f"Wymuś obliczenie progu: {args.recalculate_threshold}")

    logging.info("Parametry modelu zostaną wczytane z checkpointu...")
    logging.info("---------------------------------")

    logging.info(f"--- Rozpoczynam Detekcję Anomalii dla: {args.family} ---")
    logging.info(f"Logi zapisywane do: {log_file_path}")

    # --- Konfiguracja Urządzenia i Ścieżek ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Używam urządzenia: {device}")
    
    checkpoint_dir = os.path.join("checkpoints", args.family)
    model_path = os.path.join(checkpoint_dir, f"model_checkpoint_{args.family}.pth")
    threshold_path = os.path.join(checkpoint_dir, f"threshold_{args.family}.json")
    processed_data_dir = os.path.join(args.base_dir, args.family)

    # --- 1. Wczytanie Modelu ---
    if not os.path.exists(model_path):
        logging.error(f"BŁĄD: Nie znaleziono modelu w {model_path}")
        return
    
    checkpoint = torch.load(model_path)
    
    
    #### WAŻNY PUNKT - wczytujemy dane z modelu - kopiujemy wartości które używaliśmy przy rozruchu modelu. Do manualnego nadpisania przy uruchomieniu programu.

    # 1.1. Wczytaj parametry domyślne z pliku
    if 'model_params' not in checkpoint:
        logging.error("BŁĄD: Checkpoint nie zawiera 'model_params'.")
        logging.error("Proszę, wytrenuj model ponownie (choćby 1 epokę) używając najnowszej wersji 'train_model.py'.")
        return
        
    params_from_file = checkpoint['model_params']

    # 1.2. Zbuduj ostateczną listę parametrów, dając priorytet 'args'
    # Użyj wartości od użytkownika (args.X) jeśli istnieje, inaczej weź z pliku
    final_params = {
        'input_dim': args.input_dim if args.input_dim is not None else params_from_file.get('input_dim'),
        'hidden_dim': args.hidden_dim if args.hidden_dim is not None else params_from_file.get('hidden_dim'),
        'num_layers': args.num_layers if args.num_layers is not None else params_from_file.get('num_layers'),
        'seq_len': args.seq_len if args.seq_len is not None else params_from_file.get('seq_len'),
        'dropout_rate': args.dropout if args.dropout is not None else params_from_file.get('dropout'),
        'vi' : args.vi if args.vi is not None else params_from_file.get('vi')
    }


    # 1.3. Zainicjuj model
    model = LSTMAutoencoder(
        input_dim=final_params['input_dim'],
        hidden_dim=final_params['hidden_dim'],
        num_layers=final_params['num_layers'],
        seq_len=final_params['seq_len'],
        dropout_rate=final_params['dropout_rate'],
        vi=final_params['vi']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info("Model wczytany pomyślnie.")


    # --- 2. Wczytanie Progu (Threshold) ---
    if not os.path.exists(threshold_path):
        logging.warning(f"Brak pliku progu w {threshold_path}.")
        logging.warning("Uruchamiam automatyczne obliczanie progu (proszę czekać)...")

        # Składamy komendę do uruchomienia find_threshold.py
        # Używamy tych samych argumentów 'family', 'base_dir' itp.
        # WAŻNE: Musisz mieć 'python' w swojej ścieżce systemowej (PATH)
        cmd = [
            "python", "find_threshold.py",
            "--family", args.family,
            "--base_dir", args.base_dir,
            "--batch_size", str(args.batch_size),
            "--vi", str(final_params['vi'])
            # Uwaga: parametry modelu (input_dim itp.) są teraz wczytywane
            # przez find_threshold.py z pliku checkpointu (zgodnie z Nr 1)
        ]

        try:
            # Uruchom proces i poczekaj na jego zakończenie
            # Wyniki zobaczysz na żywo w konsoli
            subprocess.run(cmd, check=True) 
            logging.info("Obliczanie progu zakończone. Kontynuuję detekcję.")
        except Exception as e:
            logging.error(f"Automatyczne uruchomienie 'find_threshold.py' nie powiodło się: {e}")
            logging.error("Uruchom skrypt ręcznie, aby zobaczyć błąd. Przerywam.")
            return

        ### TO WSZYSTKO to zabezpieczenie i automatyzacja pracy.
        
    try:
        with open(threshold_path, 'r') as f:
            data = json.load(f)
            threshold = data['threshold']
        logging.info(f"Pomyślnie wczytano próg: {threshold:.6f}")
    except Exception as e:
        logging.error(f"Nie udało się wczytać pliku progu ({e}). Przerywam.")
        return

    # --- 3. Detekcja Anomalii (Pętla po plikach "Wady") ---
    logging.info(f"\n--- Rozpoczynam analizę plików 'Wady' (Próg = {threshold:.6f}) ---")
    
    wady_files_data = sorted(glob.glob(os.path.join(processed_data_dir, "*_data.pt")))
    if not wady_files_data:
        logging.warning(f"Nie znaleziono żadnych plików '*_data.pt' w {processed_data_dir}")
        return

    criterion = nn.MSELoss(reduction='none')
    all_raw_events = [] # Nowa lista na wszystkie zdarzenia
    global_losses = [] # lista do globalnych błędów
    total_files_with_anomalies = 0

    for data_file_path in wady_files_data:
        # Pomiń plik treningowy, jeśli się tam znalazł
        if os.path.basename(data_file_path) == 'train_data.pt':
            continue

        file_base_name = os.path.basename(data_file_path).replace('_data.pt', '')
        logging.info(f">>> Analizuję plik: {file_base_name}.xlsx")

        map_file_path = data_file_path.replace('_data.pt', '_map.csv')
        if not os.path.exists(map_file_path):
            logging.error(f"Brak pliku mapy {map_file_path}! Pomijam.")
            continue
        
        try:
            wady_tensor = torch.load(data_file_path)
        except Exception as e:
            logging.error(f"Błąd wczytywania {file_base_name}: {e}. Pomijam.")
            continue

        wady_loader = DataLoader(TensorDataset(wady_tensor, wady_tensor), batch_size=args.batch_size)
        
        
        file_losses = []
        reconstructions = []
        with torch.no_grad():
            for (batch_data, _) in wady_loader:
                batch_data = batch_data.to(device)
                reconstruction = model(batch_data)
                loss = criterion(reconstruction, batch_data) # Porównaj z oryginałem
                loss = torch.mean(loss, dim=(1, 2))
                file_losses.extend(loss.cpu().numpy())
                reconstructions.append(reconstruction.cpu().numpy())
        
        if not file_losses:
            logging.info("Plik pusty. Pomijam.")
            continue
            
        file_losses = np.array(file_losses)
        global_losses.extend(file_losses)
        reconstructions = np.concatenate(reconstructions, axis = 0)
        anomaly_indices = np.where(file_losses > threshold)[0]
        
        if len(anomaly_indices) == 0:
            logging.info("Nie znaleziono anomalii w tym pliku.")
            continue

        # Grupowanie sąsiadujących anomalii
        anomaly_events = []
        start_index = anomaly_indices[0]
        for i in range(1, len(anomaly_indices)):
            if anomaly_indices[i] != anomaly_indices[i-1] + 1:
                anomaly_events.append((start_index, anomaly_indices[i-1]))
                start_index = anomaly_indices[i]
        anomaly_events.append((start_index, anomaly_indices[-1]))

        logging.info(f"WYKRYTO {len(anomaly_events)} zdarzeń anomalii w tym pliku.")
        total_files_with_anomalies += 1

        # Zapisz zdarzenia do klasyfikacji
        for (start, end) in anomaly_events:
            all_raw_events.append({
                'file_name': file_base_name,
                'start_index': int(start), # int() dla kompatybilności z JSON
                'end_index': int(end),
                'seq_len': final_params['seq_len'],
                # Przekaż ścieżki, aby classify.py wiedział, co otworzyć
                'data_file_path': data_file_path, 
                'map_file_path': map_file_path
            })
        
        timestamps = None
        map_file_path = data_file_path.replace('_data.pt', '_map.csv')
            
        if os.path.exists(map_file_path):
            try:
                df_map = pd.read_csv(map_file_path)
                # LSTM ucina pierwsze (seq_len - 1) próbek przy tworzeniu sekwencji
                current_seq_len = model.seq_len
                
                if len(df_map) >= current_seq_len:
                    # Bierzemy czasy od (seq_len - 1) do końca
                    raw_timestamps = df_map['Timestamp'].values[current_seq_len-1:]
                    
                    # Zabezpieczenie długości (gdyby preprocess coś uciął inaczej)
                    if len(raw_timestamps) > len(file_losses):
                        raw_timestamps = raw_timestamps[:len(file_losses)]
                    
                    timestamps = pd.to_datetime(raw_timestamps)
            except Exception as e:
                logging.warning(f"Nie udało się wczytać mapy czasu: {e}")

        # Generowanie wykresów
        generate_dashboard(
            family=args.family, 
            test_filename=os.path.basename(data_file_path),
            vi=final_params['vi'], 
            threshold=threshold,
            file_losses=file_losses,       
            anomaly_indices=anomaly_indices,
            test_tensor=wady_tensor,
            reconstructions=reconstructions,
            output_dir=log_dir,
            timestamps=timestamps
        )

    logging.info(f"\n--- Detekcja Wstępna Zakończona ---")
    logging.info(f"Łącznie znaleziono {len(all_raw_events)} zdarzeń w {total_files_with_anomalies} plikach.")

    if global_losses:
        logging.info("Generuję histogram dla wszystkich plików łącznie...")
        plt.figure(figsize=(12, 6)) # Nieco szerszy
        
        min_val = max(min(global_losses), 1e-6) # Zabezpieczenie przed 0
        max_val = max(global_losses)
        
        # Tworzymy 100 kubełków rozmieszczonych logarytmicznie
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)
        
        plt.hist(global_losses, bins=bins, color='purple', alpha=0.7, label='Rozkład błędu')
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Próg ({threshold:.2f})')
        
        plt.xscale('log') # Logarytmiczna oś X
        plt.yscale('log') # Logarytmiczna oś Y
        
        plt.title(f"Globalny Histogram Błędu ({args.family}) - Skala Logarytmiczna")
        plt.xlabel("Błąd Rekonstrukcji (MSE) - Skala Log")
        plt.ylabel("Liczba próbek (Log)")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        global_hist_path = os.path.join(log_dir, f"GLOBAL_HISTOGRAM_{args.family}.png")
        plt.savefig(global_hist_path)
        plt.close()
        logging.info(f"Zapisano globalny histogram: {global_hist_path}")

    # --- 4. Zapis zdarzeń i wywołanie classify.py ---
    if not all_raw_events:
        logging.info("Brak zdarzeń do klasyfikacji. Zakończono.")
        return

    # Ścieżka zapisu pliku roboczego, np. outputs/RLM1/raw_events_RLM1.json
    events_file_path = os.path.join(log_dir, f"raw_events_{args.family}.json")
    try:
        with open(events_file_path, 'w') as f:
            json.dump(all_raw_events, f, indent=4)
        logging.info(f"Zapisano surowe zdarzenia w: {events_file_path}")
    except Exception as e:
        logging.error(f"Nie udało się zapisać pliku zdarzeń: {e}")
        return

    # Automatyczne wywołanie classify.py
    logging.info("\n--- Uruchamiam automatyczną klasyfikację i raportowanie... ---")
    cmd = [
        "python", "classify.py",
        "--family", args.family
    ]
    try:
        # Uruchom proces i poczekaj na jego zakończenie
        subprocess.run(cmd, check=True) 
        logging.info("Klasyfikacja zakończona pomyślnie.")
    except Exception as e:
        logging.error(f"Automatyczne uruchomienie 'classify.py' nie powiodło się: {e}")

   
        
# =============================================================================
# --- Argumenty Wiersza Poleceń ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skrypt do detekcji anomalii")
    
    parser.add_argument('--family', type=str, required=True, 
                        help="Rodzina danych (np. RLM1)")
    parser.add_argument('--recalculate-threshold', action='store_true',
                        help="Wymuś ponowne obliczenie progu")
    
    # Argumenty Modelu (muszą pasować do trenowanego)   (Domyślnie wczytywane z checkpointu)
    parser.add_argument('--input_dim', type=int, default=None,
                        help="[Opcjonalne] Nadpisuje 'input_dim' z checkpointu")
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help="[Opcjonalne] Nadpisuje 'hidden_dim' z checkpointu")
    parser.add_argument('--num_layers', type=int, default=None,
                        help="[Opcjonalne] Nadpisuje 'num_layers' z checkpointu")
    parser.add_argument('--seq_len', type=int, default=None,
                        help="[Opcjonalne] Nadpisuje 'seq_len' z checkpointu")
    parser.add_argument('--dropout', type=float, default=None,
                        help="[Opcjonalne] Nadpisuje 'dropout' z checkpointu")
    parser.add_argument('--vi', type=float, default=None, 
                        help="Opcjonalnie: wersja Vi modelu")
    
    # Argumenty Procesu
    parser.add_argument('--base_dir', type=str, default="preprocessed_data")
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()
    main(args)