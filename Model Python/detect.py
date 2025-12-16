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
import seaborn as sns # Dodane do heatmapy
import joblib


# =========================================================================
# --- FUNKCJA RYSUJĄCA (DASHBOARD 4-w-1) ---
# =========================================================================
def generate_dashboard(family, test_filename, vi, threshold, 
                       file_losses, anomaly_indices, 
                       test_tensor, reconstructions, feature_errors,
                       output_dir, timestamps=None, feature_names=None):
    """
    Generuje i zapisuje panel z 4 wykresami w folderze 'plots'.
    """
    # Tworzenie podfolderu na wykresy
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(24, 14))
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

    idx_to_plot = 0
    feature_to_plot_idx = 0

    if len(anomaly_indices) > 0:
        # Szukamy indexu z największym błędem wśród anomalii
        # anomaly_indices to np.array indeksów
        losses_at_anomalies = file_losses[anomaly_indices]
        max_loss_arg = np.argmax(losses_at_anomalies)
        idx_to_plot = anomaly_indices[max_loss_arg]

        errors_at_moment = feature_errors[idx_to_plot] # wektor błędów wszystkich cech w tym momencie
        feature_to_plot_idx = np.argmax(errors_at_moment)
        
        label_text = f"Największa Anomalia (idx={idx_to_plot})"
    else:
        # Jeśli brak anomalii, pokaż środek pliku i pierwszą cechę
        idx_to_plot = len(file_losses) // 2
        feature_to_plot_idx = 0
        label_text = f"Przykładowy przebieg (Normalny - idx={idx_to_plot})"

    # Pobieranie nazwy cechy
    if feature_names is not None and feature_to_plot_idx < len(feature_names):
        feature_name = feature_names[feature_to_plot_idx]
    else:
        feature_name = f"Cecha {feature_to_plot_idx}"

    orig_seq = test_tensor[idx_to_plot]
    recon_seq = reconstructions[idx_to_plot]

    # Rysujemy wybraną, "najgorszą" cechę
    ax3.plot(np.sqrt(np.abs(orig_seq[:, feature_to_plot_idx])), label=f'Oryginał: {feature_name}', color='black', alpha=0.7)
    ax3.plot(np.sqrt(np.abs(recon_seq[:, feature_to_plot_idx])), label=f'Rekonstrukcja', color='orange', linestyle='--')
    ax3.set_title(f"{label_text}\nNajwiększy błąd w: {feature_name}")
    ax3.legend()

    # 4. Heatmanap błędów cech
    ax4 = fig.add_subplot(2, 2, 4)
    
    # 1. Obliczamy średni błąd dla każdej cechy w całym pliku
    # feature_errors_scaled jest [Samples, Features]
    mean_error_per_feature = np.mean(feature_errors, axis=0)
    
    # 2. Wybieramy TOP 20 cech z największym błędem
    TOP_N = 50
    top_indices = np.argsort(mean_error_per_feature)[-TOP_N:] # Ostatnie N to największe
    # Sortujemy je, żeby największy błąd był na górze (odwracamy kolejność)
    top_indices = top_indices[::-1] 
    
    # Wycinamy dane tylko dla tych cech
    heatmap_data = feature_errors[:, top_indices].T # [Top_Features, Samples]
    heatmap_data = np.log1p(heatmap_data) # Logarytm dla czytelności
    
    # Przygotowanie etykiet
    if feature_names:
        yticklabels = [feature_names[i] for i in top_indices]
    else:
        yticklabels = [f"Cecha {i}" for i in top_indices]

    # Rysujemy Heatmapę
    # Używamy rasterized=True dla szybkości zapisu wektorowego/png przy dużej ilości punktów
    sns.heatmap(
        heatmap_data,
        ax=ax4,
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Log(Błąd)'},
        yticklabels=yticklabels,
        xticklabels=False, # Wyłączamy X bo i tak nieczytelne
        rasterized=True,
        vmin=0
    )
    
    # Ręczne dodanie etykiet Osi X (Czasu)
    # Dzielimy oś czasu na 10 równych punktów
    num_ticks = 10 
    total_steps = heatmap_data.shape[1]
    tick_positions = np.linspace(0, total_steps - 1, num_ticks, dtype=int)
    
    if timestamps is not None and len(timestamps) == total_steps:
        # Formatowanie daty do HH:MM:SS
        import pandas as pd
        # Upewniamy się, że timestamps to format datetime
        ts_temp = pd.to_datetime(timestamps)
        xticklabels = [ts_temp[i].strftime('%H:%M:%S') for i in tick_positions]
        xlabel_text = "Czas (Godzina)"
    else:
        xticklabels = tick_positions
        xlabel_text = "Krok czasowy"

    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(xticklabels, rotation=45, ha='right')

    ax4.set_title(f"Heatmapa Błędów - TOP {TOP_N} Najgorszych Procesów", fontsize=10)
    ax4.set_ylabel("Proces / Cecha")
    ax4.set_xlabel(xlabel_text)
    ax4.tick_params(axis='y', labelsize=8)

    # Zapis
    save_path = os.path.join(plots_dir, f"dashboard_{test_filename.replace('.pt', '')}.png")
    
    plt.tight_layout()

        
    plt.savefig(save_path, dpi=100)
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

    # --- Wczytanie nazw cech (procesów np. proces_003) ---
    feature_names = None
    names_json_path = os.path.join(args.base_dir, args.family, f"{args.family}_column_names.json")
    
    if os.path.exists(names_json_path):
        try:
            with open(names_json_path, 'r', encoding='utf-8') as f:
                feature_names = json.load(f)
            logging.info(f"Wczytano nazwy {len(feature_names)} procesów z: {names_json_path}")
        except Exception as e:
            logging.warning(f"Nie udało się wczytać nazw procesów: {e}")
    else:
        logging.warning(f"Nie znaleziono pliku z nazwami: {names_json_path}. Wykresy będą używać indeksów.")


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

    # --- 1.4 Wczytanie PCA do odwrotnej transformacji ---
    pca_path = os.path.join(processed_data_dir, f'pca_{args.family}.gz')
    if os.path.exists(pca_path):
        try:
            logging.info(f"Wczytuję PCA z {pca_path} w celu inwersji (wizualizacja)...")
            pca_model = joblib.load(pca_path)
            
            # Przygotowanie macierzy do obliczeń na GPU (szybkie odwrocenie PCA)
            # sklearn PCA: components_ shape (n_components, n_features)
            # Transformacja odwrotna: X_orig = X_pca @ components_ + mean_
            pca_components_tensor = torch.tensor(pca_model.components_, dtype=torch.float32).to(device)
            pca_mean_tensor = torch.tensor(pca_model.mean_, dtype=torch.float32).to(device)
            
            logging.info(f"PCA załadowane. Wymiary komponentów: {pca_components_tensor.shape}")
        except Exception as e:
            logging.error(f"Nie udało się wczytać PCA: {e}. Heatmapy będą w skali PCA.")
            pca_components_tensor = None
            pca_mean_tensor = None
    else:
        logging.warning("Brak pliku PCA. Heatmapy pozostaną w skali PCA.")
        pca_components_tensor = None
        pca_mean_tensor = None

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
        
        
        file_losses = []          # Błędy w przestrzeni PCA (do progu)
        file_feature_losses = []  # Błędy w przestrzeni ORYGINALNEJ (do heatmapy)
        reconstructions = []      # Rekonstrukcje w przestrzeni ORYGINALNEJ
        original_inputs = []      # Oryginały w przestrzeni ORYGINALNEJ (do wykresu)

        with torch.no_grad():
            for (batch_data, _) in wady_loader:
                batch_data = batch_data.to(device) # PCA input [Batch, Seq, Components]
                reconstruction_pca = model(batch_data) # PCA output

                # A. Obliczamy błąd w przestrzeni PCA (dla kompatybilności z Threshold)
                raw_loss_pca = criterion(reconstruction_pca, batch_data) 

                # Błąd skalarny do histogramu i progu (Batch,) - z PCA
                scalar_loss = torch.mean(raw_loss_pca, dim=(1, 2))
                file_losses.extend(scalar_loss.cpu().numpy())
                
          
                # B. Obliczamy błąd w przestrzeni ORYGINALNEJ (dla wizualizacji i heatmapy)
                if pca_components_tensor is not None:
                    # Odwracamy PCA na GPU: X_orig = X_pca @ Components + Mean
                    # batch_data: [B, S, C] @ [C, F] -> [B, S, F]
                    batch_data_orig = torch.matmul(batch_data, pca_components_tensor) + pca_mean_tensor
                    reconstruction_orig = torch.matmul(reconstruction_pca, pca_components_tensor) + pca_mean_tensor
                    
                    # Błąd per cecha (oryginalna)
                    # (Data - Recon)^2
                    raw_loss_orig = (batch_data_orig - reconstruction_orig) ** 2
                    
                    # Uśredniamy po czasie sekwencji (dim=1) -> [Batch, Features]
                    feat_loss = torch.mean(raw_loss_orig, dim=1)
                    
                    file_feature_losses.extend(feat_loss.cpu().numpy())
                    reconstructions.append(reconstruction_orig.cpu().numpy())
                    original_inputs.append(batch_data_orig.cpu().numpy())
                
                else:
                    # Fallback jeśli nie ma PCA
                    feat_loss = torch.mean(raw_loss_pca, dim=1)
                    file_feature_losses.extend(feat_loss.cpu().numpy())
                    reconstructions.append(reconstruction_pca.cpu().numpy())
                    original_inputs.append(batch_data.cpu().numpy())
        
        if not file_losses:
            logging.info("Plik pusty. Pomijam.")
            continue
            
        file_losses = np.array(file_losses)
        file_feature_losses = np.array(file_feature_losses) # [Samples, Original_Features]
        reconstructions = np.concatenate(reconstructions, axis = 0)
        original_inputs = np.concatenate(original_inputs, axis = 0)

        # Detekcja
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

        # Zapisz zdarzenia do klasyfikacji (z wektorem cech)
        for (start, end) in anomaly_events:
            # Wyciągamy średni wektor błędu dla tego zdarzenia
            # event_features ma wymiar [Czas_trwania, Liczba_Cech_Oryginalnych]
            event_features_matrix = file_feature_losses[start:end+1, :]

            # Uśredniamy po czasie trwania anomalii -> dostajemy wektor [Liczba_Cech_Oryginalnych]
            # To jest "Sygnatura Anomalii" (już przetłumaczona na procesy!)
            avg_feature_vector = np.mean(event_features_matrix, axis=0)

            all_raw_events.append({
                'file_name': file_base_name,
                'start_index': int(start), # int() dla kompatybilności z JSON
                'end_index': int(end),
                'seq_len': final_params['seq_len'],
                # Przekaż ścieżki, aby classify.py wiedział, co otworzyć
                'data_file_path': data_file_path, 
                'map_file_path': map_file_path,
                'feature_vector': avg_feature_vector.tolist() 
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
            vi=final_params['vi'],  # Parametr Vi na którym uczył się model
            threshold=threshold,    # Próg modelu
            file_losses=file_losses,  # Błędy PCA (do wykresu 1 i 2)     
            anomaly_indices=anomaly_indices,
            test_tensor=original_inputs,    # Oryginały (do wykresu 3)
            feature_errors=file_feature_losses, # Błędy ORYGINALNE (do heatmapy)
            reconstructions=reconstructions, # Rekonstrukcje ORYGINALNE (do wykresu 3)
            output_dir=log_dir, # Ścieżka do wykresów
            timestamps=timestamps,
            feature_names=feature_names
        )

    logging.info(f"\n--- Detekcja Wstępna Zakończona ---")
    logging.info(f"Łącznie znaleziono {len(all_raw_events)} zdarzeń w {total_files_with_anomalies} plikach.")

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