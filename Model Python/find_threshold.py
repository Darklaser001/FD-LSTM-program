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
import json # Do zapisu progu
from tqdm import tqdm
from fractional_modules import LSTMAutoencoder

# =============================================================================
# --- Główna Funkcja Obliczania Progu ---
# =============================================================================
def main(args):
    # --- Konfiguracja Logowania (do pliku outputs/threshold_...) ---
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
            logging.FileHandler(log_file_path, mode='a'), # 'w' - nadpisz stary log
            logging.StreamHandler(sys.stdout)
        ]
    )

    
    # --- Wypisanie konfiguracji ---
    logging.info("--- Konfiguracja Uruchomienia ---")
    logging.info(f"Rodzina danych: {args.family}")
    logging.info(f"Katalog bazowy danych: {args.base_dir}")
    logging.info(f"Rozmiar partii (Batch Size): {args.batch_size}")
    
    # Specyficzne dla find_threshold.py
    if 'quantile' in args:
        logging.info(f"Kwantyl progu: {args.quantile}")

    logging.info("Parametry modelu zostaną wczytane z checkpointu...")
    logging.info("---------------------------------")
    
    logging.info(f"--- Rozpoczynam Obliczanie Progu Anomalii dla: {args.family} ---")
    logging.info(f"Logi zapisywane do: {log_file_path}")

    # --- Konfiguracja Urządzenia i Ścieżek ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Używam urządzenia: {device}")
    
    checkpoint_dir = os.path.join("checkpoints", args.family)
    model_path = os.path.join(checkpoint_dir, f"model_checkpoint_{args.family}.pth")
    threshold_save_path = os.path.join(checkpoint_dir, f"threshold_{args.family}.json") # Tu zapiszemy wynik
    
    processed_data_dir = os.path.join(args.base_dir, args.family)
    train_data_path = os.path.join(processed_data_dir, 'train_data.pt')

    # --- 1. Wczytanie Modelu ---
    if not os.path.exists(model_path):
        logging.error(f"BŁĄD: Nie znaleziono modelu w {model_path}")
        logging.error("Uruchom najpierw 'train_model.py'")
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


    # --- 2. Wczytanie danych TRENINGOWYCH (do kalibracji) ---
    try:
        train_tensor = torch.load(train_data_path)
    except FileNotFoundError:
        logging.error(f"BŁĄD: Nie znaleziono danych {train_data_path}")
        return

    train_loader = DataLoader(
        dataset=TensorDataset(train_tensor, train_tensor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- 3. Obliczanie błędów rekonstrukcji ---
    logging.info("Obliczam błędy rekonstrukcji na danych 'Zgodne' ...")
    criterion = nn.MSELoss(reduction='none') 
    all_losses = []
    
    with torch.no_grad():
        for (batch_data, batch_target) in tqdm(train_loader, desc="Kalibracja", file=sys.stdout):
            batch_data = batch_data.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            
            reconstruction = model(batch_data)
            loss = criterion(reconstruction, batch_target)
            loss = torch.mean(loss, dim=(1, 2)) # Średni błąd dla każdej sekwencji
            all_losses.append(loss.cpu().numpy())

    all_losses = np.concatenate(all_losses)
    
    # --- 4. Ustalanie i Zapis Progu ---
    threshold = np.quantile(all_losses, args.quantile) 
    threshold = float(threshold) # Przerobienie na floata (json nie rozumie specjalnego zapisu od numpy)
    
    logging.info(f"Średni błąd na danych 'Zgodne': {np.mean(all_losses):.6f}")
    logging.info(f"Maksymalny błąd na danych 'Zgodne': {np.max(all_losses):.6f}")
    logging.info(f"Obliczony próg anomalii ({args.quantile*100} percentyl): {threshold:.6f}")

    try:
        with open(threshold_save_path, 'w') as f:
            json.dump({'threshold': threshold, 'quantile': args.quantile}, f)
        logging.info(f"Pomyślnie zapisano próg w pliku: {threshold_save_path}")
    except Exception as e:
        logging.error(f"Nie udało się zapisać pliku progu: {e}")

    logging.info("\n--- Zakończono Obliczanie Progu ---")

# =============================================================================
# --- Argumenty Wiersza Poleceń ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skrypt do ustalania progu anomalii")
    
    parser.add_argument('--family', type=str, required=True, help="Rodzina danych (np. RLM1)")
    
    # Argumenty Modelu (muszą pasować do trenowanego) (Domyślnie wczytywane z checkpointu)
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
    parser.add_argument('--quantile', type=float, default=0.999,
                        help="Percentyl do ustalenia progu (np. 0.999 = 99.9%)")

    args = parser.parse_args()
    main(args)