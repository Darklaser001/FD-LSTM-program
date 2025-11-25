import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse # Do obsługi argumentów z wiersza poleceń
from tqdm import tqdm # Korzystam do wizualizacji postępu w wierszu poleceń
import logging # Importujemy moduł logowania
import sys     # Potrzebne do obsługi konsoli

# =============================================================================
# --- Krok 2: Definicja Modelu (LSTM Autoencoder) ---
# (Ta klasa pozostaje bez zmian, ale musi być zdefiniowana 
#  zanim wczytamy model)
# =============================================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, seq_len, dropout_rate):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden_state, cell_state) = self.encoder(x)
        decoder_input = hidden_state[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        decoder_output, _ = self.decoder(decoder_input, (hidden_state, cell_state))
        reconstruction = self.output_layer(decoder_output)
        return reconstruction

# =============================================================================
# --- Funkcja Główna (main) ---
# =============================================================================
def main(args):
    
    # Tworzy ścieżkę do podfolderu, np. "outputs/RLM1"
    log_dir = os.path.join("outputs", args.family) 
    os.makedirs(log_dir, exist_ok=True) 
    
    # Dynamicznie pobiera nazwę bieżącego skryptu (np. "train_model")
    script_name = os.path.splitext(os.path.basename(__file__))[0] 
    log_file_name = f"{script_name}_{args.family}_output.txt"
    log_file_path = os.path.join(log_dir, log_file_name)

    # Ustawiamy loggera. Usuwamy domyślne ustawienia, jeśli istnieją.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Konfigurujemy, aby zapisywał do pliku
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode='a'), # 'a' - append (dopisywanie)
            logging.StreamHandler(sys.stdout) # Wypisywanie na konsolę
        ]
    )
    # --- Koniec sekcji logowania ---

    logging.info("--- Konfiguracja Treningu ---")
    logging.info(f"Rodzina danych: {args.family}")
    logging.info(f"Liczba epok: {args.epochs}")
    logging.info(f"Rozmiar partii (Batch Size): {args.batch_size}")
    logging.info(f"Wymiar wejściowy (Input Dim): {args.input_dim if args.input_dim else 'Wykrywany automatycznie'}")    
    logging.info(f"Wznawiam trening: {args.load_model}")
    logging.info("---------------------------------")

    # --- Krok 1b: Ładowanie Danych ---
    logging.info("--- Krok 1b: Ładowanie Danych ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Używam urządzenia: {device}")
    
    # --- Definiowanie ścieżek ---

    # 1. Ścieżka do ODCZYTU danych (ona się nie zmienia)
    processed_data_dir = os.path.join(args.base_dir, args.family)
    train_data_path = os.path.join(processed_data_dir, 'train_data.pt')
    
    # 2. Ścieżka do ZAPISU modelu
    # Tworzy ścieżkę typu "checkpoints/RLM1"
    checkpoint_dir = os.path.join("checkpoints", args.family) 
    model_save_path = os.path.join(checkpoint_dir, f"model_checkpoint_{args.family}.pth")
    
    # Upewnij się, że folder ZAPISU modelu istnieje
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Automatyczne usuwanie starego progu ---
    threshold_path = os.path.join(checkpoint_dir, f"threshold_{args.family}.json")
    try:
        if os.path.exists(threshold_path):
            os.remove(threshold_path)
            logging.info(f"Pomyślnie usunięto stary plik progu: {threshold_path}")
        else:
            logging.info("Nie znaleziono pliku progu.")
    except Exception as e:
        # To jest błąd, np. brak uprawnień
        logging.warning(f"UWAGA: Nie udało się usunąć pliku progu: {e}")

    
    train_data_path = os.path.join(processed_data_dir, 'train_data.pt')
    try:
        train_tensor = torch.load(train_data_path)
    except FileNotFoundError:
        logging.error(f"BŁĄD: Nie znaleziono pliku {train_data_path}")
        exit
        ()

    # Automatycznie wykryj input_dim z pliku
    file_input_dim = train_tensor.shape[2]

    if args.input_dim is None:
        # Użytkownik nie podał wymiaru, więc bierzemy ten z pliku
        logging.info(f"Automatycznie wykryto 'input_dim' z pliku: {file_input_dim}")
        args.input_dim = file_input_dim
    elif args.input_dim != file_input_dim:
        # Użytkownik podał wymiar, ale jest on błędny
        logging.error(f"KRYTYCZNY BŁĄD: Podany --input_dim ({args.input_dim}) nie zgadza się "
                    f"z wymiarem w pliku ({file_input_dim})!")
        exit()
        
    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    logging.info(f"Utworzono DataLoader z {len(train_loader)} partiami.")

    # --- Krok 2: Inicjalizacja Modelu i Optymalizatora ---
    logging.info("\n--- Krok 2: Inicjalizacja Modelu ---")
    
    model = LSTMAutoencoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        dropout_rate=args.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_loss = float('inf') # Do przerywania programu i podstawa przy tworzeniu nowego modelu

    if args.load_model:
        if os.path.exists(model_save_path):
            try:
                logging.info(f"Wczytuję zapisany checkpoint z: {model_save_path}")
                checkpoint = torch.load(model_save_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                
                # Wczytaj poprzedni najlepszy loss
                if 'best_loss' in checkpoint:
                    best_loss = checkpoint['best_loss'] # wczytaj z zapisanych danych ostatni best_loss
                    logging.info(f"Wczytano poprzedni najlepszy loss: {best_loss:.6f}")
                else:
                    # To się zdarzy, jeśli wczytany jest model zapisany przed poprawką (Archaiczny przypadek. Teraz każdy model ma automatycznie zapisywaną takową wartość)
                    logging.warning("Nie znaleziono 'best_loss' w checkpoincie. Używam 'inf'.")
                    logging.warning("Zapisz model ponownie (pozwól mu na 1 epokę), aby zaktualizować checkpoint.")
                
                logging.info(f"Wznowiono trening. Zaczynam od epoki {start_epoch + 1}")
                logging.info(f"Wczytano poprzedni najlepszy loss: {best_loss:.6f}")
            except Exception as e:
                logging.warning(f"Błąd przy wczytywaniu checkpointu: {e}")
                logging.warning("Rozpoczynam nowy trening od zera.")
        else:
            logging.warning(f"Ostrzeżenie: Flaga --load_model podana, ale plik {model_save_path} nie istnieje.")
            logging.warning("Rozpoczynam nowy trening od zera.")
    else:
        logging.info("Rozpoczynam nowy trening od zera.")

    # =============================================================================
    # --- Krok 3: Pętla Treningowa ---
    # =============================================================================
    logging.info(f"\n--- Krok 3: Rozpoczynam Trening (od epoki {start_epoch + 1}) ---")

    patience_counter = 0 # licznik ile razy model nie był lepszy

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0
        
        # Pasek postępu tqdm - on będzie pisał do konsoli, ale nie do pliku
        progress_bar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{start_epoch + args.epochs}", leave=True, file=sys.stdout)
        
        for (batch_data, batch_target) in progress_bar:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            
            reconstruction = model(batch_data)
            loss = criterion(reconstruction, batch_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        
        logging.info(f"Epoka {epoch+1} zakończona. Średnia strata (Loss): {avg_epoch_loss:.6f}")

        # Zapis checkpointu
        if avg_epoch_loss < best_loss:
            # Nadal jest poprawa modelu. Zapisujemy model i resetujemy licznik
            logging.info(f"Epoka {epoch+1} zakończona. Nowy najlepszy wynik! Strata spadła z {best_loss:.6f} do {avg_epoch_loss:.6f}.")
            best_loss = avg_epoch_loss
            patience_counter = 0 # Resetuj licznik
            
            logging.info(f"Zapisuję checkpoint w: {model_save_path}")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss, # Zapisujemy nowy najlepszy wynik
                    'model_params': vars(args) # Zapisujemy konfigurację uruchomieniową do wykorzystania w detect.py i find_threshold.py. Można manualnie nadpisać ale to prosta automatyzacja.
                }, model_save_path)
                logging.info("Zapisano pomyślnie.")
            except Exception as e:
                logging.error(f"KRYTYCZNY BŁĄD ZAPISU MODELU: {e}")
        
        else:
            # Brak poprawy. Nie zapisujemy modelu, zwiększamy licznik.
            patience_counter += 1
            logging.warning(f"Epoka {epoch+1} zakończona. Brak poprawy. Strata: {avg_epoch_loss:.6f} (Najlepsza: {best_loss:.6f})")
            logging.warning(f"Cierpliwość: {patience_counter}/{args.patience}")

        # Sprawdzenie, czy przerywamy trening
        if patience_counter >= args.patience:
            logging.info(f"Przekroczono limit cierpliwości ({args.patience}). Zatrzymuję trening.")
            break

    logging.info("\n--- Trening Zakończony ---")
# =============================================================================
# --- Definicja Argumentów Wiersza Poleceń ---
# =============================================================================
if __name__ == "__main__":
    # 1. Tworzymy parser
    parser = argparse.ArgumentParser(description="Skrypt do trenowania modelu LSTM Autoencoder")

    # 2. Definiujemy argumenty
    
    # Argumenty Danych
    parser.add_argument('--family', type=str, required=True, 
                        help="Rodzina danych do trenowania (np. RLM1)")
    parser.add_argument('--base_dir', type=str, default="preprocessed_data",
                        help="Folder bazowy, gdzie są przetworzone dane")
    
    # Argumenty Modelu
    parser.add_argument('--input_dim', type=int, default=None,
                        help="Wymiar wejściowy (liczba komponentów PCA, domyślnie wykrywany z pliku train_data.pt)")
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help="Wymiar warstwy ukrytej LSTM")
    parser.add_argument('--num_layers', type=int, default=3,
                        help="Liczba warstw LSTM")
    parser.add_argument('--seq_len', type=int, default=256,
                        help="Długość sekwencji")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Współczynnik dropout")
    
    # Argumenty Treningu
    parser.add_argument('--epochs', type=int, default=20,
                        help="Liczba *dodatkowych* epok do trenowania")
    parser.add_argument('--batch_size', type=int, default=512,
                        help="Rozmiar partii (Batch Size)")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Współczynnik uczenia (Learning Rate)")
    parser.add_argument('--patience', type=int, default=5,
                        help="Liczba epok 'cierpliwości' dla early stopping")

    # Argument Wznawiania
    parser.add_argument('--load_model', action='store_true',
                        help="Włącz, aby wczytać ostatni checkpoint i kontynuować trening")

    # 3. Parsujemy argumenty
    args = parser.parse_args()

    # 4. Uruchamiamy funkcję główną z argumentami
    main(args)