import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import logging
import sys
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from fractional_modules import LSTMAutoencoder

# =============================================================================
# --- Funkcje Pomocnicze ---
# =============================================================================

def plot_3d_scatter_analysis(processed_events, log_dir, family_name):
    """Wersja PANCERNA: Ręczne obliczanie logarytmu przed rysowaniem."""
    from mpl_toolkits.mplot3d import Axes3D

    # 1. Przygotowanie danych
    data = []
    for event in processed_events:
        if 'type' not in event: continue
        data.append({
            'Duration': event['features']['duration'],
            'Mean Error': event['features']['mean_error'],
            'Variability': event['features']['variability'],
            'Type': event['type'] # Int
        })
    
    if not data:
        logging.warning("Brak danych do wykresu 3D.")
        return

    df = pd.DataFrame(data)

    # 2. Matematyka (Ręczny Logarytm)
    # Zamieniamy 0 na małą liczbę, żeby logarytm nie wybuchł
    epsilon = 1e-6
    df['Log_Duration'] = np.log10(df['Duration'] + epsilon)
    df['Log_Mean_Error'] = np.log10(df['Mean Error'] + epsilon)
    df['Log_Variability'] = np.log10(df['Variability'] + epsilon)

    # 3. Rysowanie (Na zwykłych osiach liniowych, bo dane są już zlogarytmowane)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    unique_types = sorted(df['Type'].unique())

    for type_id in unique_types:
        subset = df[df['Type'] == type_id]
        color = CLUSTER_COLORS.get(type_id, 'black')
        
        ax.scatter(
            subset['Log_Duration'], 
            subset['Log_Mean_Error'], 
            subset['Log_Variability'],
            c=color,
            label=f'Typ {type_id}',
            s=60, 
            alpha=0.8, 
            edgecolors='w'
        )

    # Opisy osi (informujemy, że to Log10)
    ax.set_xlabel('Log10(Czas Trwania)')
    ax.set_ylabel('Log10(Średni Błąd)')
    ax.set_zlabel('Log10(Zmienność)')
    
    ax.set_title(f"Analiza 3D Anomalii ({family_name})\n(Wartości zlogarytmowane ręcznie)")
    ax.legend(loc='upper left')
    
    # Kąt kamery
    ax.view_init(elev=20, azim=135)

    path = os.path.join(log_dir, f"ANALYSIS_3D_Scatter_{family_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Zapisano wykres 3D (metoda pancerna): {path}")

def plot_scatter_analysis(processed_events, log_dir, family_name):
    """Rysuje mapę anomalii: Czas trwania vs Amplituda."""
    plt.figure(figsize=(12, 8))
    
    data = []
    for event in processed_events:
        if 'type' not in event: continue
        data.append({
            'Duration': event['features']['duration'],
            'Mean Error': event['features']['mean_error'],
            'Type': f"Typ {event['type']}"
        })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        logging.warning("Brak danych do wykresu punktowego.")
        return

    # Tworzymy nowy słownik, gdzie kluczem jest np. "Typ 1", a nie 1
    type_palette = {f"Typ {k}": v for k, v in CLUSTER_COLORS.items()}
    
    # Rysujemy kropki
    sns.scatterplot(
        data=df, x='Duration', y='Mean Error', 
        hue='Type', style='Type', s=100, palette=type_palette
    )
    
    plt.yscale('log') # Logarytmiczna oś Y
    plt.xscale('log') # Logarytmiczna oś X
    plt.title(f"Mapa Anomalii: Czas trwania vs Błąd ({family_name})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    path = os.path.join(log_dir, f"ANALYSIS_Scatter_{family_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Zapisano wykres punktowy: {path}")

def plot_boxplot_analysis(processed_events, log_dir, family_name):
    """Rysuje wykres pudełkowy rozkładu błędów dla każdego typu."""
    plt.figure(figsize=(12, 8))
    
    data = []
    for event in processed_events:
        if 'type' not in event: continue
        data.append({
            'Mean Error': event['features']['mean_error'],
            'Type': f"Typ {event['type']}"
        })
        
    df = pd.DataFrame(data)
    
    if df.empty:
        logging.warning("Brak danych do wykresu pudełkowego.")
        return
    
    # Sortujemy typy
    df = df.sort_values('Type')

    type_palette = {f"Typ {k}": v for k, v in CLUSTER_COLORS.items()}

    sns.boxplot(data=df, x='Type', y='Mean Error', hue='Type', palette=type_palette, legend=False)    
    plt.yscale('log')
    plt.title(f"Statystyka Błędu dla Typów Anomalii ({family_name})")
    plt.grid(True, axis='y', ls="--", alpha=0.5)
    
    path = os.path.join(log_dir, f"ANALYSIS_Boxplot_{family_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Zapisano wykres pudełkowy: {path}")
    
    
def calculate_loss_profile(tensor_data, model, device, criterion, batch_size=512):
    """Oblicza profil błędu dla podanego tensora."""
    loader = DataLoader(TensorDataset(tensor_data, tensor_data), batch_size=batch_size, shuffle=False)
    all_losses = []
    
    with torch.no_grad():
        for batch_data, _ in loader:
            batch_data = batch_data.to(device)
            reconstruction = model(batch_data)
            loss = criterion(reconstruction, batch_data)
            # Średni błąd dla każdej sekwencji
            loss_val = torch.mean(loss, dim=(1, 2)) 
            all_losses.append(loss_val.cpu().numpy())
            
    if not all_losses: return np.array([])
    return np.concatenate(all_losses)

# Kolory dla klastrów
CLUSTER_COLORS = {
    1: '#e6194b', # Red
    2: '#3cb44b', # Green
    3: '#ffe119', # Yellow
    4: '#4363d8', # Blue
    5: '#f58231', # Orange
    6: '#911eb4', # Purple
    7: '#42d4f4', # Cyan
}

# =============================================================================
# --- Główna Funkcja ---
# =============================================================================
def main(args):
    # --- Konfiguracja Logowania ---
    log_dir = os.path.join("outputs", args.family)
    os.makedirs(log_dir, exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_file_name = f"{script_name}_{args.family}_output.txt"
    log_file_path = os.path.join(log_dir, log_file_name)
    
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file_path, mode='w'), logging.StreamHandler(sys.stdout)])
    
    logging.info(f"--- Rozpoczynam Globalną Klasyfikację i Wizualizację dla: {args.family} ---")

    # --- Konfiguracja ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join("checkpoints", args.family)
    model_path = os.path.join(checkpoint_dir, f"model_checkpoint_{args.family}.pth")
    events_file_path = os.path.join(log_dir, f"raw_events_{args.family}.json")

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
    criterion = nn.MSELoss(reduction='none')

    # --- 2. Wczytanie Zdarzeń (Raw Events) ---
    try:
        with open(events_file_path, 'r') as f:
            all_raw_events = json.load(f)
    except FileNotFoundError:
        logging.error("Brak pliku z surowymi zdarzeniami.")
        return

    # --- 3. Przygotowanie Danych do Wykresu (Zszywanie) ---
    logging.info("Przygotowywanie danych do globalnego wykresu...")
    
    # A. Dane Normalne (Referencyjne) - Wczytujemy fragment train_data.pt
    try:
        train_tensor = torch.load(train_data_path)
        # Bierzemy np. 30% długości sumarycznej plików wadliwych lub stałą ilość
        # Tutaj weźmiemy po prostu pierwsze 20% danych treningowych jako reprezentację
        limit = int(len(train_tensor) * 0.2) 
        if limit > 50000: limit = 50000 # Ogranicznik żeby wykres nie był za wielki 
                                        #(ograniczamy ilość danych treningowych wyświetlanych na wykresie)
        train_tensor_sample = train_tensor[:limit]
        
        logging.info("Obliczam błędy dla danych referencyjnych...")
        normal_losses = calculate_loss_profile(train_tensor_sample, model, device, criterion)
    except Exception as e:
        logging.warning(f"Nie udało się wczytać danych referencyjnych: {e}")
        normal_losses = np.array([])

    # B. Dane Wadliwe - Grupujemy zdarzenia po plikach
    events_by_file = {}
    for event in all_raw_events:
        fname = event['data_file_path']
        if fname not in events_by_file: events_by_file[fname] = []
        events_by_file[fname].append(event)

    # Sortujemy pliki alfabetycznie, żeby wykres był spójny
    sorted_files = sorted(events_by_file.keys())
    
    # Listy do przechowywania "zszytych" danych
    global_losses = []   # Ciągła lista błędów (Y axis)
    file_separators = [] # Gdzie kończy się jeden plik a zaczyna drugi (index)
    file_names_map = []  # Nazwa pliku dla separatora
    
    # Dodajemy najpierw dane normalne
    current_idx = 0
    if len(normal_losses) > 0:
        global_losses.extend(normal_losses)
        current_idx += len(normal_losses)
        file_separators.append(current_idx)
        file_names_map.append("DANE NORMALNE (WZORZEC)")

    # Lista cech do klastrowania (zbieramy je teraz, przy okazji iteracji)
    feature_list = []
    valid_events = []

    logging.info(f"Przetwarzanie {len(sorted_files)} plików wadliwych...")
    
    for file_path in tqdm(sorted_files):
        # 1. Obliczamy błędy dla całego pliku
        try:
            tensor_data = torch.load(file_path)
            losses = calculate_loss_profile(tensor_data, model, device, criterion)
        except: continue

        # Przesunięcie indeksów (offset), bo doklejamy ten plik na koniec wykresu
        offset = current_idx 
        
        # 2. Zbieramy cechy anomalii w tym pliku
        file_events = events_by_file[file_path]
        for event in file_events:
            # Wycinamy błędy dla tej konkretnej anomalii
            ev_losses = losses[event['start_index'] : event['end_index'] + 1]
            if len(ev_losses) == 0: continue
            
            features = {
                'duration': len(ev_losses), 
                'amplitude': float(np.max(ev_losses)), 
                'mean_error': float(np.mean(ev_losses)),
                'variability': float(np.std(ev_losses)) 
            }
            
            # Zapisujemy zdarzenie z GLOBALNYM indeksem (do wykresu)
            # Dodajemy offset, żeby wiedzieć gdzie narysować to na dużym wykresie
            event_copy = event.copy()
            event_copy['global_start'] = event['start_index'] + offset
            event_copy['global_end'] = event['end_index'] + offset
            event_copy['features'] = features
            
            feature_list.append([features['duration'], features['amplitude'], features['mean_error'], features['variability']])
            valid_events.append(event_copy)

        # 3. Dodajemy błędy pliku do globalnej listy
        global_losses.extend(losses)
        current_idx += len(losses)
        
        # Zaznaczamy koniec pliku
        file_separators.append(current_idx)
        base_name = os.path.basename(file_path).replace('_data.pt', '')
        file_names_map.append(base_name)

    # --- 4. Klastrowanie ---
    if not feature_list:
        logging.error("Brak anomalii do klastrowania.")
        return

    logging.info("Klastrowanie anomalii...")
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_list)
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Przypisz typy
    for i, event in enumerate(valid_events):
        event['type'] = int(labels[i] + 1)

    # --- 5. Rysowanie Wielkiego Wykresu ---
    logging.info("Generowanie globalnego wykresu...")
    
    # Konwertujemy na numpy dla szybszego rysowania
    global_losses_np = np.array(global_losses)
    
    plt.figure(figsize=(100, 15)) # Bardzo szeroki wykres
    
    plt.yscale('log') # Skalowanie dla lepszego zobrazowania wykresu

    # A. Rysujemy tło (szare) - całość danych
    plt.plot(global_losses_np, color='#d9d9d9', linewidth=0.8, label='Tło (Cały sygnał)')
    
    # B. Rysujemy część "Normalną" (na początku) na zielono
    if len(normal_losses) > 0:
        plt.plot(range(len(normal_losses)), normal_losses, color='#2d8659', linewidth=1.0, label='Dane Wzorcowe')

    # C. Rysujemy Anomalie (Kolorowe)
    
    for cluster_id in range(1, args.num_clusters + 1):
        color = CLUSTER_COLORS.get(cluster_id, 'black')
        # Znajdź wszystkie zdarzenia tego typu
        type_events = [e for e in valid_events if e['type'] == cluster_id]
        
        # Dla każdego zdarzenia rysujemy fragment
        for event in type_events:
            s = event['global_start']
            e = event['global_end']
            plt.plot(range(s, e+1), global_losses_np[s:e+1], color=color, linewidth=1.5)

    # D. Dodajemy separatory plików i etykiety
    # Rysujemy linię przerywaną między plikami
    for i, sep_idx in enumerate(file_separators):
        plt.axvline(x=sep_idx, color='black', linestyle=':', alpha=0.5)
        
        # Dodajemy nazwę pliku (tylko co któryś, żeby nie zamazać, lub obrócony tekst)
        # Obliczamy środek sekcji pliku
        prev_sep = file_separators[i-1] if i > 0 else 0
        center = (prev_sep + sep_idx) / 2
        
        if i < len(file_names_map):
            plt.text(center, max(global_losses_np)*0.95, file_names_map[i], 
                     rotation=90, verticalalignment='top', horizontalalignment='center', fontsize=8, alpha=0.7)

    # Legenda (Ręczna, żeby była ładna)
    patches = [mpatches.Patch(color='#2d8659', label='Dane Wzorcowe')]
    for i in range(1, args.num_clusters + 1):
        patches.append(mpatches.Patch(color=CLUSTER_COLORS[i], label=f'Anomalia Typ {i}'))
    
    plt.legend(handles=patches, loc='upper right')
    plt.title(f"Globalna Analiza Anomalii - Rodzina: {args.family}")
    plt.xlabel("Ciągła Oś Czasu (Sekwencje)")
    plt.ylabel("Błąd Rekonstrukcji")
    plt.margins(x=0) # Usuwa marginesy po bokach
    
    plot_path = os.path.join(log_dir, f"GLOBAL_TIMELINE_{args.family}.png")
    plt.savefig(plot_path, dpi=400, bbox_inches='tight') # Wyższa rozdzielczość z usunięciem zbędnego tła
    plt.close()
    
    logging.info(f"Wykres globalny zapisany: {plot_path}")

    # --- Generowanie Wykresów Analitycznych ---
    logging.info("Generowanie wykresów analitycznych (Scatter & Boxplot)...")
    # Musimy przekazać 'valid_events', które mają już przypisany 'type'
    plot_scatter_analysis(valid_events, log_dir, args.family)
    plot_boxplot_analysis(valid_events, log_dir, args.family)
    plot_3d_scatter_analysis(valid_events, log_dir, args.family)

    # --- 6. Zapis JSON ---
    final_json = {f"Anomalia Typu {i}": [] for i in range(1, args.num_clusters + 1)}
    
    for event in valid_events:
        # Tutaj musimy wczytać mapę, żeby odzyskać oryginalny czas (dla raportu)
        # Używamy 'map_file_path' z eventu
        try:
            fname = os.path.basename(event['data_file_path']).replace('_data.pt', '.xlsx')
            # Do JSONa wrzucamy standardowe dane (nie globalne indeksy, tylko lokalne)
            
            # Proste dodanie do JSON (bez ponownego czytania mapy dla prędkości)
            event_data = {
                'file': fname,
                'local_start_index': event['start_index'],
                'local_end_index': event['end_index'],
                'features': event['features']
            }
            final_json[f"Anomalia Typu {event['type']}"].append(event_data)
        except: pass

    json_save_path = os.path.join(log_dir, f"classified_anomalies_{args.family}.json")
    with open(json_save_path, 'w') as f:
        json.dump(final_json, f, indent=4)

    logging.info("Zakończono.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skrypt do klasyfikacji i raportowania anomalii")
    
    parser.add_argument('--family', type=str, required=True, 
                        help="Rodzina danych (np. RLM1)")
    
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
    parser.add_argument('--base_dir', type=str, default="preprocessed_data")
    parser.add_argument('--num_clusters', type=int, default=5)

    
    args = parser.parse_args()
    
    main(args)