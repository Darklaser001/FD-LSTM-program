import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# --- Funkcje Pomocnicze (Matematyka) ---
# =============================================================================

def compute_gl_coeffs(alpha, history_size, device='cpu'):
    """
    Oblicza współczynniki a_k dla pochodnej ułamkowej metodą Grünwalda-Letnikowa.
    """
    coeffs = [1.0] 
    for k in range(1, history_size):
        prev = coeffs[-1]
        new_coeff = prev * (1 - (alpha + 1) / k)
        coeffs.append(new_coeff)
    
    return torch.tensor(coeffs, device=device, dtype=torch.float32)

# =============================================================================
# --- Szybka Warstwa Aktywacji Ułamkowej (Zwektoryzowana) ---
# =============================================================================

class FractionalActivation(nn.Module):
    def __init__(self, vi, history_size=10, h=1.0, device='cpu'):
        super(FractionalActivation, self).__init__()
        self.vi = vi
        self.history_size = history_size
        self.h = h
        
        # Obliczamy wagi i od razu przygotowujemy jako tensor o odpowiednim kształcie do broadcastingu
        coeffs = compute_gl_coeffs(vi, history_size, device)
        # Kształt: [1, 1, 1, history_size] - pasuje do (Batch, Seq, Dim, History)
        self.register_buffer('coeffs', coeffs.view(1, 1, 1, -1))
        
        # Przygotowujemy przesunięcia: [0, h, 2h, ..., (N-1)h]
        shifts = torch.arange(history_size, device=device, dtype=torch.float32) * h
        # Kształt: [1, 1, 1, history_size]
        self.register_buffer('shifts', shifts.view(1, 1, 1, -1))

    def forward(self, x):
        # x ma kształt: [Batch, Seq_len, Input_dim]
        
        # 1. Rozszerzamy x o nowy wymiar dla "historii" przesunięć
        # x_expanded: [Batch, Seq, Dim, 1]
        x_expanded = x.unsqueeze(-1)
        
        # 2. Odejmujemy przesunięcia (Broadcasting)
        # x_shifted: [Batch, Seq, Dim, History_size]
        x_shifted = x_expanded - self.shifts
        
        # 3. Aplikujemy funkcję bazową (Softplus: log(1+e^x))
        val = F.softplus(x_shifted)
        
        # 4. Mnożymy przez wagi i sumujemy po wymiarze historii
        # Wynik: [Batch, Seq, Dim]
        weighted_sum = torch.sum(val * self.coeffs, dim=-1)
        
        # 5. Skalowanie
        scale_factor = self.h ** (-self.vi)
        return weighted_sum * scale_factor

# =============================================================================
# --- Model Autoencodera z "Ułamkowym Neuronem" w przestrzeni ukrytej ---
# =============================================================================

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, seq_len, dropout_rate, vi=1.0):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len          # Zmienna potrzebna do unsqueeze w forward
        self.vi = vi 

        # Wykrycie urządzenia (ważne dla buforów w nowej klasie)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        # Aktywacja działa na przestrzeni ukrytej (hidden_dim)
        self.activation = FractionalActivation(
            vi=self.vi, 
            history_size=10, 
            h=1.0,          
            device=device
        )

    def forward(self, x):
        # 1. Encoder
        _, (hidden_state, cell_state) = self.encoder(x)
        
        # 2. Aplikujemy "Ułamkową Dynamikę" na stan ukryty
        # hidden_state ma kształt [num_layers, batch, hidden_dim]
        # Permutacja do [Batch, Layers, Hidden] aby pasowało do FractionalActivation
        hidden_transformed = hidden_state.permute(1, 0, 2) 
        
        # Zastosowanie aktywacji ułamkowej
        hidden_transformed = self.activation(hidden_transformed)
        
        # Wracamy do kształtu [Layers, Batch, Hidden] dla dekodera
        hidden_transformed = hidden_transformed.permute(1, 0, 2).contiguous()
        
        # 3. Decoder Input
        # Używamy self.seq_len
        decoder_input = hidden_transformed[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # 4. Decoder
        # Podajemy zmodyfikowany hidden_state (teraz już ciągły w pamięci) do inicjalizacji dekodera
        decoder_output, _ = self.decoder(decoder_input, (hidden_transformed, cell_state))
        
        # 5. Output Layer (Liniowa - pozwala na ujemne wartości)
        reconstruction = self.output_layer(decoder_output)
        
        return reconstruction