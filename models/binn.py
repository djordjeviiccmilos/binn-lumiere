import torch
import torch.nn as nn
from torchdiffeq import odeint

class GompertzODE(nn.Module):
    """
    Prosirena Gompertz jednacina sa terapijskim faktorom:
        dV/dt = alpha * V * ln(K / V) - beta * V

    Parametri:
        alpha   -   brzina rasta tumora
        K       -   maksimalna zapremina koju tumor moze dostici
        beta    -   efekat terapije (usporava ili smanjuje tumor)

    Bioloska interpretacija:
        Visok alpha, nizak beta    =>   agresivan rast, terapija ne pomaze
        Nizak alpha, visok beta    =>   tumor se smanjuje, terapija je efikasna
        Kada K > V(t)              =>   tumor tezi da raste ka K
        Kada K < V(t)              =>   tumor tezi da se smanjuje
    """
    def __init__(self, alpha, K, beta):
        super().__init__()
        self.alpha = alpha
        self.K = K
        self.beta = beta

    def forward(self, t, V):
        #Sprecavanje log(0) i negativne vrednosti
        V = torch.clamp(V, min=1e-6)
        K = torch.clamp(self.K, min=1e-6)

        growth = self.alpha * V * torch.log(K / V)
        treatment = self.beta * V
        dVdt = (growth - treatment) / 10.0

        return dVdt


class BINN(nn.Module):
    """
    Bioloski informisana neuronska mreza zaz modelovanje rasta tumora

    Neuronska mreza prima karakteristike pacijenata kao ulaz
    Mreza uci tri bioloska parametra: alpha, K i beta
    ODE solver koristi te parametre da integruje Gompertz jednacinu
    Rezultati su predvidjene zapremine tumora kroz vreme

    Ulazne karakteristike pacijenata:
        1. Normalizovana pocetna zapremina
        2. Log stvarne pocetne zapremine
        3. Normalizovan broj merenja
        4. Normalizovano vremensko pracenje
    """
    def __init__(self, hidden_size=64):
        super().__init__()

        self.parameter_network = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, patient_features, t_points):
        raw_params = self.parameter_network(patient_features)

        #Softplus osigurava da su svi parametri pozitivni
        alpha = nn.functional.softplus(raw_params[:, 0:1])
        K = nn.functional.softplus(raw_params[:, 1:2])
        beta = nn.functional.softplus(raw_params[:, 2:3])

        #Pocetni uslov ODE je uvek 1.0 jer je zapremina normalizovana
        V0 = torch.ones(
            patient_features.shape[0], 1,
            dtype=torch.float32,
        ).to(patient_features.device)

        predictions = []

        for i in range(patient_features.shape[0]):
            ode_func = GompertzODE(alpha[i], K[i], beta[i])

            V_pred = odeint(
                ode_func,
                V0[i],
                t_points,
                method="dopri5",
                rtol=1e-4,
                atol=1e-5,
            )
            predictions.append(V_pred)

        predictions = torch.stack(predictions, dim=1)
        return predictions, alpha, K, beta


class BINNLoss(nn.Module):
    """
    Ukupan loss = data_loss + biology_weight * biology_loss

    data_loss    -   MSE izmedju predvidjenih i stvarnih zapremina
    biology_loss -   Kazna ako parametri izlaze iz bioloski realnih opsega
                     alpha i beta tipicno su izmedju 0 i 1 za tumore
                     K tipicno nije vece od 10 (normalizovano)
    """
    def __init__(self, biology_weight=0.01):
        super().__init__()
        self.biology_weight = biology_weight
        self.mse = nn.MSELoss()

    def forward(self, V_pred, V_true, alpha, K, beta):
        V_pred_log = torch.log1p(torch.clamp(V_pred, min=1e-6))
        V_true_log = torch.log1p(torch.clamp(V_true, min=1e-6))
        data_loss = self.mse(V_pred_log, V_true_log)

        alpha_penalty = torch.mean(torch.relu(alpha - 1.0))
        K_penalty = torch.mean(torch.relu(K - 10.0))
        beta_penalty = torch.mean(torch.relu(beta - 1.0))

        biology_loss = alpha_penalty + K_penalty + beta_penalty
        total_loss = data_loss + self.biology_weight * biology_loss

        return total_loss, data_loss, biology_loss