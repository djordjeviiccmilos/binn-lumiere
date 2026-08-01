import torch
import torch.nn as nn
from torchdiffeq import odeint


class GompertzTreatmentODE(nn.Module):
    """
    Gompertz jednacina sa VREMENSKI PROMENLJIVIM efektom terapije:
        dV/dt = alpha * V * ln(K / V) - beta(t) * V

    Za razliku od baseline modela, beta vise nije konstanta vec funkcija
    vremena:
        beta(t) = beta_max * sigmoid(steepness * (t - t_switch))

    Bioloska interpretacija parametara:
        alpha       -   brzina rasta tumora (nepromenjeno)
        K           -   maksimalna zapremina koju tumor moze dostici (nepromenjeno)
        beta_start  -   efekat terapije NA POCETKU perioda pracenja
        beta_end    -   efekat terapije NA KRAJU perioda pracenja
        t_switch    -   (normalizovano) vreme prelaska izmedju ta dva rezima
                        - klinicki: trenutak kada se efekat terapije bitno menja
        steepness   -   koliko naglo dolazi do prelaza (fiksirano, ne uci se)

    Kljucno: beta_start i beta_end su NEZAVISNI i oba pozitivna, pa prelaz moze
    ici u oba smera:
        beta_start < beta_end  ->  terapija pocinje da deluje kasnije
                                    (npr. odgovor na radioterapiju sa zakasnjenjem)
        beta_start > beta_end  ->  rani odgovor na terapiju, zatim slabljenje
                                    efekta / rezistencija / relaps - vrlo cest
                                    klinicki obrazac kod glioblastoma nakon
                                    inicijalne resekcije i terapije

    Motivacija: LUMIERE dataset ne sadrzi tacne datume terapije, pa se t_switch
    uci direktno iz podataka o zapremini - model "otkriva" kada i u kom smeru
    se efekat terapije promenio kod svakog pacijenta. Ovo je i dalje potpuno
    transparentno (za razliku od dodavanja neuronske korekcije u ODE): svaki
    naucen broj ima jasno bioloshko/klinicko znacenje.
    """

    STEEPNESS = 15.0  # fiksirano - ostrina prelaza; veci broj = nagliji "prekidac"

    def __init__(self, alpha, K, beta_start, beta_end, t_switch):
        super().__init__()
        self.alpha = alpha
        self.K = K
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.t_switch = t_switch

    def beta_of_t(self, t):
        w = torch.sigmoid(self.STEEPNESS * (t - self.t_switch))
        return self.beta_start * (1 - w) + self.beta_end * w

    def forward(self, t, V):
        V = torch.clamp(V, min=1e-6)
        K = torch.clamp(self.K, min=1e-6)

        growth = self.alpha * V * torch.log(K / V)
        beta_t = self.beta_of_t(t)
        treatment = beta_t * V
        dVdt = (growth - treatment) / 10.0

        return dVdt


class PatientBINN(nn.Module):
    """
    Stage 2 BINN - direktno fitovanje parametara po pacijentu.

    Kljucna razlika u odnosu na baseline: NEMA hipermreze koja pogadja
    (alpha, K, beta) iz 4 staticka broja o pacijentu. Umesto toga, svaki
    pacijent ima SVOJE slobodne parametre koji se direktno optimizuju
    prema njegovim stvarno izmerenim tackama.

    Zasto: baseline mreza je morala da "pogodi" parametre bez ikakvog
    uvida u stvarnu putanju zapremine, sto je vodilo ka prosecnim/skoro
    linearnim predikcijama za sve pacijente. Direktno fitovanje resava
    ovaj problem jer se svaki parametar uci tako da najbolje objasni
    TACNO te podatke - a ostaje 100% interpretabilno jer nema deljene
    mreze/embeddinga koji "generalizuje" na netransparentan nacin.

    Parametri se cuvaju kao jedan tenzor oblika (n_patients, 5):
        [raw_alpha, raw_K, raw_beta_start, raw_beta_end, raw_t_switch]
    """

    def __init__(self, n_patients):
        super().__init__()
        self.n_patients = n_patients

        # Inicijalizacija: alpha, K blago iznad nule, beta_start/beta_end
        # pocinju JEDNAKI (bez pretpostavke o smeru promene efekta terapije),
        # t_switch na sredini normalizovanog vremena (sigmoid(0) = 0.5)
        init = torch.zeros(n_patients, 5)
        init[:, 0] = 0.5   # raw_alpha
        init[:, 1] = 2.0   # raw_K   -> softplus(2.0) ~ 2.13, blizu V0=1 pa moze da raste
        init[:, 2] = 0.0   # raw_beta_start
        init[:, 3] = 0.0   # raw_beta_end
        init[:, 4] = 0.0   # raw_t_switch -> sigmoid(0) = 0.5

        self.raw_params = nn.Parameter(init)

    def get_params(self, patient_idx):
        raw = self.raw_params[patient_idx]

        alpha      = nn.functional.softplus(raw[0])
        K          = nn.functional.softplus(raw[1])
        beta_start = nn.functional.softplus(raw[2])
        beta_end   = nn.functional.softplus(raw[3])
        t_switch   = torch.sigmoid(raw[4])

        return alpha, K, beta_start, beta_end, t_switch

    def forward(self, patient_idx, t_points):
        alpha, K, beta_start, beta_end, t_switch = self.get_params(patient_idx)

        ode_func = GompertzTreatmentODE(alpha, K, beta_start, beta_end, t_switch)

        V0 = torch.ones(1, dtype=torch.float32, device=t_points.device)

        V_pred = odeint(
            ode_func,
            V0,
            t_points,
            method="dopri5",
            rtol=1e-4,
            atol=1e-5,
        )

        return V_pred.squeeze(-1), alpha, K, beta_start, beta_end, t_switch

    def beta_curve(self, patient_idx, t_points):
        """Vraca beta(t) krivu za dati vremenski niz - korisno za
        vizuelizaciju i klinicku interpretaciju ("da li i kada se efekat
        terapije kod ovog pacijenta pojacao ili oslabio")."""
        alpha, K, beta_start, beta_end, t_switch = self.get_params(patient_idx)
        ode_func = GompertzTreatmentODE(alpha, K, beta_start, beta_end, t_switch)
        with torch.no_grad():
            return ode_func.beta_of_t(t_points)


class BINNLossV2(nn.Module):
    """
    Ukupan loss = data_loss + biology_weight * biology_loss

    Izmene u odnosu na baseline:
      1. data_loss koristi Huber (SmoothL1) umesto MSE u log-prostoru,
         cime se smanjuje uticaj ekstremnih pacijenata (npr. tumor koji
         naraste 70x) na gradijent svih ostalih pacijenata - Option H.
      2. Granice bioloske kazne se ne uzimaju proizvoljno (alpha<=1, K<=10)
         vec se racunaju iz stvarne raspodele podataka (npr. 95-ti
         percentil observed alpha/K sa jednostavnog per-pacijent fita) -
         Option E. Ovo se prosledjuje spolja preko alpha_max/K_max.
    """

    def __init__(self, biology_weight=0.01, alpha_max=25.0, K_max=20.0, beta_max_cap=2.0):
        super().__init__()
        self.biology_weight = biology_weight
        self.alpha_max = alpha_max      # globalna, generozna granica (videti napomenu ispod)
        self.K_max = K_max              # default - obicno se prepisuje po pacijentu (K_max_override)
        self.beta_max_cap = beta_max_cap
        self.huber = nn.SmoothL1Loss(beta=0.1)

    def forward(self, V_pred, V_true, alpha, K, beta_start, beta_end, K_max_override=None):
        """
        K_max_override: gornja granica specificna za pacijenta, izracunata iz
        NJEGOVIH stvarnih izmerenih zapremina (npr. 1.5 * max opazene V).
        Bioloski princip: kapacitet K ne moze biti manji od zapremine koja je
        vec stvarno izmerena kod tog pacijenta - pa granica prati podatke
        umesto proizvoljne globalne vrednosti (npr. baseline K<=10 koji je
        onemogucavao fitovanje pacijenata sa tumorima koji narastu 30-70x).
        """
        K_max = K_max_override if K_max_override is not None else self.K_max

        V_pred_log = torch.log1p(torch.clamp(V_pred, min=1e-6))
        V_true_log = torch.log1p(torch.clamp(V_true, min=1e-6))
        data_loss = self.huber(V_pred_log, V_true_log)

        alpha_penalty = torch.relu(alpha - self.alpha_max)
        K_penalty = torch.relu(K - K_max)
        beta_penalty = torch.relu(beta_start - self.beta_max_cap) + torch.relu(beta_end - self.beta_max_cap)

        biology_loss = alpha_penalty + K_penalty + beta_penalty
        total_loss = data_loss + self.biology_weight * biology_loss

        return total_loss, data_loss, biology_loss