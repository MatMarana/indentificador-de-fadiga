"""
=============================================================================
 DETECÇÃO DE FADIGA MUSCULAR VIA EMG COM PINN
=============================================================================
 Autores  : Mateus Marana Assuena
 Descrição: Extrai features do sinal EMG (RMS, MAV, ZCR, FFT, Espectrograma)
             e treina uma Physics-Informed Neural Network (PINN) para
             identificar o instante de fadiga muscular.
=============================================================================
"""

import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.signal import spectrogram, windows
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.ndimage import gaussian_filter1d

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ARQUIVOS_TREINO = [
    "ColetaMateus2Rep.csv",
    "MateusDia2Rep2.csv",
    "ColetaMateus3Rep.csv",
    "MateusDia2Rep1.csv",
    "ColetaMateusDia3Rep3.csv",
]

ARQUIVOS_TESTE = [
    "ColetaMateus1Rep.csv",
]

ARQUIVOS_VALIDACAO = [
    "ColetaMateusDia3Rep1.csv",
    "ColetaMateusDia3Rep2.csv",
    "MateusDia2Rep3.csv",
]

EMG_NOMES = [
    "R BRACHIORADIALIS: EMG 1 [Volts]",
    "R EXTENSOR CARPI ULNARIS: EMG 2 [Volts]",
    "R PALMARIS LONGUS: EMG 3 [Volts]",
    "R EXTENSOR DIGITORUM: EMG 4 [Volts]",
]

JANELA_SEGUNDOS = 5
EPOCHS          = 2000
LR              = 1e-3
LAMBDA_MONO     = 0.01
LAMBDA_REG      = 0.01

plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#C9D1D9",
    "xtick.color":      "#C9D1D9",
    "ytick.color":      "#C9D1D9",
    "text.color":       "#C9D1D9",
    "grid.color":       "#21262D",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.framealpha": 0.3,
    "font.family":      "monospace",
})

CORES_MUSCULO = ["#58A6FF", "#3FB950", "#F78166", "#D2A8FF"]

def calcular_rms(janela: np.ndarray) -> float:
    return np.sqrt(np.mean(janela ** 2))

def calcular_mav(janela: np.ndarray) -> float:
    return np.mean(np.abs(janela))

def calcular_zero_crossing_rate(janela: np.ndarray) -> float:
    sinais = np.sign(janela)
    sinais[sinais == 0] = 1
    zcr = np.sum(sinais[:-1] != sinais[1:])
    return zcr / len(janela)

def calcular_features_fft(janela: np.ndarray, fs: float) -> tuple:
    # Features espectrais via rfft com janela Hamming e suavização gaussiana.

    N = len(janela)
    janela_win = janela * np.hamming(N)
    S     = np.fft.rfft(janela_win)
    freqs = np.fft.rfftfreq(N, d=1 / fs)

    magnitude = np.abs(S)
    magnitude_suave = gaussian_filter1d(magnitude, sigma=2)

    mask      = (freqs >= 20) & (freqs <= 250)
    freqs     = freqs[mask]
    magnitude = magnitude_suave[mask]

    mag_total = np.sum(magnitude) + 1e-8

    freq_media = np.sum(freqs * magnitude) / mag_total

    cum_energia  = np.cumsum(magnitude)
    freq_mediana = freqs[np.searchsorted(cum_energia, cum_energia[-1] / 2)]

    desvio = np.std(magnitude)

    return freq_media, freq_mediana, desvio

def calcular_features_janela(emg: np.ndarray, fs: float, segundos: float) -> dict:
    amostras_por_janela = int(segundos * fs)
    n_janelas = len(emg) // amostras_por_janela

    resultados = {k: [] for k in [
        "tempo_medio", "rms", "mav", "zcr",
        "fft_media", "fft_mediana", "fft_desvio",
        "spec_energia", "spec_freq_dom",
    ]}

    nwin     = min(amostras_por_janela, 256)
    win_ham  = windows.hamming(nwin)
    noverlap = nwin // 2
    tempo_base = np.arange(len(emg)) / fs

    for i in range(n_janelas):
        inicio = i * amostras_por_janela
        fim    = inicio + amostras_por_janela
        janela = emg[inicio:fim]

        resultados["rms"].append(calcular_rms(janela))
        resultados["mav"].append(calcular_mav(janela))
        resultados["zcr"].append(calcular_zero_crossing_rate(janela))
        resultados["tempo_medio"].append(np.mean(tempo_base[inicio:fim]))

        fm, fmed, fstd = calcular_features_fft(janela, fs)
        resultados["fft_media"].append(fm)
        resultados["fft_mediana"].append(fmed)
        resultados["fft_desvio"].append(fstd)

        f, _, Sxx = spectrogram(
            janela, fs=fs,
            window=win_ham,
            nperseg=nwin,
            noverlap=noverlap,
            scaling="density",
        )
        Sxx      = np.abs(Sxx)
        energia  = np.mean(Sxx)
        freq_dom = f[np.argmax(np.mean(Sxx, axis=1))]
        resultados["spec_energia"].append(energia)
        resultados["spec_freq_dom"].append(freq_dom)

    return {k: np.array(v) for k, v in resultados.items()}


def extrair_features_df(df: pd.DataFrame, tempo: np.ndarray) -> tuple:
    fs = 1.0 / np.mean(np.diff(tempo))
    features_canais = []
    tempo_m = None

    for nome in EMG_NOMES:
        emg = df[nome].values
        res = calcular_features_janela(emg, fs, JANELA_SEGUNDOS)

        if tempo_m is None:
            tempo_m = res["tempo_medio"]

        feats = np.column_stack([
            res["rms"], res["mav"], res["zcr"],
            res["fft_media"], res["fft_mediana"], res["fft_desvio"],
            res["spec_energia"], res["spec_freq_dom"],
        ])
        features_canais.append(feats)

    return np.hstack(features_canais), tempo_m

def calcular_indice_fadiga_fisiologico(
    features_por_canal: list,
    pesos: dict | None = None,
    janela_suavizacao: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score de fadiga baseado em features fisiológicas reais do EMG.
    Aumentam com fadiga  → usadas diretamente : RMS, MAV, spec_energia
    Diminuem com fadiga  → invertidas (1-x)   : ZCR, MNF, MDF, spec_freq_dom
    """
    if pesos is None:
        pesos = {
            "rms":           1.0,
            "mav":           1.0,
            "zcr":           1.2,
            "fft_media":     1.0,
            "fft_mediana":   1.5,
            "spec_energia":  0.8,
            "spec_freq_dom": 1.2,
        }

    peso_total = sum(pesos.values())

    def _norm(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min()
        return (x - x.min()) / (rng + 1e-8)

    scores_canais = []
    for res in features_por_canal:
        score = (
            pesos["rms"]           * _norm(res["rms"])                    +
            pesos["mav"]           * _norm(res["mav"])                    +
            pesos["zcr"]           * (1.0 - _norm(res["zcr"]))            +
            pesos["fft_media"]     * (1.0 - _norm(res["fft_media"]))      +
            pesos["fft_mediana"]   * (1.0 - _norm(res["fft_mediana"]))    +
            pesos["spec_energia"]  * _norm(res["spec_energia"])            +
            pesos["spec_freq_dom"] * (1.0 - _norm(res["spec_freq_dom"]))
        ) / peso_total
        scores_canais.append(score)

    fadiga_score = _norm(np.mean(scores_canais, axis=0))
    fadiga_suave = (
        pd.Series(fadiga_score)
        .rolling(janela_suavizacao, center=True, min_periods=1)
        .mean()
        .values
    )
    return fadiga_score, fadiga_suave

def detectar_ponto_fadiga(
    fadiga_suave: np.ndarray,
    tempo: np.ndarray,
    metodo: str = "hibrido",
    percentil_limiar: float = 75.0,
) -> tuple[int, float]:
    """
    Detecta o instante de fadiga no score suavizado.
    """
    if metodo == "hibrido":
        grad     = np.gradient(fadiga_suave)
        fadiga_n = (fadiga_suave - fadiga_suave.min()) / (fadiga_suave.max() - fadiga_suave.min() + 1e-8)
        grad_n   = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        score = 0.85 * fadiga_n + 0.15 * grad_n
        idx   = int(np.argmax(score))

    elif metodo == "limiar":
        limiar = np.percentile(fadiga_suave, percentil_limiar)
        acima  = np.where(fadiga_suave >= limiar)[0]
        idx    = int(acima[0]) if len(acima) > 0 else int(np.argmax(fadiga_suave))

    else:
        raise ValueError(f"Método desconhecido: {metodo!r}")

    return idx, float(tempo[idx])

class RedeNeuralFadiga(nn.Module):
    """
    Rede neural densa com ativações Tanh.
    Tanh é preferida em PINNs por ter derivadas analíticas suaves.
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(n_features, 128), nn.Tanh(),
            nn.Linear(128, 128),         nn.Tanh(),
            nn.Linear(128, 64),          nn.Tanh(),
            nn.Linear(64, 1),           
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x)

def pinn_loss(
    modelo: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lambda_mono: float = LAMBDA_MONO,
    lambda_reg:  float = LAMBDA_REG,
) -> tuple:
    """
    Loss da PINN com restrições fisiológicas.
    L_data — MSE contra o score fisiológico.
    L_mono — penaliza df/dt < 0 (fadiga deve ser monotonicamente crescente
             no tempo). O gradiente é calculado APENAS em relação à última
             coluna de x, que é o tempo normalizado. Usar a norma de todas
             as features era incorreto: a norma é sempre >= 0, então
             relu(-norma) nunca ativa e L_mono era sempre zero.
    L_reg  — penaliza d²f/dt² (suavidade de Tikhonov): evita oscilações
             bruscas fisiologicamente implausíveis na curva de fadiga.
    """
    x      = x.clone().requires_grad_(True)
    y_pred = modelo(x)

    grads_1 = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    df_dt = grads_1[:, -1].reshape(-1, 1)

    grads_2 = torch.autograd.grad(
        outputs=df_dt,
        inputs=x,
        grad_outputs=torch.ones_like(df_dt),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2f_dt2 = grads_2[:, -1].reshape(-1, 1)

    L_data  = nn.MSELoss()(y_pred, y)
    L_mono  = torch.mean(torch.relu(-df_dt))
    L_reg   = torch.mean(d2f_dt2 ** 2)
    L_total = L_data + lambda_mono * L_mono + lambda_reg * L_reg

    return L_total, L_data.item(), L_mono.item(), L_reg.item()

def treinar_modelo(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    epochs: int = EPOCHS,
    lr:     float = LR,
) -> tuple:
    modelo     = RedeNeuralFadiga(n_features=X_tensor.shape[1])
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, patience=100, factor=0.5,
    )
    historico = {"total": [], "data": [], "mono": [], "reg": []}

    print("─" * 60)
    print(f"{'Época':>8} │ {'L_total':>10} │ {'L_data':>10} │ {'L_mono':>10} │ {'L_reg':>10}")
    print("─" * 60)

    for epoca in range(epochs):
        otimizador.zero_grad()
        L_total, L_data, L_mono, L_reg = pinn_loss(modelo, X_tensor, y_tensor)
        L_total.backward()
        otimizador.step()
        scheduler.step(L_total)

        historico["total"].append(L_total.item())
        historico["data"].append(L_data)
        historico["mono"].append(L_mono)
        historico["reg"].append(L_reg)

        if epoca % 200 == 0:
            print(f"{epoca:>8} │ {L_total.item():>10.5f} │ {L_data:>10.5f} │ {L_mono:>10.5f} │ {L_reg:>10.5f}")

    print("─" * 60)
    return modelo, historico


def carregar_e_extrair(arquivos: list, pasta: str = "dados_coleta_mateus") -> tuple:
    """Lê múltiplos CSVs e empilha X (features) e t (tempo médio)."""
    X_list, t_list = [], []
    for arq in arquivos:
        df = pd.read_csv(f"{pasta}/{arq}")
        X, t = extrair_features_df(df, df["X [s]"].values)
        X_list.append(X)
        t_list.append(t)
    return np.vstack(X_list), np.concatenate(t_list)


def gerar_score_para_arquivos(
    arquivos: list,
    pasta: str = "dados_coleta_mateus",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o score fisiológico de fadiga para todos os arquivos listados
    e concatena os resultados.
    """
    scores, tempos = [], []
    for arq in arquivos:
        df    = pd.read_csv(f"{pasta}/{arq}")
        tempo = df["X [s]"].values
        fs    = 1.0 / np.mean(np.diff(tempo))
        feats = [
            calcular_features_janela(df[nome].values, fs, JANELA_SEGUNDOS)
            for nome in EMG_NOMES
        ]
        score, _ = calcular_indice_fadiga_fisiologico(feats)
        scores.append(score)
        tempos.append(feats[0]["tempo_medio"])
    return np.concatenate(scores), np.concatenate(tempos)


def normalizar_tempo(t: np.ndarray) -> np.ndarray:
    return (t - t.min()) / (t.max() - t.min())

def avaliar_conjunto(
    modelo: nn.Module,
    X_tensor: torch.Tensor,
    y_true: np.ndarray,
    nome: str,
) -> np.ndarray:

    with torch.no_grad():
        pred = modelo(X_tensor).numpy().flatten()

    mse = mean_squared_error(y_true, pred)
    mae = mean_absolute_error(y_true, pred)
    r2  = r2_score(y_true, pred)

    print(f"\n{'═' * 50}")
    print(f"  {nome}")
    print(f"{'═' * 50}")
    print(f"  MSE : {mse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  R²  : {r2:.4f}")

    return pred


def comparar_pinn_vs_fisiologico(
    fadiga_suave: np.ndarray,
    pred: np.ndarray,
    tempo: np.ndarray,
    nome_arquivo: str,
) -> None:

    erro_curva = np.mean((fadiga_suave - pred) ** 2)

    idx_fisio, t_fisio = detectar_ponto_fadiga(fadiga_suave, tempo, metodo="hibrido")
    idx_pinn,  t_pinn  = detectar_ponto_fadiga(pred,         tempo, metodo="hibrido")
    erro_tempo = abs(t_fisio - t_pinn)

    print(f"\n  ── Comparação PINN vs Fisiológico [{nome_arquivo}] ──")
    print(f"  MSE entre curvas      : {erro_curva:.6f}")
    print(f"  Fadiga fisiológica em : {t_fisio:.2f} s  (janela #{idx_fisio})")
    print(f"  Fadiga PINN em        : {t_pinn:.2f} s  (janela #{idx_pinn})")
    print(f"  Erro temporal         : {erro_tempo:.2f} s")

def plotar_metricas_simples(
    t: np.ndarray,
    features_por_canal: list,
    tempo_fadiga: float,
    nomes_canais: list,
    nome_arquivo: str,
) -> None:

    metricas_labels = [
        ("rms",         "RMS (V)"),
        ("mav",         "MAV (V)"),
        ("zcr",         "ZCR"),
        ("fft_media",   "Freq. Média (Hz)"),
        ("fft_mediana", "Freq. Mediana (Hz)"),
        ("fft_desvio",  "Desvio Espectral"),
    ]
    n_met    = len(metricas_labels)
    n_canais = len(features_por_canal)

    fig, axes = plt.subplots(
        n_met, n_canais,
        figsize=(5 * n_canais, 2.8 * n_met),
        sharex=True,
    )
    fig.suptitle(f"Métricas EMG — {nome_arquivo}", fontsize=14, y=1.01, fontweight="bold")

    for col, (res, nome_canal, cor) in enumerate(
        zip(features_por_canal, nomes_canais, CORES_MUSCULO)
    ):
        for row, (chave, label) in enumerate(metricas_labels):
            ax = axes[row][col]
            ax.plot(t, res[chave], color=cor, linewidth=1.2, alpha=0.9)
            ax.axvline(tempo_fadiga, color="#FF4B4B", linestyle="--",
                       linewidth=1.2, alpha=0.85, label="Fadiga" if row == 0 else "")
            ax.grid(True)
            if row == 0:
                ax.set_title(nome_canal, fontsize=9, color=cor, fontweight="bold")
                ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            if row == n_met - 1:
                ax.set_xlabel("Tempo (s)", fontsize=8)

    plt.tight_layout()
    fname = f"metricas_{nome_arquivo.replace('.csv','')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.show()
    print(f"Salvo: {fname}")


def plotar_fft_e_espectrograma(
    df: pd.DataFrame,
    nome_arquivo: str,
    tempo_fadiga: float,
    fs_t
) -> None:

    Fs       = fs_t
    Tjan     = 1.0
    nwin     = int(Tjan * Fs)
    win_ham  = windows.hamming(nwin)
    noverlap = nwin // 2

    for idx_emg, nome_emg in enumerate(EMG_NOMES):
        nome_curto = nome_emg.split(":")[0].replace("R ", "")
        cor = CORES_MUSCULO[idx_emg]

        if nome_emg not in df.columns:
            continue

        sinal = df[nome_emg].values - df[nome_emg].mean()
        N     = len(sinal)
        freqs = np.fft.fftfreq(N, d=1 / Fs)
        S     = np.fft.fft(sinal)
        mask  = (freqs >= 0) & (freqs <= 500)

        # FFT
        fig, ax = plt.subplots(figsize=(11, 3.5))
        fig.suptitle(f"FFT — {nome_curto} | {nome_arquivo}", fontsize=12, fontweight="bold")
        ax.plot(freqs[mask], np.abs(S)[mask], color=cor, linewidth=0.8)
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        plt.tight_layout()
        fname = f"fft_{idx_emg+1}_{nome_arquivo.replace('.csv','')}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight",
                    facecolor=plt.rcParams["figure.facecolor"])
        plt.show()

        # Espectrograma
        f, t_spec, Sxx = spectrogram(
            sinal, fs=Fs,
            window=win_ham, nperseg=nwin, noverlap=noverlap,
            nfft=nwin, scaling="density", mode="magnitude",
        )
        mask_f = f <= 250
        Sxx    = Sxx[mask_f, :] / (Sxx[mask_f, :].max() + 1e-8)

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(f"Espectrograma — {nome_curto} | {nome_arquivo}", fontsize=12, fontweight="bold")
        img = ax.pcolormesh(t_spec, f[mask_f], Sxx, shading="gouraud", cmap="inferno")
        ax.axvline(tempo_fadiga, color="#00FFFF", linestyle="--",
                   linewidth=1.4, label=f"Fadiga {tempo_fadiga:.1f}s")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequência (Hz)")
        ax.legend(fontsize=8)
        plt.colorbar(img, ax=ax, label="Amplitude norm.")
        plt.tight_layout()
        fname = f"espectrograma_{idx_emg+1}_{nome_arquivo.replace('.csv','')}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight",
                    facecolor=plt.rcParams["figure.facecolor"])
        plt.show()

    print(f"FFT e espectrogramas salvos para {nome_arquivo}")


def plotar_curva_fadiga_teste(
    t: np.ndarray,
    fadiga_suave: np.ndarray,
    pred_pinn: np.ndarray,
    tempo_fadiga_fisio: float,
    tempo_fadiga_pinn: float,
    historico: dict,
    nome_arquivo: str,
) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(20, 4))
    fig.suptitle(f"PINN — Detecção de Fadiga | {nome_arquivo}",
                 fontsize=13, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(t, fadiga_suave, color="#58A6FF", linewidth=1.8, label="Score fisiológico")
    ax1.axvline(tempo_fadiga_fisio, color="#FF4B4B", linestyle="--",
                linewidth=2, label=f"Fadiga fisiológica: {tempo_fadiga_fisio:.1f}s")
    ax1.fill_between(t, fadiga_suave, alpha=0.15, color="#58A6FF")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Índice de Fadiga")
    ax1.set_title("Score Fisiológico")
    ax1.legend(fontsize=8)
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(t, pred_pinn, color="#3FB950", linewidth=1.8, label="Predição PINN")
    ax2.plot(t, fadiga_suave, color="#58A6FF", linewidth=1.0,
             linestyle=":", alpha=0.6, label="Score fisiológico")
    ax2.axvline(tempo_fadiga_pinn, color="#F78166", linestyle="--",
                linewidth=2, label=f"Fadiga PINN: {tempo_fadiga_pinn:.1f}s")
    ax2.axvline(tempo_fadiga_fisio, color="#FF4B4B", linestyle=":",
                linewidth=1.5, alpha=0.7, label=f"Fadiga fisiológica: {tempo_fadiga_fisio:.1f}s")
    ax2.fill_between(t, pred_pinn, alpha=0.15, color="#3FB950")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Índice de Fadiga")
    ax2.set_title("PINN vs Fisiológico")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    fname = f"fadiga_pinn_{nome_arquivo.replace('.csv','')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.show()
    print(f"Salvo: {fname}")


def main():

    print("\n[1/4] Extraindo features e score de TREINO...")

    X, t = carregar_e_extrair(ARQUIVOS_TREINO)
    y_fisio, t_score = gerar_score_para_arquivos(ARQUIVOS_TREINO)

    n = min(len(X), len(y_fisio))
    X        = X[:n]
    t        = t[:n]
    y_fisio  = y_fisio[:n]

    print(f"      X shape    : {X.shape}")
    print(f"      y_fisio len: {len(y_fisio)}")

    t_norm   = normalizar_tempo(t).reshape(-1, 1)
    X_aug    = np.hstack([X, t_norm])

    X_tensor = torch.tensor(X_aug,             dtype=torch.float32)
    y_tensor = torch.tensor(y_fisio.reshape(-1, 1), dtype=torch.float32)

    print("\n[2/4] Treinando PINN...")
    modelo, historico = treinar_modelo(X_tensor, y_tensor)

    print("\n[3/4] Avaliando modelo...")

    for nome_conj, arquivos in [
        ("VALIDAÇÃO", ARQUIVOS_VALIDACAO),
        ("TESTE",     ARQUIVOS_TESTE),
    ]:
        X_ev, t_ev   = carregar_e_extrair(arquivos)
        score_ev, _  = gerar_score_para_arquivos(arquivos)
        n_ev         = min(len(X_ev), len(score_ev))

        t_norm_ev   = normalizar_tempo(t_ev[:n_ev])
        X_ev_aug    = np.hstack([X_ev[:n_ev], t_norm_ev.reshape(-1, 1)])
        X_ev_tensor = torch.tensor(X_ev_aug, dtype=torch.float32)

        avaliar_conjunto(modelo, X_ev_tensor, score_ev[:n_ev], nome_conj)

    print("\n[4/4] Gerando gráficos dos arquivos de TESTE...")

    nomes_curtos = [nome.split(":")[0].replace("R ", "") for nome in EMG_NOMES]

    for nome_arq in ARQUIVOS_TESTE:
        print(f"\n  ── Processando {nome_arq} ──")

        df_teste = pd.read_csv(f"dados_coleta_mateus/{nome_arq}")
        tempo_t  = df_teste["X [s]"].values
        fs_t     = 1.0 / np.mean(np.diff(tempo_t))

        feats_teste = [
            calcular_features_janela(df_teste[nome].values, fs_t, JANELA_SEGUNDOS)
            for nome in EMG_NOMES
        ]
        score_t, fadiga_suave_t = calcular_indice_fadiga_fisiologico(feats_teste)
        t_janelas_t = feats_teste[0]["tempo_medio"]

        idx_fisio, tempo_fadiga_fisio = detectar_ponto_fadiga(
            fadiga_suave_t, t_janelas_t, metodo="hibrido"
        )

        t_norm_t  = normalizar_tempo(t_janelas_t).reshape(-1, 1)
        X_t, _    = extrair_features_df(df_teste, tempo_t)
        n_t       = min(len(X_t), len(score_t))
        X_t_aug   = np.hstack([X_t[:n_t], t_norm_t[:n_t]])
        X_t_tensor = torch.tensor(X_t_aug, dtype=torch.float32)

        with torch.no_grad():
            pred_t = modelo(X_t_tensor).numpy().flatten()
            pred_t = (pred_t - pred_t.min()) / (pred_t.max() - pred_t.min() + 1e-8)

        pred_t_suave = (
            pd.Series(pred_t)
            .rolling(7, center=True, min_periods=1)
            .mean()
            .values
        )

        idx_pinn, tempo_fadiga_pinn = detectar_ponto_fadiga(
            pred_t_suave, t_janelas_t[:n_t], metodo="hibrido"
        )
        comparar_pinn_vs_fisiologico(
            fadiga_suave_t[:n_t], pred_t_suave, t_janelas_t[:n_t], nome_arq
        )

        plotar_metricas_simples(
            t_janelas_t, feats_teste, tempo_fadiga_fisio, nomes_curtos, nome_arq
        )

        plotar_fft_e_espectrograma(df_teste, nome_arq, tempo_fadiga_fisio, fs_t)

        plotar_curva_fadiga_teste(
            t_janelas_t[:n_t],
            fadiga_suave_t[:n_t],
            pred_t_suave,
            tempo_fadiga_fisio,
            tempo_fadiga_pinn,
            historico,
            nome_arq,
        )

    print("\nFinalizado com sucesso!")



if __name__ == "__main__":
    main()
