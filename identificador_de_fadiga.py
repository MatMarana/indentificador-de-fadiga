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

# ── Imports ──────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.signal import spectrogram, windows
from sklearn.metrics import mean_squared_error, classification_report

# ── Configurações globais ─────────────────────────────────────────────────────

# Arquivos de dados
ARQUIVOS_TREINO = [
    "ColetaMateus1Rep.csv",
    "MateusDia2Rep2.csv",
    "ColetaMateus3Rep.csv",
    "MateusDia2Rep1.csv",
]

ARQUIVOS_TESTE = [
    "ColetaMateusDia3Rep1.csv",
    "ColetaMateusDia3Rep2.csv",
    "MateusDia2Rep3.csv",
]

ARQUIVOS_VALIDACAO = [
    "ColetaMateusDia3Rep3.csv",
    "ColetaMateus2Rep.csv",
]

# Nomes das colunas EMG no CSV
EMG_NOMES = [
    "R BRACHIORADIALIS: EMG 1 [Volts]",
    "R EXTENSOR CARPI ULNARIS: EMG 2 [Volts]",
    "R PALMARIS LONGUS: EMG 3 [Volts]",
    "R EXTENSOR DIGITORUM: EMG 4 [Volts]",
]

# Tamanho da janela de análise (segundos)
JANELA_SEGUNDOS = 5

# Hiperparâmetros de treinamento
EPOCHS       = 2000
LR           = 1e-3
LAMBDA_MONO  = 0.1   # peso da perda de monotonicidade (física)
LAMBDA_REG   = 0.01  # peso da regularização L2 (suavidade)
LIMIAR_FADIGA = 0.7  # limiar de classificação binária de fadiga

# Estilo dos gráficos
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


# =============================================================================
# BLOCO 1 — EXTRAÇÃO DE FEATURES
# =============================================================================

def calcular_rms(janela: np.ndarray) -> float:
    """Root Mean Square: mede a amplitude eficaz do sinal."""
    return np.sqrt(np.mean(janela ** 2))


def calcular_mav(janela: np.ndarray) -> float:
    """Mean Absolute Value: robusto a outliers, correlaciona com força."""
    return np.mean(np.abs(janela))


def calcular_zero_crossing_rate(janela: np.ndarray) -> float:
    """
    Zero Crossing Rate: taxa de cruzamentos por zero.
    Correlaciona com conteúdo espectral médio do sinal EMG.
    """
    sinais = np.sign(janela)
    sinais[sinais == 0] = 1
    zcr = np.sum(sinais[:-1] != sinais[1:])
    return zcr / len(janela)


def calcular_features_fft(janela: np.ndarray, fs: float) -> tuple:
    """
    Features espectrais via FFT:
      - Frequência média (Mean Frequency — MNF)
      - Frequência mediana (Median Frequency — MDF)
      - Desvio padrão espectral
    Durante fadiga, MNF e MDF tendem a diminuir (shift para baixas freq.).
    """
    N  = len(janela)
    S  = np.fft.fft(janela)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    mask      = freqs >= 0
    freqs     = freqs[mask]
    magnitude = np.abs(S[mask])

    mag_total   = np.sum(magnitude) + 1e-8
    freq_media  = np.sum(freqs * magnitude) / mag_total

    # MDF: frequência que divide o espectro ao meio
    cum_energia = np.cumsum(magnitude)
    freq_mediana = freqs[np.searchsorted(cum_energia, cum_energia[-1] / 2)]

    desvio = np.std(magnitude)

    return freq_media, freq_mediana, desvio


def calcular_features_janela(emg: np.ndarray, fs: float, segundos: float) -> dict:
    """
    Aplica janelamento deslizante e calcula todas as features por janela.

    Retorna dicionário com arrays de comprimento n_janelas para cada feature.
    """
    amostras_por_janela = int(segundos * fs)
    n_janelas = len(emg) // amostras_por_janela

    resultados = {k: [] for k in [
        "tempo_medio", "rms", "mav", "zcr",
        "fft_media", "fft_mediana", "fft_desvio",
        "spec_energia", "spec_freq_dom"
    ]}

    # Janela de Hamming para o espectrograma (reduz vazamento espectral)
    nwin    = min(amostras_por_janela, 256)
    win_ham = windows.hamming(nwin)
    noverlap = nwin // 2

    # Vetor de tempo proporcional ao índice de janela
    tempo_base = np.arange(len(emg)) / fs

    for i in range(n_janelas):
        inicio = i * amostras_por_janela
        fim    = inicio + amostras_por_janela
        janela = emg[inicio:fim]

        # Features temporais
        resultados["rms"].append(calcular_rms(janela))
        resultados["mav"].append(calcular_mav(janela))
        resultados["zcr"].append(calcular_zero_crossing_rate(janela))
        resultados["tempo_medio"].append(np.mean(tempo_base[inicio:fim]))

        # Features espectrais (FFT)
        fm, fmed, fstd = calcular_features_fft(janela, fs)
        resultados["fft_media"].append(fm)
        resultados["fft_mediana"].append(fmed)
        resultados["fft_desvio"].append(fstd)

        # Features do espectrograma
        f, _, Sxx = spectrogram(
            janela, fs=fs,
            window=win_ham,
            nperseg=nwin,
            noverlap=noverlap,
            scaling="density"
        )
        Sxx = np.abs(Sxx)
        energia    = np.mean(Sxx)
        freq_dom   = f[np.argmax(np.mean(Sxx, axis=1))]
        resultados["spec_energia"].append(energia)
        resultados["spec_freq_dom"].append(freq_dom)

    return {k: np.array(v) for k, v in resultados.items()}


def extrair_features_df(df: pd.DataFrame, tempo: np.ndarray) -> tuple:
    """
    Itera sobre todos os canais EMG e concatena as features de cada músculo.

    Retorna:
      X       : array (n_janelas, n_features_total)
      tempo_m : array (n_janelas,) com tempo médio de cada janela
    """
    fs = 1.0 / np.mean(np.diff(tempo))
    features_canais = []
    tempo_m = None

    for nome in EMG_NOMES:
        emg = df[nome].values
        res = calcular_features_janela(emg, fs, JANELA_SEGUNDOS)

        if tempo_m is None:
            tempo_m = res["tempo_medio"]

        feats = np.column_stack([
            res["rms"],
            res["mav"],
            res["zcr"],
            res["fft_media"],
            res["fft_mediana"],
            res["fft_desvio"],
            res["spec_energia"],
            res["spec_freq_dom"],
        ])
        features_canais.append(feats)

    return np.hstack(features_canais), tempo_m


# =============================================================================
# BLOCO 2 — PINN (Physics-Informed Neural Network)
# =============================================================================

class RedeNeuralFadiga(nn.Module):
    """
    Rede neural densa com ativações Tanh.
    Tanh é preferida em PINNs por ter derivadas analíticas suaves,
    facilitando o cálculo das restrições físicas via autodiferenciação.

    Ref: Raissi et al. (2019), J. Comput. Phys.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),   # saída em [0,1] — índice de fadiga normalizado
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
    Função de perda física para detecção de fadiga EMG.

    Componentes:
    ───────────────────────────────────────────────────────────────────────
    1. L_data  — MSE entre predição e target (tempo normalizado).
                 Faz a rede aprender a progressão temporal.

    2. L_mono  — Penaliza derivada temporal NEGATIVA (df/dt < 0).
                 Fadiga muscular é um processo monotonicamente crescente:
                 o índice de fadiga não pode diminuir no tempo.
                 Formulação: relu(-df/dt)
                 Ref: Christodoulou et al. (2022); Fang & Zhan (2020).

    3. L_reg   — Penaliza a norma L2 da derivada segunda (suavidade).
                 Evita oscilações bruscas na curva de fadiga, que seriam
                 fisiologicamente implausíveis.
                 Ref: formulação de regularização de Tikhonov.

    Loss total: L = L_data + λ_mono · L_mono + λ_reg · L_reg
    ───────────────────────────────────────────────────────────────────────
    """
    x = x.clone().requires_grad_(True)
    y_pred = modelo(x)

    # Gradiente de primeira ordem em relação à última coluna (tempo)
    grads_1 = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    df_dt = grads_1[:, -1].reshape(-1, 1)

    # Gradiente de segunda ordem (curvatura)
    grads_2 = torch.autograd.grad(
        outputs=df_dt,
        inputs=x,
        grad_outputs=torch.ones_like(df_dt),
        create_graph=True,
        retain_graph=True,
    )[0]
    d2f_dt2 = grads_2[:, -1].reshape(-1, 1)

    # Perdas individuais
    L_data = nn.MSELoss()(y_pred, y)
    L_mono = torch.mean(torch.relu(-df_dt))          # monotonicidade
    L_reg  = torch.mean(d2f_dt2 ** 2)               # suavidade

    L_total = L_data + lambda_mono * L_mono + lambda_reg * L_reg

    return L_total, L_data.item(), L_mono.item(), L_reg.item()


def treinar_modelo(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    epochs:   int = EPOCHS,
    lr:       float = LR,
) -> tuple:
    """
    Treina a PINN e registra o histórico de perdas para análise.

    Retorna:
      modelo    : modelo treinado
      historico : dict com listas de perda por época
    """
    modelo = RedeNeuralFadiga(n_features=X_tensor.shape[1])
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, patience=100, factor=0.5,
    )

    historico = {"total": [], "data": [], "mono": [], "reg": []}

    print("─" * 50)
    print(f"{'Época':>8} │ {'L_total':>10} │ {'L_data':>10} │ {'L_mono':>10} │ {'L_reg':>10}")
    print("─" * 50)

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
            print(
                f"{epoca:>8} │ {L_total.item():>10.5f} │"
                f" {L_data:>10.5f} │ {L_mono:>10.5f} │ {L_reg:>10.5f}"
            )

    print("─" * 50)
    return modelo, historico


# =============================================================================
# BLOCO 3 — GRÁFICOS
# =============================================================================

def _marca_fadiga(ax, tempo_fadiga: float, **kwargs):
    """Adiciona linha vertical de fadiga e anotação em qualquer eixo."""
    ax.axvline(tempo_fadiga, color="#FF4B4B", linestyle="--", linewidth=1.5, **kwargs)
    ax.annotate(
        f"Fadiga\n{tempo_fadiga:.1f}s",
        xy=(tempo_fadiga, ax.get_ylim()[1]),
        xytext=(8, -20),
        textcoords="offset points",
        color="#FF4B4B",
        fontsize=8,
    )


def plotar_metricas_simples(
    t: np.ndarray,
    features_por_canal: list,
    tempo_fadiga: float,
    nomes_canais: list,
):
    """
    Página 1 — Métricas temporais simples (RMS, MAV, ZCR) e espectrais (FFT)
    para cada canal EMG. Tudo em uma única figura organizada.
    """
    n_canais = len(features_por_canal)

    # Layout: 4 colunas (canais) × 6 linhas (métricas)
    metricas_labels = [
        ("rms",         "RMS (V)"),
        ("mav",         "MAV (V)"),
        ("zcr",         "ZCR"),
        ("fft_media",   "Freq. Média (Hz)"),
        ("fft_mediana", "Freq. Mediana (Hz)"),
        ("fft_desvio",  "Desvio Espectral"),
    ]

    n_met = len(metricas_labels)
    fig, axes = plt.subplots(
        n_met, n_canais,
        figsize=(5 * n_canais, 2.8 * n_met),
        sharex=True,
    )
    fig.suptitle("Métricas EMG por Canal Muscular", fontsize=16, y=1.01, fontweight="bold")

    for col, (res, nome_canal, cor) in enumerate(
        zip(features_por_canal, nomes_canais, CORES_MUSCULO)
    ):
        for row, (chave, label) in enumerate(metricas_labels):
            ax = axes[row][col]
            ax.plot(t, res[chave], color=cor, linewidth=1.2, alpha=0.9)
            ax.grid(True)

            # Título do canal apenas na linha do topo
            if row == 0:
                ax.set_title(nome_canal, fontsize=9, color=cor, fontweight="bold")

            # Label do eixo Y apenas na coluna da esquerda
            if col == 0:
                ax.set_ylabel(label, fontsize=8)

            # Marca de fadiga
            ylim = ax.get_ylim()
            ax.axvline(tempo_fadiga, color="#FF4B4B", linestyle="--", linewidth=1.2, alpha=0.8)

            # Label do eixo X apenas na última linha
            if row == n_met - 1:
                ax.set_xlabel("Tempo (s)", fontsize=8)

    plt.tight_layout()
    plt.savefig("metricas_simples.png", dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.show()
    print("[✓] Figura salva: metricas_simples.png")


def plotar_fft_e_espectrograma(
    df_list: list,
    nomes_arquivos: list,
    tempo_fadiga: float,
):
    """
    Página 2 — FFT completo do sinal + Espectrograma estilo MATLAB
    para cada canal EMG. Inclui marca de fadiga onde aplicável.
    """
    Fs       = 1925.93
    Tjan     = 1.0
    nwin     = int(Tjan * Fs)
    win_ham  = windows.hamming(nwin)
    noverlap = nwin // 2

    for idx_emg, nome_emg in enumerate(EMG_NOMES):
        nome_curto = nome_emg.split(":")[0].replace("R ", "")
        cor = CORES_MUSCULO[idx_emg]

        # ── FFT do primeiro arquivo de treino ────────────────────────────
        fig_fft, ax_fft = plt.subplots(figsize=(11, 3.5))
        fig_fft.suptitle(f"FFT — {nome_curto}", fontsize=13, fontweight="bold")

        for df_temp, nome_arq in zip(df_list, nomes_arquivos):
            if nome_emg not in df_temp.columns:
                continue
            sinal = df_temp[nome_emg].values - df_temp[nome_emg].mean()
            N     = len(sinal)
            S     = np.fft.fft(sinal)
            freqs = np.fft.fftfreq(N, d=1 / Fs)
            mask  = (freqs >= 0) & (freqs <= 500)
            ax_fft.plot(freqs[mask], np.abs(S)[mask], linewidth=0.8,
                        alpha=0.7, label=nome_arq)

        ax_fft.set_xlabel("Frequência (Hz)")
        ax_fft.set_ylabel("Amplitude")
        ax_fft.legend(fontsize=7, ncol=2)
        ax_fft.grid(True)
        plt.tight_layout()
        plt.savefig(f"fft_{idx_emg+1}.png", dpi=150, bbox_inches="tight",
                    facecolor=plt.rcParams["figure.facecolor"])
        plt.show()

        # ── Espectrograma por arquivo ────────────────────────────────────
        n_arqs = len(df_list)
        fig_sp, axes_sp = plt.subplots(
            1, n_arqs,
            figsize=(5 * n_arqs, 4),
            sharey=True,
        )
        fig_sp.suptitle(f"Espectrograma — {nome_curto}", fontsize=13, fontweight="bold")

        if n_arqs == 1:
            axes_sp = [axes_sp]

        for i, (df_temp, nome_arq) in enumerate(zip(df_list, nomes_arquivos)):
            ax = axes_sp[i]

            if nome_emg not in df_temp.columns:
                ax.set_visible(False)
                continue

            sinal = df_temp[nome_emg].values - df_temp[nome_emg].mean()

            f, t_spec, Sxx = spectrogram(
                sinal, fs=Fs,
                window=win_ham,
                nperseg=nwin,
                noverlap=noverlap,
                nfft=nwin,
                scaling="density",
                mode="magnitude",
            )

            # Limitar a 250 Hz
            mask = f <= 250
            f    = f[mask]
            Sxx  = Sxx[mask, :]
            Sxx  = Sxx / (Sxx.max() + 1e-8)

            img = ax.pcolormesh(t_spec, f, Sxx, shading="gouraud", cmap="inferno")
            ax.axvline(tempo_fadiga, color="#00FFFF", linestyle="--",
                       linewidth=1.4, label=f"Fadiga {tempo_fadiga:.1f}s")
            ax.set_title(nome_arq, fontsize=8)
            ax.set_xlabel("Tempo (s)")
            if i == 0:
                ax.set_ylabel("Frequência (Hz)")
            plt.colorbar(img, ax=ax, label="Amplitude norm.")

        plt.tight_layout()
        plt.savefig(f"espectrograma_{idx_emg+1}.png", dpi=150, bbox_inches="tight",
                    facecolor=plt.rcParams["figure.facecolor"])
        plt.show()

    print("[✓] Figuras FFT e Espectrograma salvas.")


def plotar_curva_fadiga(
    t: np.ndarray,
    fadiga_suave: np.ndarray,
    tempo_fadiga: float,
    historico: dict,
):
    """
    Figura final — Curva de fadiga predita e histórico de perdas da PINN.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("PINN — Resultado e Convergência", fontsize=14, fontweight="bold")

    # Curva de fadiga
    ax1 = axes[0]
    ax1.plot(t, fadiga_suave, color="#58A6FF", linewidth=1.8, label="Índice de fadiga")
    ax1.axvline(tempo_fadiga, color="#FF4B4B", linestyle="--",
                linewidth=2, label=f"Fadiga detectada: {tempo_fadiga:.2f}s")
    ax1.fill_between(t, fadiga_suave, alpha=0.15, color="#58A6FF")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Índice de Fadiga (normalizado)")
    ax1.set_title("Detecção de Fadiga Muscular")
    ax1.legend()
    ax1.grid(True)

    # Histórico de perda
    ax2 = axes[1]
    epocas = range(len(historico["total"]))
    ax2.plot(epocas, historico["total"], color="#58A6FF",  label="L total",  linewidth=1.5)
    ax2.plot(epocas, historico["data"],  color="#3FB950",  label="L data",   linewidth=1.2, alpha=0.85)
    ax2.plot(epocas, historico["mono"],  color="#F78166",  label="L mono",   linewidth=1.2, alpha=0.85)
    ax2.plot(epocas, historico["reg"],   color="#D2A8FF",  label="L reg",    linewidth=1.2, alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss (log)")
    ax2.set_title("Convergência da PINN")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("curva_fadiga_pinn.png", dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.show()
    print("[✓] Figura salva: curva_fadiga_pinn.png")


# =============================================================================
# BLOCO 4 — PIPELINE PRINCIPAL
# =============================================================================

def carregar_e_extrair(arquivos: list, pasta: str = "dados_coleta_mateus") -> tuple:
    """Lê múltiplos CSVs e empilha features e tempo médio."""
    X_list, t_list = [], []
    for arq in arquivos:
        df = pd.read_csv(f"{pasta}/{arq}")
        X, t = extrair_features_df(df, df["X [s]"].values)
        X_list.append(X)
        t_list.append(t)
    return np.vstack(X_list), np.concatenate(t_list)


def normalizar_tempo(t: np.ndarray) -> np.ndarray:
    """Normaliza vetor de tempo para [0, 1]."""
    return (t - t.min()) / (t.max() - t.min())


def avaliar_conjunto(modelo, X_tensor, t_norm, nome: str):
    """Prediz, calcula MSE e exibe classification report."""
    with torch.no_grad():
        pred = modelo(X_tensor).numpy().flatten()

    mse = mean_squared_error(t_norm, pred)
    y_true = (t_norm > LIMIAR_FADIGA).astype(int)
    y_pred = (pred > LIMIAR_FADIGA).astype(int)

    print(f"\n{'═'*50}")
    print(f"  {nome}")
    print(f"{'═'*50}")
    print(f"  MSE : {mse:.6f}")
    print(classification_report(y_true, y_pred, target_names=["Sem fadiga", "Com fadiga"]))


def main():

    # ─── TREINO ──────────────────────────────────────────────────────────────
    print("\n[1/4] Extraindo features de TREINO...")
    X, t = carregar_e_extrair(ARQUIVOS_TREINO)
    print(f"      Shape: {X.shape}")

    t_norm = normalizar_tempo(t).reshape(-1, 1)
    X_aug  = np.hstack([X, t_norm])

    X_tensor = torch.tensor(X_aug,  dtype=torch.float32)
    y_tensor = torch.tensor(t_norm, dtype=torch.float32)

    # ─── TREINAMENTO DA PINN ─────────────────────────────────────────────────
    print("\n[2/4] Treinando PINN...")
    modelo, historico = treinar_modelo(X_tensor, y_tensor)

    # Predição e detecção do ponto de fadiga
    with torch.no_grad():
        pred_treino = modelo(X_tensor).numpy().flatten()

    fadiga_suave = pd.Series(pred_treino).rolling(5, center=True, min_periods=1).mean().values
    dfadiga      = np.gradient(fadiga_suave)
    ponto_fadiga = np.argmax(dfadiga)
    tempo_fadiga = t[ponto_fadiga]

    print(f"\n  ▶ Fadiga detectada em: {tempo_fadiga:.2f} segundos")

    # ─── VALIDAÇÃO E TESTE ───────────────────────────────────────────────────
    print("\n[3/4] Avaliando modelo...")

    for nome, arquivos in [("VALIDAÇÃO", ARQUIVOS_VALIDACAO), ("TESTE", ARQUIVOS_TESTE)]:
        X_ev, t_ev = carregar_e_extrair(arquivos)
        t_norm_ev  = normalizar_tempo(t_ev)
        X_ev_aug   = np.hstack([X_ev, t_norm_ev.reshape(-1, 1)])
        X_ev_tensor = torch.tensor(X_ev_aug, dtype=torch.float32)
        avaliar_conjunto(modelo, X_ev_tensor, t_norm_ev, nome)

    # ─── GRÁFICOS ────────────────────────────────────────────────────────────
    print("\n[4/4] Gerando gráficos...")

    # Calcula features por canal separadamente para plots
    df_treino_0 = pd.read_csv(f"dados_coleta_mateus/{ARQUIVOS_TREINO[0]}")
    tempo_0     = df_treino_0["X [s]"].values
    fs_0        = 1.0 / np.mean(np.diff(tempo_0))

    nomes_curtos = [n.split(":")[0].replace("R ", "") for n in EMG_NOMES]
    features_por_canal = [
        calcular_features_janela(df_treino_0[nome].values, fs_0, JANELA_SEGUNDOS)
        for nome in EMG_NOMES
    ]
    t_janelas = features_por_canal[0]["tempo_medio"]

    # Página 1: métricas simples
    plotar_metricas_simples(t_janelas, features_por_canal, tempo_fadiga, nomes_curtos)

    # Página 2: FFT e espectrograma (arquivos de treino)
    dfs_treino = [
        pd.read_csv(f"dados_coleta_mateus/{arq}") for arq in ARQUIVOS_TREINO
    ]
    plotar_fft_e_espectrograma(dfs_treino, ARQUIVOS_TREINO, tempo_fadiga)

    # Curva de fadiga + convergência
    plotar_curva_fadiga(t, fadiga_suave, tempo_fadiga, historico)

    print("\n[✓] Pipeline finalizado com sucesso!")


# =============================================================================

if __name__ == "__main__":
