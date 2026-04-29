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
import matplotlib.patches as mpatches

from scipy.signal import spectrogram, windows
from sklearn.metrics import mean_squared_error, classification_report

# ── Configurações globais ─────────────────────────────────────────────────────

# Arquivos de dados
ARQUIVOS_TREINO = [
    "ColetaMateus2Rep.csv",
    "MateusDia2Rep2.csv",
    "ColetaMateus3Rep.csv",
    "MateusDia2Rep1.csv",
    "ColetaMateusDia3Rep3.csv"
]

ARQUIVOS_TESTE = [
    "ColetaMateusDia3Rep1.csv",
    "ColetaMateusDia3Rep2.csv",
    "MateusDia2Rep3.csv",
]

ARQUIVOS_VALIDACAO = [
    "ColetaMateus1Rep.csv"
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

# Paleta de cores por arquivo
CORES_ARQUIVO = ["#58A6FF", "#3FB950", "#F78166", "#D2A8FF", "#FFA657", "#79C0FF"]

# Estilo escuro global
plt.rcParams.update({
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#C9D1D9",
    "xtick.color":       "#C9D1D9",
    "ytick.color":       "#C9D1D9",
    "text.color":        "#C9D1D9",
    "grid.color":        "#21262D",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.framealpha": 0.3,
    "font.family":       "monospace",
})


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

def _banda_arquivos(ax, limites, alpha=0.06):
    """Faixas verticais alternadas para separar arquivos no eixo de tempo."""
    for i, (t0, t1, _) in enumerate(limites):
        if i % 2 == 0:
            ax.axvspan(t0, t1, color="#FFFFFF", alpha=alpha, lw=0)
        ax.axvline(t0, color="#30363D", linewidth=0.6, alpha=0.8)

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


def plotar_pagina1_metricas(arquivos, tempo_fadiga, pasta="dados_coleta_mateus"):
    """
    Figura 1 — RMS, MAV e ZCR para cada arquivo de treino.
    Colunas = arquivos | Linhas = métricas.
    Eixo X em segundos reais (tempo acumulado global).
    Cada arquivo usa a média dos 4 canais EMG para representatividade.
    """
    n_arqs   = len(arquivos)
    metricas = [("rms", "RMS (V)"), ("mav", "MAV (V)"), ("zcr", "ZCR")]
    n_met    = len(metricas)

    fig, axes = plt.subplots(n_met, n_arqs,
                             figsize=(4.5 * n_arqs, 3 * n_met),
                             sharey="row")
    fig.suptitle("Página 1 — Métricas Temporais por Arquivo",
                 fontsize=14, fontweight="bold", y=1.01)

    offset = 0.0
    for col, arq in enumerate(arquivos):
        df    = pd.read_csv(f"{pasta}/{arq}")
        tempo = df["X [s]"].values
        fs    = 1.0 / np.mean(np.diff(tempo))
        dur   = tempo[-1] - tempo[0]
        cor   = CORES_ARQUIVO[col % len(CORES_ARQUIVO)]

        # Média dos 4 canais EMG para cada feature
        rms_list, mav_list, zcr_list, t_list = [], [], [], []
        for nome_emg in EMG_NOMES:
            res = calcular_features_janela(df[nome_emg].values, fs, JANELA_SEGUNDOS)
            rms_list.append(res["rms"])
            mav_list.append(res["mav"])
            zcr_list.append(res["zcr"])
            t_list.append(res["tempo_medio"])

        # Tempo global acumulado para este arquivo
        t_jan  = (t_list[0] - t_list[0][0]) + offset
        dados  = {
            "rms": np.mean(rms_list, axis=0),
            "mav": np.mean(mav_list, axis=0),
            "zcr": np.mean(zcr_list, axis=0),
        }

        for row, (chave, label) in enumerate(metricas):
            ax = axes[row][col] if n_arqs > 1 else axes[row]
            ax.plot(t_jan, dados[chave], color=cor, linewidth=1.4)
            ax.fill_between(t_jan, dados[chave], alpha=0.12, color=cor)
            _marca_fadiga(ax, tempo_fadiga, label=(row == 0 and col == 0))
            ax.grid(True)
            ax.set_xlim(t_jan[0], t_jan[-1])

            if row == 0:
                nome_curto = arq.replace(".csv", "")
                ax.set_title(nome_curto, fontsize=9, color=cor, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == n_met - 1:
                ax.set_xlabel("Tempo (s)", fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")

        offset += dur

    plt.tight_layout()
    plt.savefig("pagina1_metricas.png", dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.show()
    print("[✓] pagina1_metricas.png salvo")


def plotar_pagina2_fft_espectrograma(arquivos, tempo_fadiga,
                                      pasta="dados_coleta_mateus"):
    """
    Figura 2 — FFT do sinal e Espectrograma por arquivo.
    Linha 1: FFT (domínio da frequência).
    Linha 2: Espectrograma (tempo × frequência).
    Marca de fadiga aparece no espectrograma quando dentro do intervalo.
    """
    Fs       = 1925.93
    nwin     = int(1.0 * Fs)
    win_ham  = windows.hamming(nwin)
    noverlap = nwin // 2
    n_arqs   = len(arquivos)

    fig, axes = plt.subplots(2, n_arqs,
                             figsize=(5 * n_arqs, 7),
                             gridspec_kw={"hspace": 0.4})
    fig.suptitle("Página 2 — FFT e Espectrograma por Arquivo",
                 fontsize=14, fontweight="bold", y=1.01)

    offset = 0.0
    for col, arq in enumerate(arquivos):
        df    = pd.read_csv(f"{pasta}/{arq}")
        tempo = df["X [s]"].values
        dur   = tempo[-1] - tempo[0]
        cor   = CORES_ARQUIVO[col % len(CORES_ARQUIVO)]

        # Média dos canais EMG para representar o arquivo
        sinal = np.mean([df[n].values for n in EMG_NOMES], axis=0)
        sinal = sinal - sinal.mean()

        nome_curto = arq.replace(".csv", "")
        ax_fft = axes[0][col] if n_arqs > 1 else axes[0]
        ax_sp  = axes[1][col] if n_arqs > 1 else axes[1]

        # ── FFT ─────────────────────────────────────────────────────────
        N     = len(sinal)
        S     = np.fft.fft(sinal)
        freqs = np.fft.fftfreq(N, d=1 / Fs)
        mask  = (freqs >= 0) & (freqs <= 400)

        ax_fft.plot(freqs[mask], np.abs(S)[mask], color=cor, linewidth=0.9)
        ax_fft.fill_between(freqs[mask], np.abs(S)[mask], alpha=0.15, color=cor)
        ax_fft.set_title(nome_curto, fontsize=9, color=cor, fontweight="bold")
        ax_fft.set_xlabel("Frequência (Hz)", fontsize=8)
        ax_fft.grid(True)
        if col == 0:
            ax_fft.set_ylabel("Amplitude", fontsize=9)

        # ── Espectrograma ────────────────────────────────────────────────
        f, t_sp, Sxx = spectrogram(
            sinal, fs=Fs, window=win_ham,
            nperseg=nwin, noverlap=noverlap,
            nfft=nwin, scaling="density", mode="magnitude",
        )
        mask_f = f <= 250
        Sxx    = Sxx[mask_f, :] / (Sxx[mask_f, :].max() + 1e-8)

        img = ax_sp.pcolormesh(t_sp, f[mask_f], Sxx,
                               shading="gouraud", cmap="inferno")
        plt.colorbar(img, ax=ax_sp, label="Ampl. norm.", pad=0.02)

        # Marca de fadiga no tempo relativo ao arquivo
        t_fadiga_local = tempo_fadiga - offset
        if 0 <= t_fadiga_local <= dur:
            ax_sp.axvline(t_fadiga_local, color="#00FFFF", linestyle="--",
                          linewidth=1.5, label=f"Fadiga {tempo_fadiga:.1f}s")
            ax_sp.legend(fontsize=7, loc="upper right")

        ax_sp.set_xlabel("Tempo (s)", fontsize=8)
        if col == 0:
            ax_sp.set_ylabel("Freq. (Hz)", fontsize=8)

        offset += dur

    plt.tight_layout()
    plt.savefig("pagina2_fft_espectrograma.png", dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.show()
    print("[✓] pagina2_fft_espectrograma.png salvo")


def plotar_fadiga_e_convergencia(t, fadiga_suave, tempo_fadiga,
                                  limites, historico):
    """
    Figura 3 — Índice de fadiga ao longo do tempo global e histórico de perda.
    Faixas verticais alternadas indicam os limites de cada arquivo de treino.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("PINN — Resultado e Convergência",
                 fontsize=14, fontweight="bold")

    # ── Curva de fadiga ──────────────────────────────────────────────────
    ax1 = axes[0]
    _banda_arquivos(ax1, limites)
    ax1.plot(t, fadiga_suave, color="#58A6FF", linewidth=2,
             label="Índice de fadiga", zorder=3)
    ax1.fill_between(t, fadiga_suave, alpha=0.12, color="#58A6FF", zorder=2)
    _marca_fadiga(ax1, tempo_fadiga)

    # Legenda: um patch colorido por arquivo
    patches = [
        mpatches.Patch(color=CORES_ARQUIVO[i % len(CORES_ARQUIVO)],
                       label=nome.replace(".csv",""), alpha=0.7)
        for i, (_, _, nome) in enumerate(limites)
    ]
    patches.append(
        plt.Line2D([0],[0], color="#FF4B4B", linestyle="--",
                   label=f"Fadiga: {tempo_fadiga:.1f}s")
    )
    ax1.legend(handles=patches, fontsize=7, ncol=2, loc="upper left")
    ax1.set_xlabel("Tempo acumulado (s)")
    ax1.set_ylabel("Índice de Fadiga")
    ax1.set_title("Detecção de Fadiga Muscular")
    ax1.grid(True)

    # ── Histórico de perda ───────────────────────────────────────────────
    ax2 = axes[1]
    ep  = range(len(historico["total"]))
    ax2.plot(ep, historico["total"], color="#58A6FF", label="L total",  lw=1.8)
    ax2.plot(ep, historico["data"],  color="#3FB950", label="L data",   lw=1.2, alpha=0.85)
    ax2.plot(ep, historico["mono"],  color="#F78166", label="L mono",   lw=1.2, alpha=0.85)
    ax2.plot(ep, historico["reg"],   color="#D2A8FF", label="L reg",    lw=1.2, alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss (escala log)")
    ax2.set_title("Convergência da PINN")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("curva_fadiga_pinn.png", dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.show()
    print("[✓] curva_fadiga_pinn.png salvo")


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

def carregar_arquivos(arquivos: list, pasta: str = "dados_coleta_mateus") -> tuple:
    """
    Carrega múltiplos CSVs e constrói um eixo de tempo GLOBAL contínuo.

    *** CORREÇÃO DO GRÁFICO FRAGMENTADO ***
    Cada arquivo começa onde o anterior terminou (offset acumulado),
    em vez de resetar o tempo para zero. Isso garante uma curva de
    fadiga contínua e sem segmentos desconexos.

    Retorna:
      X           : (N_total, n_features)
      t_global    : (N_total,) tempo real acumulado em segundos
      limites_arq : lista de (t_inicio, t_fim, nome_arquivo)
  plotar_metricas_simples  """
    X_list, t_list, limites = [], [], []
    offset = 0.0

    for arq in arquivos:
        df    = pd.read_csv(f"{pasta}/{arq}")
        tempo = df["X [s]"].values
        dur   = tempo[-1] - tempo[0]

        X, t_jan = extrair_features_df(df, tempo)

        # Normaliza tempo local para [0, dur] e soma offset global
        t_local  = t_jan - t_jan[0]
        t_global = t_local + offset

        limites.append((offset, offset + dur, arq))
        offset += dur

        X_list.append(X)
        t_list.append(t_global)

    return np.vstack(X_list), np.concatenate(t_list), limites


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
    _, t_global, limites_treino = carregar_arquivos(ARQUIVOS_TREINO)
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
    df_treino_0 = pd.read_csv(f"dados_coleta_mateus/{ARQUIVOS_TESTE[0]}")
    tempo_0     = df_treino_0["X [s]"].values
    fs_0        = 1.0 / np.mean(np.diff(tempo_0))

    nomes_curtos = [n.split(":")[0].replace("R ", "") for n in EMG_NOMES]
    features_por_canal = [
        calcular_features_janela(df_treino_0[nome].values, fs_0, JANELA_SEGUNDOS)
        for nome in EMG_NOMES
    ]
    t_janelas = features_por_canal[0]["tempo_medio"]

    # Página 1: métricas simples
    plotar_pagina1_metricas(ARQUIVOS_TESTE, tempo_fadiga)

    # Página 2: FFT e espectrograma (arquivos de treino)
    plotar_pagina2_fft_espectrograma(ARQUIVOS_TESTE, tempo_fadiga)

    # Curva de fadiga + convergência
    plotar_fadiga_e_convergencia(t_global, fadiga_suave, tempo_fadiga, limites_treino, historico)

    print("\n[✓] Pipeline finalizado com sucesso!")


# =============================================================================

if __name__ == "__main__":
