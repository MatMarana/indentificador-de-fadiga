# Documentação sobre as funções do código

No ínício do código foi determinado a SEED no torch, isso foi feito para que os resultados sejam reproduzíveis.
Ela faz com que a PINN inicie o treinamento sempre com a mesma sequência de pesos, evitando variabilidade.

## Determinação de Constantes
Foram estipuladas as seguintes constantes:

```python
JANELA_SEGUNDOS = 5
EPOCHS          = 2000
LR              = 1e-3
LAMBDA_MONO     = 0.0
LAMBDA_REG      = 0.01
```

Sendo em ordem, o tamanho das janelas temporais, as épocas de treinamento e por último
- **LR** (*Learning Rate*)**:** Define a taxa de aprendizado da Rede Neural, será usada como peso no otmizador ADAM
- **LAMBDA_MONO** (*Controle de Perda da Monotonicidade*)**:** É utilizada para penalizar a Rede caso a fadiga caia no treinamento
- **LAMBDA_REG** (*Controla a regularização da PINN*)**:** Com o valor definido a rede opta pelas curvas mais suaves

## Métricas para análise de fadiga

### RMS
> Root Mean Square

Realiza a raiz quadrada da média dos quadrados do sinal na janela
Tende a aumentar conforme a fadiga aumenta

```python
def calcular_rms(janela: np.ndarray) -> float:
    return np.sqrt(np.mean(janela ** 2))
```

### MAV
> Mean Absolute Value

É obtida através da média dos valores absolutos da janela
Tende a aumentar conforme a fadigda cresce

```python
def calcular_mav(janela: np.ndarray) -> float:
    return np.mean(np.abs(janela))
```

### ZCR
> Zero Crossing Rate

Mede a quantidade de cruzamentos do sinal em relação ao zero.
Diminui conforme o aumento da fadiga.

```python
def calcular_zero_crossing_rate(janela: np.ndarray) -> float:
    sinais = np.sign(janela)
    sinais[sinais == 0] = 1
    zcr = np.sum(sinais[:-1] != sinais[1:])
    return zcr / len(janela)
```

### FFT
> Transformata de Fourier

Foi utilizada FFT para observar o comportamento espectral do EMG.
As features escolhidas foram:
- Frequência Média (MNF)
- Frequência Mediana (MDef)
- Desvio Espectral

As features espectrais MNF e MDF foram escolhidas por serem biomarcadores clássicos de fadiga muscular, refletindo o deslocamento do conteúdo espectral para frequências mais baixas. O Desvio Espectral foi adicionado para capturar alterações na dispersão da energia do espectro, fornecendo informação complementar à posição média e mediana das frequências. 

Dessa forma, o conjunto de features descreve não apenas onde a energia está concentrada, mas também como ela está distribuída ao longo do espectro.

```python
def calcular_features_fft(janela: np.ndarray, fs: float) -> tuple:
    # Features espectrais via rfft com janela Hamming e suavização gaussiana.

    N = len(janela)

    # Janela Hamming e rFFT
    janela_win = janela * np.hamming(N)
    S     = np.fft.rfft(janela_win)
    freqs = np.fft.rfftfreq(N, d=1 / fs)

    # Suavização Gaussiana
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
```

### Espectrograma
Permite observar a evolução temporal do conteúdo espectral.

Foram extraídas:
- **Energia espectral:** utilizada para quantificar a intensidade média da atividade elétrica muscular no domínio da frequência.
- **Frequência dominante:** empregada para identificar a região espectral de maior concentração de energia.

O espectrograma foi utilizado para analisar a evolução temporal do conteúdo espectral do sinal EMG. Diferentemente da FFT, que fornece uma visão global das frequências presentes em uma janela, o espectrograma permite observar como a distribuição de energia varia ao longo do tempo. 

A partir dele foram extraídas a energia espectral média e a frequência dominante, que auxiliam na identificação das alterações fisiológicas associadas ao processo de fadiga muscular.

```python
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
```

## Construção do Índice Fisiológico

Como não existe uma medida direta e contínua de fadiga muscular disponível nos dados coletados, foi construído um índice fisiológico de fadiga baseado em características temporais e espectrais do sinal EMG. 

Esse índice foi utilizado como referência supervisora durante o treinamento da PINN, permitindo que a rede aprendesse uma representação contínua do processo de fadiga a partir de conhecimento fisiológico previamente estabelecido na literatura

### Calculando o índice de fadiga fisiologico

Inicialmente, são utilizadas features extraídas do domínio do tempo e da frequência, como RMS, MAV, ZCR, frequência média (MNF), frequência mediana (MDF), energia espectral e frequência dominante. Todas as variáveis são normalizadas para uma escala entre 0 e 1, garantindo que métricas com amplitudes maiores não tenham influência desproporcional no resultado final.

Em seguida, cada feature recebe um peso de acordo com sua relevância para a identificação da fadiga muscular. As métricas que tendem a aumentar durante a fadiga, como RMS, MAV e energia espectral, são utilizadas diretamente. Já as que normalmente diminuem, como ZCR, MNF, MDF e frequência dominante, são invertidas para que todas sigam a mesma interpretação: quanto maior o valor, maior o nível de fadiga.

O score é calculado individualmente para cada músculo por meio de uma média ponderada dessas features. Depois, os scores de todos os canais são combinados em uma única medida global de fadiga e normalizados novamente. Para reduzir oscilações causadas por ruídos do sinal, aplica-se uma suavização utilizando média móvel.

### Detecção da fadiga

Por fim, o ponto de fadiga é determinado por um método híbrido que considera tanto o valor do índice de fadiga quanto sua taxa de crescimento ao longo do tempo. A combinação dessas informações permite identificar o instante em que a fadiga se torna mais evidente, gerando uma referência fisiológica contínua que será utilizada para supervisionar o treinamento da PINN.

O método híbrido foi utilizado para tornar a detecção do ponto de fadiga mais robusta e fisiologicamente coerente. Utilizar apenas o valor do índice de fadiga poderia levar à identificação dos instantes finais do exercício, onde a fadiga costuma atingir seus maiores valores, enquanto utilizar apenas a taxa de crescimento da curva poderia tornar o método sensível a oscilações e ruídos do sinal.

```python
def detectar_ponto_fadiga(
    fadiga_suave: np.ndarray,
    tempo: np.ndarray,
    metodo: str = "hibrido",
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
    else:
        raise ValueError(f"Método desconhecido: {metodo!r}")

    return idx, float(tempo[idx])
```

O método híbrido funciona combinando duas informações extraídas da curva de fadiga suavizada: o próprio nível de fadiga e a sua taxa de crescimento ao longo do tempo. Inicialmente, é calculado o gradiente da curva, que representa a velocidade com que a fadiga está aumentando em cada instante. 

Em seguida, tanto a curva de fadiga quanto o gradiente são normalizados para uma escala entre 0 e 1, permitindo que sejam comparados e combinados adequadamente. A pontuação final é obtida por meio de uma média ponderada, na qual 85% do peso é atribuído ao nível de fadiga e 15% ao gradiente. 

Dessa forma, o algoritmo não procura apenas o ponto onde a fadiga é máxima, mas também considera regiões em que ela está crescendo de forma relevante. Por fim, o instante correspondente ao maior valor dessa pontuação híbrida é identificado como o ponto de fadiga muscular detectado pelo método.

## PINN
> Physics-Informed Neural Network

### Arquitetura

```python
nn.Linear(n_features, 128), nn.Tanh(),
nn.Linear(128, 128), nn.Tanh(),
nn.Linear(128, 64), nn.Tanh(),
nn.Linear(64, 1),
```

A rede recebe como entrada todas as features extraídas do EMG e produz como saída um único valor: o índice de fadiga.

Foram utilizadas três camadas ocultas porque o relacionamento entre as features do EMG e a fadiga não é linear. O RMS, a MAV, a MDF e as demais métricas interagem entre si de forma complexa, e uma rede muito pequena poderia não ter capacidade suficiente para aprender esses padrões.

Os 128 neurônios das primeiras camadas fornecem capacidade para aprender relações complexas, enquanto a redução para 64 neurônios na última camada ajuda a condensar as informações antes da saída.
Uma rede neural comum aprende apenas pelos dados.

### Ativação

Função **Tanh**

A função de ativação Tangente Hiperbólica (Tanh) foi utilizada nas camadas ocultas da rede neural por produzir saídas suaves e continuamente diferenciáveis no intervalo entre -1 e 1. 

Essa característica é especialmente importante em Physics-Informed Neural Networks (PINNs), pois o treinamento envolve o cálculo de derivadas de primeira e segunda ordem da saída da rede.

### Função de Loss

# Treinamento

Otimizador:

```python
Adam
```

Taxa inicial:

```python
1e-3
```

Número de épocas:

```python
2000
```

Scheduler:

```python
ReduceLROnPlateau
```

Quando o treinamento para de melhorar, o learning rate é reduzido automaticamente.

---

# Avaliação

As métricas utilizadas foram:

## MSE

Erro quadrático médio.

---

## MAE

Erro absoluto médio.

---

## R²

Coeficiente de determinação.

Avalia o quanto a PINN consegue explicar o comportamento do score fisiológico.

## Curva de Fadiga

Compara:

* Score fisiológico
* Predição da PINN

Também mostra:

* Fadiga fisiológica
* Fadiga prevista pela PINN



