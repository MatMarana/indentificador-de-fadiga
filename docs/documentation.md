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
- **LAMBDA_REG** (*Controla a regularização da PINN*)**:** Com o valor de finido a rede opta pelas curvas mais suaves

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

## FFT
> Transformata de Fourier

Foi utilizada FFT para observar o comportamento espectral do EMG.
As features escolhidas foram:
- Frequência Média (MNF)
- Frequência Mediana (MDef)
- Desvio Espectral

## Espectrograma

```python
spectrogram()
```

Permite observar a evolução temporal do conteúdo espectral.

Foram extraídas:

### Energia espectral

```python
spec_energia
```

Tende a aumentar.

---

### Frequência dominante

```python
spec_freq_dom
```

Tende a diminuir.

---

# Parte 2 — Construção do Índice Fisiológico

## O que eu estava pensando?

Antes de treinar a PINN, eu precisava criar um "professor".

Esse professor é o score fisiológico.

---

## Normalização

Cada feature é normalizada:

```python
(x - min) / (max - min)
```

Assim todas ficam na mesma escala.

---

## Features que aumentam com fadiga

Usadas diretamente:

```python
RMS
MAV
Energia espectral
```

---

## Features que diminuem com fadiga

Invertidas:

```python
1 - feature
```

São elas:

```python
ZCR
MNF
MDF
Frequência dominante
```

---

## Score Final

O score é calculado pela média ponderada:

```python
score = soma(features ponderadas)
```

---

## Suavização

Aplicado:

```python
rolling mean
```

Objetivo:

Remover oscilações fisiologicamente irreais.

---

# Parte 3 — Detecção da Fadiga

Foram implementados dois métodos.

---

## Método Híbrido

```python
metodo="hibrido"
```

Combina:

* valor do score;
* derivada temporal.

```python
score = 0.85*fadiga + 0.15*gradiente
```

A ideia era detectar o instante em que a fadiga cresce rapidamente.

---

## Método Limiar

```python
metodo="limiar"
```

Utiliza percentis.

Exemplo:

```python
percentil 75
```

A fadiga é detectada quando o score ultrapassa esse valor.

---

# Parte 4 — PINN

## O que eu estava pensando?

Uma rede neural comum aprende apenas pelos dados.

Eu queria inserir conhecimento fisiológico.

Por isso escolhi uma PINN.

---

# Arquitetura

```text
Entrada
 ↓
128 neurônios
 ↓
128 neurônios
 ↓
64 neurônios
 ↓
Saída
```

Ativação:

```python
Tanh
```

Motivo:

PINNs funcionam melhor com funções suaves e diferenciáveis.

---

# Entrada da Rede

A entrada contém:

```text
32 features EMG
+
tempo normalizado
```

---

# Saída

A rede produz:

```text
Índice de fadiga previsto
```

---

# Função de Perda

## L_data

```python
MSE(predição, score fisiológico)
```

Faz a rede aprender o comportamento observado.

---

## L_mono

```python
relu(-df/dt)
```

Penaliza quedas na fadiga.

Pensamento:

Fisicamente a fadiga deveria crescer ao longo do exercício.

---

## L_reg

```python
(d²f/dt²)²
```

Penaliza oscilações bruscas.

Objetivo:

Produzir curvas mais suaves e fisiologicamente plausíveis.

---

## Loss Final

```python
L_total =
L_data
+ λmono * L_mono
+ λreg * L_reg
```

---

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

---

# Gráficos Gerados

## Métricas Temporais

```text
RMS
MAV
ZCR
MNF
MDF
Desvio espectral
```

---

## FFT

Mostra a distribuição espectral do sinal.

---

## Espectrograma

Mostra a evolução temporal das frequências.

---

## Curva de Fadiga

Compara:

* Score fisiológico
* Predição da PINN

Também mostra:

* Fadiga fisiológica
* Fadiga prevista pela PINN

---

# Interpretação dos Resultados

Idealmente espera-se que:

* a curva da PINN siga a curva fisiológica;
* o instante de fadiga previsto seja próximo do fisiológico;
* o erro temporal seja pequeno;
* o R² seja elevado.

Diferenças pequenas são esperadas porque a PINN tenta generalizar padrões presentes em múltiplas coletas.


