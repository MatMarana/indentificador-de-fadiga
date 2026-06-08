# Documentação sobre as funções do código

## Calcular as Métricas de Fadiga

```python
calcular_rms()
```

Durante a fadiga:
* RMS tende a aumentar.

```python
calcular_mav()
```

---

## ZCR

```python
calcular_zero_crossing_rate()
```

Conta quantas vezes o sinal cruza o zero.

Durante a fadiga:

* o conteúdo de alta frequência diminui;
* o ZCR tende a cair.

---

## FFT

```python
calcular_features_fft()
```

Foi utilizada FFT para observar o comportamento espectral do EMG.

As features escolhidas foram:

### Frequência Média (MNF)

```python
fft_media
```

Tende a diminuir com a fadiga.

---

### Frequência Mediana (MDF)

```python
fft_mediana
```

Uma das métricas mais utilizadas na literatura para análise de fadiga.

Também tende a diminuir.

---

### Desvio Espectral

```python
fft_desvio
```

Representa a dispersão das frequências.

---

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


