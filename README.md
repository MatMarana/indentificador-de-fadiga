# Detecção de Fadiga Muscular via EMG com PINN
> Este projeto utiliza sinais de Eletromiografia (EMG) coletados durante exercícios físicos para identificar o instante de fadiga muscular.

**A abordagem combina:**
* Processamento de sinais biomédicos;
* Extração de características temporais e espectrais;
* Construção de um índice fisiológico de fadiga;
* Treinamento de uma Physics-Informed Neural Network (PINN).

A ideia principal foi criar um sistema capaz de aprender o comportamento fisiológico da fadiga muscular a partir dos sinais EMG e posteriormente detectar automaticamente o momento em que a fadiga ocorre.

## Estrutura do Projeto

```text
Projeto/
│
├── dados_coleta_mateus/
│   ├── ColetaMateus1Rep.csv
│   ├── ColetaMateus2Rep.csv
│   ├── ColetaMateus3Rep.csv
│   ├── ...
│
├── identificador_de_fadiga.py
│
└── README.md
```

## Para executar

Clone este repositório

```bash
git clone https://github.com/MatMarana/indentificador-de-fadiga.git
```

Instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

Rode o arquivo:

```bash
python identificador_de_fadiga.py
```

## Fluxo Geral do Sistema

```text
CSV EMG
   ↓
Extração de Features
   ↓
Índice Fisiológico
   ↓
Treinamento PINN
   ↓
Predição
   ↓
Detecção de Fadiga
   ↓
Gráficos e Métricas
```
--- 

Caso queira saber sobre o funcionamendo de alguma parte específica leia o arquivo [Documentation.md](https://github.com/MatMarana/indentificador-de-fadiga/blob/main/docs/documentation.md)
