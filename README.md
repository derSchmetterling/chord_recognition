# Reconhecimento de Acordes


## Relatório

Um resumo de todos os passos realizados neste projeto está disponível no arquivo **project_notebook.ipynb**.


## Como reproduzir esse repositório?

Se você apenas deseja apenas reproduzir o dashboard, pode pular todos os passos definidos como opcionais.

### 1. (Opcional) Download dos dados
Faça o download da base de dados em https://guitarset.weebly.com/ na pasta raiz do diretório.
Os arquivos aqui utilizados são **audio_hex-pickup_original** e **annotation**.
Não é necessário descomprimir os arquivos.


### 2. Dependências
Faça o download das dependências disponíveis no arquivos **requirements.txt.**

### 3.  (Opcional) Preprocessamento

Abra o arquivo **preprocess.ipynb** para ser guiado pelos procedimentos de preprocessamento.

### 3.1 (Opcional) Análise Exploratória

Uma rápida análise exploratória dos metadados extraídos é feita em **exploratory.ipynb.**

### 4.  (Opcional) Extração de Características

Abra o arquivo **PCP.ipynb** para ser guiado pelos procedimentos de extração de características.

### 5. (Opcional) Modelos

Abra o arquivo **models_pcp.ipynb** para ver os modelos testados para este trabalho.

### 6. Dashboard

Para rodar o dashbord basta escrever 'streamlit run dashboard/app.py' na linha de comando.
