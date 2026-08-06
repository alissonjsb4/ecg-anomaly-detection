# ecg-anomaly-detection

Detecção de batimentos anômalos na base ECG5000 treinando apenas com exemplos
normais (classe única). Atributos extraídos nos domínios do tempo e da frequência;
detectores por distância euclidiana, distância de Mahalanobis e erro de
reconstrução PCA, implementados em NumPy — sem scikit-learn.

## Uso

    pip install -r requirements.txt
    cd codigo && python executar.py

Roda o experimento completo — extração de atributos, posto, 100 rodadas Monte
Carlo por configuração, redução PCA e curva ROC — e regrava `resultados/` em
cerca de 13 s. Semente fixa: duas execuções produzem os mesmos números.

## Protocolo

Treino com 80% dos batimentos normais; teste com os 20% restantes mais todas as
anomalias. Limiar de anomalia no percentil 95 do escore observado no treino.
Normalização z-score ajustada só no treino. Métricas reportadas como média de
100 rodadas.

## Resultados

Atributos do domínio do tempo (20 atributos, posto 15):

| Detector | Acurácia | Sensibilidade | F1 | Tempo |
|---|---|---|---|---|
| Euclidiano | 0,719 | 0,654 | 0,784 | 0,36 ms |
| Mahalanobis | 0,908 | 0,897 | 0,939 | 3,75 ms |
| PCA (reconstrução) | 0,598 | 0,500 | 0,660 | 2,71 ms |

Com redução PCA de 20 para 9 componentes (96,8% da variância) antes do detector:

| Detector | Acurácia | F1 | Tempo |
|---|---|---|---|
| Mahalanobis | 0,916 | 0,944 | 1,67 ms |

AUC do melhor detector: 0,962.

![Curva ROC](resultados/figuras/roc.png)

Atributos do domínio da frequência ficam em F1 ≤ 0,29 com qualquer detector: a
anomalia do ECG5000 está na forma do batimento, e a densidade espectral de
potência descarta a fase que carrega essa forma.

## Notas

- Mahalanobis supera o euclidiano porque os atributos são correlacionados (posto
  15 em 20 dimensões): a covariância pondera cada direção pela sua dispersão.
- Covariância mal-condicionada recebe regularização de Tikhonov com λ adaptativo:
  parte de `1e-6·tr(C)/p` e multiplica por 10 até o número de condição permitir a
  inversão.
- Reduzir de 20 para 9 componentes subiu o F1 (0,939 → 0,944) e cortou 55% do
  tempo: as direções descartadas eram ruído e dependência linear.
- O mesmo PCA falha como detector (F1 0,660) e vence como redutor antes do
  Mahalanobis.
- Distâncias vetorizadas com `np.einsum`, sem laço por amostra.

## Estrutura

    codigo/deteccao_anomalias.py   atributos, detectores, métricas e protocolo
    codigo/executar.py             experimento completo; salva tabelas e figuras
    dados/ecg5000.csv              4998 batimentos × 140 amostras (2919 normais)
    resultados/                    tabelas CSV, resumo JSON e figuras
    relatorio/                     relatório em LaTeX e PDF
    notebook.ipynb                 versão notebook, com a discussão passo a passo

## Stack

Python, NumPy, SciPy, pandas, matplotlib.

---

Origem: projeto final da disciplina de Reconhecimento de Padrões (UFC).
