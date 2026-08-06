# -*- coding: utf-8 -*-
"""Detecção de anomalias em ECG por modelagem de classe única.

Base: ECG5000 -- 4998 batimentos de 140 amostras.
Rótulo: 1 = batimento normal (negativo), 0 = anomalia (positivo).

Pipeline:
  1. Extração de atributos nos domínios do tempo e da frequência -> matriz X (N x p).
  2. Posto (rank) de X, para verificar dependência linear entre atributos.
  3. Três detectores: distância euclidiana, distância de Mahalanobis e erro de
     reconstrução por PCA (autoencoder linear).
     Treino: 80% dos negativos. Teste: 20% dos negativos + todos os positivos.
     Limiar de anomalia: percentil do escore observado no treino.
  4. Nr = 100 rodadas Monte Carlo -> acurácia, sensibilidade, especificidade,
     precisão, F1 e tempo, com média e desvio-padrão.
  5. PCA como redutor de dimensionalidade antes dos detectores de distância.
  6. Curva ROC e AUC do melhor detector.

Escrito do zero em NumPy/SciPy: nenhum estimador de scikit-learn é usado.
Origem: projeto final da disciplina de Reconhecimento de Padrões (UFC).
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DIR_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ = os.path.join(DIR_BASE, "dados", "ecg5000.csv")
DIR_FIGS = os.path.join(DIR_BASE, "resultados", "figuras")
DIR_TABS = os.path.join(DIR_BASE, "resultados", "tabelas")

NR = 100
PERCENTIL = 95.0          # limiar de anomalia (percentil do escore no treino)
VAR_PCA_DETECTOR = 0.95   # variância retida pelo autoencoder linear (detector PCA)
VAR_PCA_REDUTOR = 0.95    # variância retida na redução de dimensionalidade
SEED = 42

CORES = {"Euclidiano": "#2a78d6", "Mahalanobis": "#1baf7a", "PCA": "#eb6834"}


def configurar_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.grid": True, "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


# ---------------------------------------------------------------- dados
def carregar_sinais():
    d = np.loadtxt(ARQ, delimiter=",")
    sinais = d[:, :-1]
    normal = d[:, -1].astype(int) == 1     # negativos
    return sinais, normal


# ---------------------------------------------------------------- extração de atributos
def atributos_tempo(S):
    """Atributos estatísticos no domínio do tempo (um vetor por batimento)."""
    dif = np.diff(S, axis=1)
    dif2 = np.diff(S, n=2, axis=1)
    var = S.var(axis=1)
    mob = np.sqrt(dif.var(axis=1) / var)                       # mobilidade de Hjorth
    comp = np.sqrt(dif2.var(axis=1) / dif.var(axis=1)) / mob   # complexidade de Hjorth
    feats = np.column_stack([
        S.mean(1), S.std(1), var, np.sqrt((S ** 2).mean(1)),   # média, dp, var, rms
        S.min(1), S.max(1), np.ptp(S, axis=1), np.median(S, 1),
        np.mean(np.abs(S - S.mean(1, keepdims=True)), axis=1),  # desvio absoluto médio
        stats.skew(S, axis=1), stats.kurtosis(S, axis=1),
        (S ** 2).sum(1),                                        # energia
        np.mean(np.abs(dif), axis=1),                           # variação média
        np.mean(np.abs(np.diff(np.sign(S), axis=1)) > 0, axis=1),  # taxa de cruzamentos por zero
        np.percentile(S, 25, axis=1), np.percentile(S, 75, axis=1),
        stats.iqr(S, axis=1), var, mob, comp,                   # atividade/mobilidade/complexidade Hjorth
    ])
    return feats


def atributos_frequencia(S):
    """Atributos no domínio da frequência a partir da densidade espectral de potência."""
    F = np.abs(np.fft.rfft(S, axis=1)) ** 2          # PSD (periodograma)
    F = F[:, 1:]                                      # remove componente DC
    pot_total = F.sum(1) + 1e-12
    Fn = F / pot_total[:, None]                       # PSD normalizada (distribuição)
    freqs = np.arange(1, F.shape[1] + 1)
    centroide = (Fn * freqs).sum(1)                   # centroide espectral
    espalh = np.sqrt((Fn * (freqs - centroide[:, None]) ** 2).sum(1))  # espalhamento
    entropia = -(Fn * np.log(Fn + 1e-12)).sum(1)      # entropia espectral
    freq_pico = 1 + np.argmax(F, axis=1)              # frequência de pico
    cumsum = np.cumsum(F, axis=1)
    freq_mediana = (cumsum >= pot_total[:, None] / 2).argmax(axis=1) + 1  # frequência mediana
    planura = np.exp(np.mean(np.log(F + 1e-12), axis=1)) / (F.mean(1) + 1e-12)  # planura espectral
    nb = 5                                            # potência em 5 bandas
    bandas = np.array([b.sum(1) for b in np.array_split(F, nb, axis=1)]).T
    bandas_rel = bandas / pot_total[:, None]
    feats = np.column_stack([
        pot_total, centroide, espalh, entropia, freq_pico, freq_mediana, planura,
        bandas_rel, F.max(1), F.mean(1), F.std(1),
    ])
    return feats


# ---------------------------------------------------------------- normalização
def padronizar(Xtr, Xte):
    """z-score ajustado apenas no treino (negativos)."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-12
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ---------------------------------------------------------------- detectores
def escore_euclidiano(Xtr, Xte):
    mu = Xtr.mean(0)
    s_tr = ((Xtr - mu) ** 2).sum(1)
    s_te = ((Xte - mu) ** 2).sum(1)
    return s_tr, s_te


def escore_mahalanobis(Xtr, Xte):
    mu = Xtr.mean(0)
    C = np.cov(Xtr, rowvar=False)
    p = C.shape[0]
    lam = 0.0
    # regularização de Tikhonov se mal-condicionada (ver TC2)
    s = np.linalg.svd(C, compute_uv=False)
    if s[-1] / s[0] < 1e-10:
        lam = 1e-6 * np.trace(C) / p
        while True:
            sr = np.linalg.svd(C + lam * np.eye(p), compute_uv=False)
            if sr[-1] / sr[0] >= 1e-10:
                break
            lam *= 10
    Cinv = np.linalg.inv(C + lam * np.eye(p))
    dtr = Xtr - mu
    dte = Xte - mu
    s_tr = np.einsum("ij,jk,ik->i", dtr, Cinv, dtr)
    s_te = np.einsum("ij,jk,ik->i", dte, Cinv, dte)
    return s_tr, s_te, lam


def escore_pca(Xtr, Xte, var_retida):
    """Autoencoder linear: escore = erro de reconstrução quadrático no subespaço PCA."""
    mu = Xtr.mean(0)
    U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    ve = np.cumsum(S ** 2) / np.sum(S ** 2)
    q = int(np.searchsorted(ve, var_retida) + 1)
    V = Vt[:q].T
    def erro(X):
        Xc = X - mu
        rec = (Xc @ V) @ V.T
        return ((Xc - rec) ** 2).sum(1)
    return erro(Xtr), erro(Xte), q


# ---------------------------------------------------------------- métricas
def metricas(scores_te, limiar, y_pos):
    """y_pos: máscara booleana de positivos (anomalias) no conjunto de teste."""
    pred_pos = scores_te > limiar
    TP = np.sum(pred_pos & y_pos)
    FP = np.sum(pred_pos & ~y_pos)
    TN = np.sum(~pred_pos & ~y_pos)
    FN = np.sum(~pred_pos & y_pos)
    acc = (TP + TN) / (TP + TN + FP + FN)
    sens = TP / max(TP + FN, 1)
    espec = TN / max(TN + FP, 1)
    prec = TP / max(TP + FP, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return acc, sens, espec, prec, f1


# ---------------------------------------------------------------- protocolo
def uma_rodada(Xneg, Xpos, rng, com_pca_redutor=False, var_red=VAR_PCA_REDUTOR, q_fixo=None):
    """Retorna dict modelo -> (métricas, tempo, escores_teste, y_pos) para uma rodada."""
    n = len(Xneg)
    idx = rng.permutation(n)
    corte = int(0.8 * n)
    tr, te_neg = idx[:corte], idx[corte:]
    Xtr_raw = Xneg[tr]
    Xte_raw = np.vstack([Xneg[te_neg], Xpos])
    y_pos = np.concatenate([np.zeros(len(te_neg), bool), np.ones(len(Xpos), bool)])

    Xtr, Xte = padronizar(Xtr_raw, Xte_raw)

    q_red = None
    if com_pca_redutor:
        mu = Xtr.mean(0)
        U, Sv, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
        if q_fixo is None:
            ve = np.cumsum(Sv ** 2) / np.sum(Sv ** 2)
            q_red = int(np.searchsorted(ve, var_red) + 1)
        else:
            q_red = q_fixo
        Vr = Vt[:q_red].T
        Xtr = (Xtr - mu) @ Vr
        Xte = (Xte - mu) @ Vr

    resultados = {}
    detectores = ["Euclidiano", "Mahalanobis"] if com_pca_redutor else ["Euclidiano", "Mahalanobis", "PCA"]
    for nome in detectores:
        t0 = time.perf_counter()
        if nome == "Euclidiano":
            s_tr, s_te = escore_euclidiano(Xtr, Xte)
        elif nome == "Mahalanobis":
            s_tr, s_te, _ = escore_mahalanobis(Xtr, Xte)
        else:
            s_tr, s_te, _ = escore_pca(Xtr, Xte, VAR_PCA_DETECTOR)
        limiar = np.percentile(s_tr, PERCENTIL)
        m = metricas(s_te, limiar, y_pos)
        dt = time.perf_counter() - t0
        resultados[nome] = (m, dt, s_te, y_pos)
    return resultados, q_red


def resumo(vals):
    a = np.array(vals)
    return a.mean(0), a.std(0, ddof=1)


def experimento(Xneg, Xpos, dominio, com_pca=False, q_fixo=None):
    rng = np.random.default_rng(SEED)
    acumulado = {}
    q_reg = []
    for r in range(NR):
        res, q_red = uma_rodada(Xneg, Xpos, rng, com_pca_redutor=com_pca, q_fixo=q_fixo)
        if q_red is not None:
            q_reg.append(q_red)
        for nome, (m, dt, _, _) in res.items():
            acumulado.setdefault(nome, []).append(list(m) + [1e3 * dt])
    linhas = []
    for nome, vals in acumulado.items():
        media, dp = resumo(vals)
        linhas.append({"dominio": dominio, "detector": nome,
                       "acuracia": media[0], "acuracia_dp": dp[0],
                       "sensibilidade": media[1], "sensibilidade_dp": dp[1],
                       "especificidade": media[2], "especificidade_dp": dp[2],
                       "precisao": media[3], "precisao_dp": dp[3],
                       "f1": media[4], "f1_dp": dp[4],
                       "tempo_ms": media[5], "tempo_ms_dp": dp[5]})
    q_moda = int(stats.mode(q_reg, keepdims=False).mode) if q_reg else None
    return linhas, q_moda
