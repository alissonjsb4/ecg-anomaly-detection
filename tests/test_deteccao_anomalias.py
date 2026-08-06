# -*- coding: utf-8 -*-
"""Testes dos detectores, métricas e protocolo.

As implementações são próprias (sem scikit-learn), então cada teste compara
contra uma referência independente: scipy para o Mahalanobis, construções
com resposta conhecida para o resto.
"""
import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis as mahalanobis_scipy

import deteccao_anomalias as det
from executar import curva_roc

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- extração
def test_atributos_tempo_forma_e_finitude():
    S = rng.normal(size=(50, 140))
    X = det.atributos_tempo(S)
    assert X.shape == (50, 20)
    assert np.isfinite(X).all()


def test_atributos_frequencia_forma_e_finitude():
    S = rng.normal(size=(50, 140))
    X = det.atributos_frequencia(S)
    assert X.shape == (50, 15)
    assert np.isfinite(X).all()


# ---------------------------------------------------------------- normalização
def test_padronizar_usa_estatisticas_do_treino():
    Xtr = rng.normal(5.0, 3.0, size=(500, 4))
    Xte = rng.normal(0.0, 1.0, size=(200, 4))
    Xtr_n, Xte_n = det.padronizar(Xtr, Xte)

    # treino fica com média 0 e desvio 1
    np.testing.assert_allclose(Xtr_n.mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Xtr_n.std(0), 1.0, rtol=1e-9)

    # teste é normalizado com média/desvio DO TREINO (sem vazamento):
    # média real 0 sob treino N(5,3) deve cair perto de (0-5)/3 ≈ -1,67
    assert np.all(Xte_n.mean(0) < -1.0)


# ---------------------------------------------------------------- detectores
def test_mahalanobis_bate_com_scipy():
    Xtr = rng.normal(size=(200, 5))
    Xte = rng.normal(size=(20, 5))
    _, s_te, lam = det.escore_mahalanobis(Xtr, Xte)

    assert lam == 0.0  # bem-condicionada: sem regularização
    mu = Xtr.mean(0)
    Cinv = np.linalg.inv(np.cov(Xtr, rowvar=False))
    ref = np.array([mahalanobis_scipy(x, mu, Cinv) ** 2 for x in Xte])
    np.testing.assert_allclose(s_te, ref, rtol=1e-10)


def test_mahalanobis_regulariza_covariancia_singular():
    X = rng.normal(size=(100, 4))
    Xtr = np.column_stack([X, X[:, 0]])  # coluna duplicada: covariância singular
    _, s_te, lam = det.escore_mahalanobis(Xtr, Xtr[:10])

    assert lam > 0.0
    assert np.isfinite(s_te).all()


def test_escore_pca_zera_em_subespaco_conhecido():
    # dados construídos num subespaço de dimensão 2 dentro de R^5
    base = rng.normal(size=(2, 5))
    Xtr = rng.normal(size=(300, 2)) @ base
    err_tr, err_te, q = det.escore_pca(Xtr, Xtr[:10], var_retida=0.999)

    assert q <= 2
    np.testing.assert_allclose(err_tr, 0.0, atol=1e-12)
    np.testing.assert_allclose(err_te, 0.0, atol=1e-12)


def test_escore_euclidiano_referencia_direta():
    Xtr = rng.normal(size=(50, 3))
    Xte = rng.normal(size=(10, 3))
    _, s_te = det.escore_euclidiano(Xtr, Xte)
    ref = ((Xte - Xtr.mean(0)) ** 2).sum(1)
    np.testing.assert_allclose(s_te, ref, rtol=1e-12)


# ---------------------------------------------------------------- métricas
def test_metricas_matriz_de_confusao_conhecida():
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    y_pos = np.array([False, True, True, False])
    # limiar 1,5: prediz positivo para escores 2 e 3 -> TP=1 FP=1 TN=1 FN=1
    acc, sens, espec, prec, f1 = det.metricas(scores, 1.5, y_pos)
    assert (acc, sens, espec, prec, f1) == (0.5, 0.5, 0.5, 0.5, 0.5)


def test_curva_roc_separacao_perfeita_e_invertida():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    y_pos = np.array([False, False, True, True])

    _, _, auc = curva_roc(scores, y_pos)
    assert auc == pytest.approx(1.0)

    _, _, auc_inv = curva_roc(-scores, y_pos)
    assert auc_inv == pytest.approx(0.0)


# ---------------------------------------------------------------- protocolo
def test_uma_rodada_sem_anomalia_no_treino():
    Xneg = rng.normal(size=(100, 6))
    Xpos = rng.normal(3.0, 1.0, size=(30, 6))
    resultados, q_red = det.uma_rodada(Xneg, Xpos, np.random.default_rng(1))

    assert q_red is None
    assert set(resultados) == {"Euclidiano", "Mahalanobis", "PCA"}
    for _, (m, dt, s_te, y_pos) in resultados.items():
        # teste = 20% dos negativos + todos os positivos
        assert len(s_te) == 20 + 30
        assert y_pos.sum() == 30
        assert dt > 0
        assert all(0.0 <= v <= 1.0 for v in m)
