# -*- coding: utf-8 -*-
"""Roda o experimento completo e salva tabelas e figuras em resultados/."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import deteccao_anomalias as det


def main():
    det.configurar_matplotlib()
    os.makedirs(det.DIR_FIGS, exist_ok=True)
    os.makedirs(det.DIR_TABS, exist_ok=True)

    S, normal = det.carregar_sinais()
    print(f"ECG5000: {S.shape[0]} batimentos de {S.shape[1]} amostras")
    print(f"  negativos (normais): {normal.sum()} | positivos (anomalias): {(~normal).sum()}")

    # figura ilustrativa: exemplos de batimento normal e anômalo
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(S[np.where(normal)[0][0]], color="#2a78d6", label="normal (negativo)")
    ax.plot(S[np.where(~normal)[0][0]], color="#e34948", label="anomalia (positivo)")
    ax.set_xlabel("amostra"); ax.set_ylabel("amplitude (normalizada)")
    ax.set_title("Exemplos de batimentos ECG5000"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(det.DIR_FIGS, "exemplos_ecg.png")); plt.close(fig)

    extratores = {"Tempo": det.atributos_tempo, "Frequência": det.atributos_frequencia}

    # ---------- extração de atributos e posto da matriz ----------
    Xs, postos = {}, []
    for dom, fn in extratores.items():
        X = fn(S)
        Xs[dom] = X
        rk = np.linalg.matrix_rank(X)
        postos.append({"dominio": dom, "N": X.shape[0], "p": X.shape[1],
                       "posto": rk, "posto_completo": rk == X.shape[1]})
        print(f"[{dom}] X = {X.shape[0]} x {X.shape[1]} | posto = {rk}")
    pd.DataFrame(postos).to_csv(os.path.join(det.DIR_TABS, "posto_atributos.csv"), index=False)

    # ---------- detectores sobre os atributos originais ----------
    linhas_t1 = []
    q_pca_detector = {}
    for dom, X in Xs.items():
        Xneg, Xpos = X[normal], X[~normal]
        linhas, _ = det.experimento(Xneg, Xpos, dom, com_pca=False)
        linhas_t1 += linhas
        # registra q do autoencoder linear na primeira rodada de treino (informativo)
        rng_q = np.random.default_rng(det.SEED)
        idxq = rng_q.permutation(len(Xneg))
        Xtrq, Xteq = det.padronizar(Xneg[idxq[:int(0.8 * len(Xneg))]], Xpos)
        _, _, q = det.escore_pca(Xtrq, Xteq, det.VAR_PCA_DETECTOR)
        q_pca_detector[dom] = q
    tab1 = pd.DataFrame(linhas_t1)
    tab1.to_csv(os.path.join(det.DIR_TABS, "tabela1_sem_reducao.csv"), index=False, float_format="%.6f")
    print("\n=== Tabela 1 (sem redução) ===")
    print(tab1[["dominio", "detector", "acuracia", "sensibilidade", "especificidade",
                "precisao", "f1", "tempo_ms"]].to_string(index=False,
                float_format=lambda x: f"{x:.4f}"))

    # escolhe o melhor domínio pela F1 média (para atividades 6 e 7)
    melhor_dom = tab1.groupby("dominio")["f1"].mean().idxmax()
    print(f"\nMelhor domínio de atributos (F1 média): {melhor_dom}")

    # ---------- PCA como redutor de dimensionalidade (VE(q) + tabela 2) ----------
    Xbest = Xs[melhor_dom]
    Xneg, Xpos = Xbest[normal], Xbest[~normal]
    # VE(q) da primeira rodada de treino
    rng = np.random.default_rng(det.SEED)
    idx = rng.permutation(len(Xneg))
    Xtr_raw = Xneg[idx[:int(0.8 * len(Xneg))]]
    Xtr, _ = det.padronizar(Xtr_raw, Xtr_raw)
    Sv = np.linalg.svd(Xtr - Xtr.mean(0), compute_uv=False)
    ve = np.cumsum(Sv ** 2) / np.sum(Sv ** 2)
    q_escolhido = int(np.searchsorted(ve, det.VAR_PCA_REDUTOR) + 1)
    print(f"Reducao PCA: q escolhido = {q_escolhido} (VE = {ve[q_escolhido-1]*100:.2f}% >= {det.VAR_PCA_REDUTOR*100:.0f}%)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, len(ve) + 1), 100 * ve, marker="o", markersize=4, color="#2a78d6")
    ax.axhline(100 * det.VAR_PCA_REDUTOR, color="#e34948", linestyle="--",
               label=f"{det.VAR_PCA_REDUTOR*100:.0f}% de variância")
    ax.axvline(q_escolhido, color="#52514e", linestyle=":", label=f"q = {q_escolhido}")
    ax.set_xlabel("número de componentes q"); ax.set_ylabel("variância explicada VE(q) (%)")
    ax.set_title(f"Variância explicada acumulada — atributos de {melhor_dom.lower()}")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(det.DIR_FIGS, "variancia_explicada.png")); plt.close(fig)

    linhas_t2, _ = det.experimento(Xneg, Xpos, melhor_dom, com_pca=True, q_fixo=q_escolhido)
    tab2 = pd.DataFrame(linhas_t2)
    tab2.to_csv(os.path.join(det.DIR_TABS, "tabela2_com_reducao.csv"), index=False, float_format="%.6f")
    print("\n=== Tabela 2 (com redução PCA, q=%d) ===" % q_escolhido)
    print(tab2[["detector", "acuracia", "sensibilidade", "especificidade",
                "precisao", "f1", "tempo_ms"]].to_string(index=False,
                float_format=lambda x: f"{x:.4f}"))

    # ---------- curva ROC do melhor detector ----------
    # melhor detector = maior F1 na Tabela 1 dentro do melhor domínio
    sub = tab1[tab1["dominio"] == melhor_dom]
    melhor_det = sub.loc[sub["f1"].idxmax(), "detector"]
    print(f"\nMelhor detector: {melhor_det} ({melhor_dom})")

    rng = np.random.default_rng(det.SEED)
    res, _ = det.uma_rodada(Xneg, Xpos, rng, com_pca_redutor=False)
    scores, y_pos = res[melhor_det][2], res[melhor_det][3]
    fpr, tpr, auc = curva_roc(scores, y_pos)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        os.path.join(det.DIR_TABS, "roc.csv"), index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color="#2a78d6", linewidth=2, label=f"{melhor_det} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#898781", linestyle="--", label="aleatório")
    ax.set_xlabel("taxa de falsos positivos (1 - especificidade)")
    ax.set_ylabel("taxa de verdadeiros positivos (sensibilidade)")
    ax.set_title(f"Curva ROC — detector {melhor_det} ({melhor_dom.lower()})")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(det.DIR_FIGS, "roc.png")); plt.close(fig)

    resumo = {"melhor_dominio": melhor_dom, "melhor_detector": melhor_det,
              "q_pca_detector": q_pca_detector, "q_pca_redutor": q_escolhido,
              "ve_q": float(ve[q_escolhido - 1]), "auc": float(auc),
              "percentil_limiar": det.PERCENTIL}
    pd.Series(resumo).to_json(os.path.join(det.DIR_TABS, "resumo.json"))
    print(f"\nAUC do melhor detector: {auc:.4f}")
    print("Resultados salvos em", det.DIR_TABS)


def curva_roc(scores, y_pos):
    ordem = np.argsort(-scores)
    y = y_pos[ordem]
    P = y.sum()
    Nn = (~y).sum()
    tpr = np.concatenate([[0], np.cumsum(y) / P])
    fpr = np.concatenate([[0], np.cumsum(~y) / Nn])
    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc


if __name__ == "__main__":
    main()
