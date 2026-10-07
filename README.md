# ecg-anomaly-detection

Detection of anomalous heartbeats in the ECG5000 dataset, trained only on normal beats (one-class). Features come from the time and frequency domains; the detectors are Euclidean distance, Mahalanobis distance and PCA reconstruction error, written in NumPy without scikit-learn.

## Usage

    pip install -r requirements.txt
    cd codigo && python executar.py   # full experiment
    python -m pytest                  # tests, from the repository root

Runs the whole experiment (feature extraction, rank, 100 Monte Carlo runs per configuration, PCA reduction and ROC curve) and rewrites `resultados/` in about 13 s. The seed is fixed, so two runs give the same numbers.

## Protocol

Training on 80% of the normal beats; testing on the other 20% plus every anomaly. The anomaly threshold is the 95th percentile of the training scores. Z-score normalization is fitted on the training set only. Metrics are the mean of 100 runs.

## Results

Time-domain features (20 features, rank 15):

| Detector | Accuracy | Sensitivity | F1 | Time |
|---|---|---|---|---|
| Euclidean | 0.719 | 0.654 | 0.784 | 0.36 ms |
| Mahalanobis | 0.908 | 0.897 | 0.939 | 3.75 ms |
| PCA (reconstruction) | 0.598 | 0.500 | 0.660 | 2.71 ms |

With PCA reduction from 20 to 9 components (96.8% of the variance) before the detector:

| Detector | Accuracy | F1 | Time |
|---|---|---|---|
| Mahalanobis | 0.916 | 0.944 | 1.67 ms |

AUC of the best detector: 0.962.

![ROC curve](resultados/figuras/roc.png)

Frequency-domain features stay at F1 ≤ 0.29 with any detector: the ECG5000 anomaly lives in the shape of the beat, and the power spectral density discards the phase that carries that shape.

## Notes

- Mahalanobis beats Euclidean because the features are correlated (rank 15 in 20 dimensions): the covariance weights each direction by its spread.
- An ill-conditioned covariance gets Tikhonov regularization with an adaptive λ: it starts at `1e-6·tr(C)/p` and grows tenfold until the condition number allows the inversion.
- Going from 20 to 9 components raised F1 (0.939 to 0.944) and cut 55% of the time: the dropped directions were noise and linear dependence.
- The same PCA fails as a detector (F1 0.660) and wins as a reducer in front of Mahalanobis.
- Distances are vectorized with `np.einsum`, with no per-sample loop.

## Layout

File and folder names are in Portuguese (course project at a Brazilian university):

    codigo/deteccao_anomalias.py   features, detectors, metrics and protocol
    codigo/executar.py             full experiment; saves tables and figures
    tests/                         pytest: detectors against independent references
    dados/ecg5000.csv              4998 beats × 140 samples (2919 normal)
    resultados/                    CSV tables, JSON summary and figures
    relatorio/                     report in LaTeX and PDF (Portuguese)
    notebook.ipynb                 notebook version with the step-by-step discussion

## Stack

Python, NumPy, SciPy, pandas, matplotlib.

Final project for Pattern Recognition at the Federal University of Ceará (UFC).
