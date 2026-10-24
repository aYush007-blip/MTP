# Benchmark Report: manifest.csv

- **Model:** `math-to-latex-v2`
- **Mode:** `PIPELINE`
- **Evaluated:** 2026-10-15 14:52:27

## Overall Metrics

| Metric | Value |
| :--- | :--- |
| **Total Samples** | 1101 |
| **Normalized Exact Match (Norm-EM)** | **6.81%** (75/1101) |
| **Strict Exact Match (Strict-EM)** | **0.91%** (10/1101) |
| **Token BLEU-4** | **25.64 / 100** |
| **Character Error Rate (CER)** | **74.43%** |
| **Token Error Rate (TER)** | **61.37%** |
| **Mean Latency** | **738.9 ms** |
| **P95 Latency** | **1946.0 ms** |
| **Throughput** | **1.4 samples/sec** |

## Sample Discrepancies (Top 10)

| Input Spoken Math | Ground Truth Target | Model Prediction |
| :--- | :--- | :--- |
| `ax plus by plus cz equals d` | `$ax + by + cz = d$` | `$+ {} -cz = d$` |
| `x minus two y minus one and z plus one dot product with one, five, ten` | `$[x-2,y-1,z+1]\cdot[1,5,10]$` | `$x-2y-1 + z + 1$` |
| `x minus two plus five times y minus one plus ten times z plus one equals zero` | `$(x - 2) + 5(y - 1) + 10(z + 1) = 0$` | `$x- 2 + 5y-1 + 10z + 1 = 0$` |
| `Ax plus By plus Cz equals D` | `$Ax + By + Cz = D$` | `$+ By + Cz = D$` |
| `A inverse is one over determinant of A times the adjoint matrix` | `$A^{-1} = \frac{1}{\text{det}(A)} \cdot \text{adj}(A)$` | `$\frac{Ainverseis1}{determinantofAtimestheadjointmatrix}$` |
| `X equals A inverse B` | `$X = A^{-1}B$` | `$X = }{$` |
| `log two minus log one` | `$\log2-\log1$` | `$\frac 2} - { 1$` |
| `b minus a over n` | `$\frac{b - a}{n}$` | `$\frac{bminusa}{n}$` |
| `a half times one plus two thirds plus a half times a half` | `$1/2\cdot1+2/3+1/2\cdot1/2$` | `$( (1 + 2) ^{3} + \frac \ 1 \sqrt \sum 2 \pi$` |
| `y0 plus 4y1 plus y2` | `$\displaystyle y_0 + 4y_1 + y_2$` | `$+ 4y1$` |
