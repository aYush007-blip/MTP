# Benchmark Report: manifest.csv

- **Model:** `math-to-latex-v2`
- **Mode:** `SLM`
- **Evaluated:** 2026-10-24 11:38:40

## Overall Metrics

| Metric | Value |
| :--- | :--- |
| **Total Samples** | 50 |
| **Normalized Exact Match (Norm-EM)** | **48.00%** (24/50) |
| **Strict Exact Match (Strict-EM)** | **28.00%** (14/50) |
| **Token BLEU-4** | **60.29 / 100** |
| **Character Error Rate (CER)** | **29.62%** |
| **Token Error Rate (TER)** | **27.08%** |
| **Mean Latency** | **566.2 ms** |
| **P95 Latency** | **1023.9 ms** |
| **Throughput** | **1.8 samples/sec** |

## Sample Discrepancies (Top 10)

| Input Spoken Math | Ground Truth Target | Model Prediction |
| :--- | :--- | :--- |
| `x minus two y minus one and z plus one dot product with one, five, ten` | `$[x-2,y-1,z+1]\cdot[1,5,10]$` | `$x - 2y + z \cdot (1, 5, 10)$` |
| `x minus two plus five times y minus one plus ten times z plus one equals zero` | `$(x - 2) + 5(y - 1) + 10(z + 1) = 0$` | `$x - 2 + 5y - 1 + 10z + 1 = 0$` |
| `Ax plus By plus Cz equals D` | `$Ax + By + Cz = D$` | `$ax + by + cz = d$` |
| `A inverse is one over determinant of A times the adjoint matrix` | `$A^{-1} = \frac{1}{\text{det}(A)} \cdot \text{adj}(A)$` | `$A^{-1} = \frac{1}{\det(A)} \cdot \text{adj}(A)$` |
| `X equals A inverse B` | `$X = A^{-1}B$` | `$X = \frac{1}{B}$` |
| `a half times one plus two thirds plus a half times a half` | `$1/2\cdot1+2/3+1/2\cdot1/2$` | `$a \frac{1}{2} \times 1 + \frac{2}{3} + \frac{1}{2} \times \frac{1}{2}$` |
| `a half plus n minus 1 plus a half` | `$\frac{1}{2}+n-1+\frac{1}{2}$` | `$a \frac{1}{2} + n - 1 + \frac{1}{2}$` |
| `square root of pi over 2` | `$\frac{\sqrt{\pi}}{2}$` | `$\sqrt{\frac{\pi}{2}}$` |
| `e to the minus b squared plus x squared` | `$e^{-(b^2 + x^2)}$` | `$e^{-b^2 + x^2}$` |
| `e to the minus b squared times e to the minus x squared` | `$e^{-b^2} e^{-x^2}$` | `$e^{-b^2} \times e^{-x^2}$` |
