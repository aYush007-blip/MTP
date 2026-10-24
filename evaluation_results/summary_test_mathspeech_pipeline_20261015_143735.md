# Benchmark Report: test_mathspeech.jsonl

- **Model:** `math-to-latex-v2`
- **Mode:** `PIPELINE`
- **Evaluated:** 2026-10-15 14:37:35

## Overall Metrics

| Metric | Value |
| :--- | :--- |
| **Total Samples** | 107 |
| **Normalized Exact Match (Norm-EM)** | **8.41%** (9/107) |
| **Strict Exact Match (Strict-EM)** | **0.93%** (1/107) |
| **Token BLEU-4** | **27.06 / 100** |
| **Character Error Rate (CER)** | **72.74%** |
| **Token Error Rate (TER)** | **61.15%** |
| **Mean Latency** | **787.4 ms** |
| **P95 Latency** | **2125.1 ms** |
| **Throughput** | **1.3 samples/sec** |

## Sample Discrepancies (Top 10)

| Input Spoken Math | Ground Truth Target | Model Prediction |
| :--- | :--- | :--- |
| `2x equals 2v over x squared` | `$2x=2v/x^{2}$` | `$\frac{2xequals2v}{xsquared}$` |
| `one half of ab times sine theta times tangent theta plus cosine theta d theta.` | `$\displaystyle \frac{1}{2}ab\sin \theta\tan \theta+\cos \theta\, d\theta$` | `$\left({\right)}$` |
| `pi over 3 sine x plus sine pi over 3 cosine x` | `$\displaystyle \frac{\pi}{3} \sin(x) + \sin\frac{\pi}{3} \cos(x)$` | `$\frac{pi}{3sinexplussinepiover3}$` |
| `one quarter plus one half cosine two theta plus one quarter cosine squared two theta` | `$\displaystyle 1/4+1/2\cos2\theta+1/4\cos^{2}2\theta$` | `$\frac{1} 2 + 1 \frac{2} ^$` |
| `d by dx of tan inverse of x` | `$\frac{d}{dx}(tan^{-1}(x))$` | `$\frac{d} ^ dx \ ( \sqrt \sum ) \pi \int 1- \delta (x)$` |
| `g of x is equal to natural log of cosine of x` | `$g(x) = \ln(\cos(x))$` | `$g(x) = \frac \left(x\right)$` |
| `two times y minus one minus two times z minus one equals zero` | `$2(y-1)-2(z-1)=0$` | `$2(y-1)-1 = 0$` |
| `dy is y sub u du plus y sub v dv` | `$\displaystyle \mathrm{d}y\,\ =y_u \mathrm{d}u + y_{v}\,\mathrm{d}v$` | `$+ y _{v}$` |
| `y prime is equal to minus 2xy divided by x squared` | `$y^{\prime}=-\frac{2xy}{x^{2}}$` | `$\frac{isequaltominus2xy}{xsquared}$` |
| `x times the sine of x` | `$x\mathinner{\sin\left(x\right)}$` | `$x- ^ (x)$` |
