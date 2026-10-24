# Benchmark Report: test_mathspeech.jsonl

- **Model:** `math-to-latex-v2`
- **Mode:** `MODEL_ONLY`
- **Evaluated:** 2026-10-15 14:55:48

## Overall Metrics

| Metric | Value |
| :--- | :--- |
| **Total Samples** | 20 |
| **Normalized Exact Match (Norm-EM)** | **10.00%** (2/20) |
| **Strict Exact Match (Strict-EM)** | **5.00%** (1/20) |
| **Token BLEU-4** | **26.94 / 100** |
| **Character Error Rate (CER)** | **66.60%** |
| **Token Error Rate (TER)** | **61.21%** |
| **Mean Latency** | **1295.1 ms** |
| **P95 Latency** | **3717.0 ms** |
| **Throughput** | **0.8 samples/sec** |

## Sample Discrepancies (Top 10)

| Input Spoken Math | Ground Truth Target | Model Prediction |
| :--- | :--- | :--- |
| `2x equals 2v over x squared` | `$2x=2v/x^{2}$` | `$\frac{2xequals2v}{xsquared}$` |
| `one half of ab times sine theta times tangent theta plus cosine theta d theta.` | `$\displaystyle \frac{1}{2}ab\sin \theta\tan \theta+\cos \theta\, d\theta$` | `$\frac{1} ^ 2 \ ab( \sqrt \sum ) + \int \theta d \pi$` |
| `pi over 3 sine x plus sine pi over 3 cosine x` | `$\displaystyle \frac{\pi}{3} \sin(x) + \sin\frac{\pi}{3} \cos(x)$` | `$\frac{pi}{3sinexplussinepiover3cosinex}$` |
| `one quarter plus one half cosine two theta plus one quarter cosine squared two theta` | `$\displaystyle 1/4+1/2\cos2\theta+1/4\cos^{2}2\theta$` | `$\frac{1} ^ 2 \ + \sqrt \cos 2 \sum \int (2 \pi )$` |
| `d by dx of tan inverse of x` | `$\frac{d}{dx}(tan^{-1}(x))$` | `$\frac{d} ^ dx \ ( \sqrt \sum ) \pi \int 1- \delta (x)$` |
| `g of x is equal to natural log of cosine of x` | `$g(x) = \ln(\cos(x))$` | `$g(x) = { ( \frac x)}$` |
| `two times y minus one minus two times z minus one equals zero` | `$2(y-1)-2(z-1)=0$` | `$2(y-1)-1 = 0$` |
| `dy is y sub u du plus y sub v dv` | `$\displaystyle \mathrm{d}y\,\ =y_u \mathrm{d}u + y_{v}\,\mathrm{d}v$` | `$dy = y _{u} du + y _ ^ v \frac, dv$` |
| `y prime is equal to minus 2xy divided by x squared` | `$y^{\prime}=-\frac{2xy}{x^{2}}$` | `$\frac{yprimeisequaltominus2xy}{xsquared}$` |
| `x times the sine of x` | `$x\mathinner{\sin\left(x\right)}$` | `$x- ^ (x)$` |
