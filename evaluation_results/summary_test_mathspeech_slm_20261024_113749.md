# Benchmark Report: test_mathspeech.jsonl

- **Model:** `math-to-latex-v2`
- **Mode:** `SLM`
- **Evaluated:** 2026-10-24 11:37:49

## Overall Metrics

| Metric | Value |
| :--- | :--- |
| **Total Samples** | 107 |
| **Normalized Exact Match (Norm-EM)** | **40.19%** (43/107) |
| **Strict Exact Match (Strict-EM)** | **4.67%** (5/107) |
| **Token BLEU-4** | **47.79 / 100** |
| **Character Error Rate (CER)** | **47.01%** |
| **Token Error Rate (TER)** | **30.75%** |
| **Mean Latency** | **519.8 ms** |
| **P95 Latency** | **969.2 ms** |
| **Throughput** | **1.9 samples/sec** |

## Sample Discrepancies (Top 10)

| Input Spoken Math | Ground Truth Target | Model Prediction |
| :--- | :--- | :--- |
| `2x equals 2v over x squared` | `$2x=2v/x^{2}$` | `$2x = \frac{2V}{x^2}$` |
| `one half of ab times sine theta times tangent theta plus cosine theta d theta.` | `$\displaystyle \frac{1}{2}ab\sin \theta\tan \theta+\cos \theta\, d\theta$` | `$1/2ab \cdot \sin\theta \cdot \tan\theta + \cos\theta \, d\theta$` |
| `pi over 3 sine x plus sine pi over 3 cosine x` | `$\displaystyle \frac{\pi}{3} \sin(x) + \sin\frac{\pi}{3} \cos(x)$` | `$\frac{\pi}{3} \sin{x} + \sin\left(\frac{\pi}{3}\right) \cos{x}$` |
| `one quarter plus one half cosine two theta plus one quarter cosine squared two theta` | `$\displaystyle 1/4+1/2\cos2\theta+1/4\cos^{2}2\theta$` | `$\frac{1}{4} + \frac{1}{2}\cos(2\theta) + \frac{1}{4}\cos^2(2\theta)$` |
| `d by dx of tan inverse of x` | `$\frac{d}{dx}(tan^{-1}(x))$` | `$\frac{d}{dx} \tan^{-1}(x)$` |
| `two times y minus one minus two times z minus one equals zero` | `$2(y-1)-2(z-1)=0$` | `$2y - 1 - 2z - 1 = 0$` |
| `dy is y sub u du plus y sub v dv` | `$\displaystyle \mathrm{d}y\,\ =y_u \mathrm{d}u + y_{v}\,\mathrm{d}v$` | `$\frac{d}{dx}y = \frac{\partial y}{\partial x}u + \frac{\partial y}{\partial x}v$` |
| `M dx plus N dy` | `$M\,dx+N\,dy$` | `$dx + dy$` |
| `y prime is equal to minus 2xy divided by x squared` | `$y^{\prime}=-\frac{2xy}{x^{2}}$` | `$y' = -\frac{2xy}{x^2}$` |
| `x times the sine of x` | `$x\mathinner{\sin\left(x\right)}$` | `$x \times \sin(x)$` |
