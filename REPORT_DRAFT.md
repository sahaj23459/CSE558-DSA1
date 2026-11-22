# CSE558 Assignment: JL for k-means and linear regression

Name: Sahaj Gupta, 2023459

AI assistance: Claude (AI assistant) was used for help with the code and explanations.

Code: https://github.com/sahaj23459/CSE558-DSA1

All code is in `DSA1.ipynb`. Seeds are fixed (`SEED = 558` and offsets from it). Timings were measured on one machine (10-core laptop) and will differ on another.

## Data

Starting from 41 features, one-hot encoding gives 118 columns. Two of them are constant and get dropped: `num_outbound_cmds` and `is_host_login`.

**n = 494,021 rows, d = 116 columns.** rank(D) = 113. The rank is 3 less than d because some columns are exact combinations of others. Each of the three one-hot groups adds up to 1 in every row, so the three group sums are equal; that gives two dependencies. The third is that `protocol_type_icmp` equals the sum of the ICMP-only services (`ecr_i`, `eco_i`, `tim_i`, …). 80.3% of the connections are attacks.

---

## Q1: JL for k-means (k = 15, KMeans with n_init = 5)

### (a)

**cost(D, B) = 55,508.13.** sklearn's inertia is 55,505.83; the small gap is float32 rounding.
Wall-clock time of k-means on D: **2.63 s**.

### (b) and (c): cost ratios relative to cost(D, B), over 5 draws of A for each m

| m | B̂ mean | B̂ min | B̂ max | B⁺ mean | B⁺ min | B⁺ max |
|---|---|---|---|---|---|---|
| 5  | 1.122 | 1.010 | 1.295 | 76.69 | 73.77 | 78.25 |
| 10 | 1.057 | 0.974 | 1.138 | 73.53 | 71.57 | 75.95 |
| 15 | 0.993 | 0.969 | 1.025 | 69.82 | 66.91 | 72.00 |
| 20 | 1.012 | 0.968 | 1.072 | 66.03 | 64.89 | 67.75 |
| 25 | 1.007 | 0.973 | 1.109 | 62.92 | 60.43 | 67.17 |

cost(D, B̂) and cost(D, B⁺) for every run are in Appendix A.

![Q1(b, c) cost ratios](figures/q1_bc_ratios.png)

**(b)**
The ratio gets closer to 1 as m grows. At m = 5 the mean is 1.12 and the worst run is 1.29. From m = 15 onwards the mean is about 1.0. The gap between the best and worst run also gets smaller. Even with only 5 dimensions, the clustering is not much worse than the one on all 116 columns. This fits the JL result for k-means: the number of dimensions needed depends on k and the error we accept, not on n or d.

Some ratios are below 1. This means the clusters found on E gave a lower cost on D than k-means on D itself. k-means only finds a local minimum, so cost(D, B) is not the best possible cost. The run on E starts from different points in a different space, and sometimes it lands in a better local minimum. So a ratio below 1 says that the run in (a) was not optimal.

**(c)**
I would use B̂. On average its cost is within 12% of the full k-means, and the worst run is within 30%. B⁺ is 60 to 77 times worse. The problem is that A⁺a always lies in the row space of A. That is only an m-dimensional subspace of ℝᵈ. Every part of the true centre outside that subspace is lost, so the centres end up far from the data. B̂ keeps the clusters found on E but averages the original rows of D. That gives a real centre in ℝᵈ. It costs only one extra pass over D.

### (d) Mean time to draw A, compute E = DAᵀ and run k-means on E

| m | 5 | 10 | 15 | 20 | 25 | k-means on D (a) |
|---|---|---|---|---|---|---|
| time (s) | 1.07 | 1.14 | 1.27 | 1.30 | 1.43 | 2.63 |

![Q1(d) running time against m](figures/q1_d_time.png)

The time goes up slowly with m. Each k-means step costs more when the points have more coordinates. At every m it is faster than running on D, about 2.5 times faster at m = 5. The speed-up is smaller than 116/m would suggest. Some costs do not depend on the dimension, such as the k-means++ start and the 5 restarts. Computing E also takes time.

### (e) 10⁴ pairs (the same for all m) and 20 draws of A

| m | pairs: mean | pairs: std | cost ratio: mean | cost ratio: std | cost ratio: min–max |
|---|---|---|---|---|---|
| 5  | 0.937 | 0.495 | 0.974 | 0.093 | 0.81–1.21 |
| 10 | 1.025 | 0.304 | 1.004 | 0.109 | 0.86–1.33 |
| 15 | 0.814 | 0.328 | 1.031 | 0.092 | 0.85–1.20 |
| 20 | 0.900 | 0.277 | 1.008 | 0.078 | 0.89–1.23 |
| 25 | 1.087 | 0.232 | 1.025 | 0.060 | 0.91–1.15 |

![Q1(e) box plots](figures/q1_e_boxplots.png)

The cost ratio in (ii) has the smaller spread. Its standard deviation is 3 to 5 times smaller than that of the pairwise ratios in (i) at every m.

A single pair can be stretched or shrunk a lot. For one pair, the ratio behaves like a chi-squared variable with m degrees of freedom, divided by m. Its standard deviation is √(2/m), which is 0.63 at m = 5 and 0.28 at m = 25. The values we measured are close to this. At m = 5 some pairs are stretched by more than 3 times.

The k-means cost is a sum over all 494,021 points. Some distances grow and some shrink, and these errors mostly cancel out. So the total stays close to its true value, and the ratio stays near 1.

Both spreads get smaller as m grows. The pairwise spread falls roughly like 1/√m, from 0.50 at m = 5 to 0.23 at m = 25. The cost-ratio spread is already small and falls more gently, from about 0.09–0.11 for m ≤ 15 to 0.06 at m = 25. In (i) the mean also moves around, for example 0.81 at m = 15. This happens because all 10⁴ pairs use the same A. Many pairs point in similar directions because the data has many repeated attack rows, so they are stretched or shrunk together.

---

## Q2: JL for linear regression (CountSketch)

### (a)

**Loss(D, y, b) = 2,949.79.** Time to compute b with `numpy.linalg.lstsq`: **1.30 s**.

### (b) and (c): 5 CountSketches for each m

| m | mean Loss(D,y,ã) | mean ratio | min ratio | max ratio | mean sketched loss ‖SDã − Sy‖² | mean time (s) |
|---|---|---|---|---|---|---|
| 50   | 10,191 | 3.455 | 3.026 | 4.508 | ≈ 0 (9e-24) | 0.314 |
| 100  | 48,575 | 16.47 | 10.91 | 24.93 | ≈ 0 (2e-23) | 0.327 |
| 200  | 6,267  | 2.125 | 1.981 | 2.246 | 1,406.9 | 0.313 |
| 500  | 3,830  | 1.298 | 1.244 | 1.360 | 2,296.1 | 0.314 |
| 1000 | 3,315  | 1.124 | 1.103 | 1.162 | 2,533.4 | 0.316 |
| 2000 | 3,132  | 1.062 | 1.051 | 1.067 | 2,847.6 | 0.318 |
| 5000 | 3,022  | 1.024 | 1.020 | 1.029 | 2,887.4 | 0.322 |

Results for every run are in Appendix B.

![Q2(b) loss ratio against m, log-log](figures/q2_b_ratio.png)

**(b)** Here rank(D) = 113.

Below the rank (m = 50 and 100), SD has fewer rows than the rank of D. So there is always an ã that fits the sketched data exactly, and the sketched loss is 0 up to rounding. But this ã is useless on the full data. The true loss is 3.5 to 16 times the best loss. It is worst at m = 100, just below the rank. There SD is almost square and badly conditioned, so the solution has very large entries.

Above the rank, the sketched problem has more equations than unknowns, and ã becomes a good estimate. The ratio drops quickly towards 1: 2.12 at m = 200, 1.30 at m = 500 and 1.02 at m = 5000. The numbers follow roughly 1 + 113/(m − 113). For m = 500 this gives 1.29, and we got 1.30.

On average the sketched loss is smaller than the true best loss of 2,950, because ã is chosen to fit the sketched data. It is not a strict bound: one run at m = 5000 gave 3,014. As m grows the average rises towards 2,950. At m = 500 it is 2,296, and at m = 5000 it is 2,887. So a small sketch makes the fit look better than it really is.

**(c)**
Drawing S, forming SD and Sy and solving for ã takes about 0.31 to 0.32 s for every m. Computing b on the full data takes 1.30 s, so the sketch is about 4 times faster. The time barely changes with m. Most of it is the single pass over the 494,021 rows to build SD. After that, the small m × 116 problem is quick to solve, even at m = 5000.

### (d) ‖Sv‖²/‖v‖² over 100 CountSketches of the residual v = Db − y

| m | mean | std | min | max | √(2/m) |
|---|---|---|---|---|---|
| 50   | 0.993 | 0.186 | 0.621 | 1.597 | 0.200 |
| 500  | 1.004 | 0.062 | 0.875 | 1.198 | 0.063 |
| 5000 | 1.001 | 0.020 | 0.955 | 1.039 | 0.020 |

![Q2(d) histograms](figures/q2_d_hist.png)

The mean is about 1 for every m, because CountSketch keeps the squared length correct on average. The spread shrinks as m grows. The standard deviation is 0.186 at m = 50, 0.062 at m = 500 and 0.020 at m = 5000. Each 10 times increase in m cuts it by about √10 ≈ 3.2. This matches √(2/m). At m = 50 the ratio ranges from 0.62 to 1.60. At m = 5000 it stays between 0.96 and 1.04.

### (e) 50,000-row subset, m = 500, 5 repetitions

Loss(D′, y′, b′) = 203.16.

| sketch | mean time to draw S and form SD′, Sy′ (s) | mean ratio | min–max ratio |
|---|---|---|---|
| CountSketch | 0.030 | 1.295 | 1.267–1.314 |
| Dense Gaussian | 0.112 | 1.287 | 1.248–1.335 |

Both sketches give almost the same accuracy, with a ratio of about 1.29. CountSketch is about 4 times faster on this subset. On the full D the difference would be much larger. A dense Gaussian S would have 500 × 494,021 ≈ 247 million entries, about 2 GB in float64. Computing SD would take about 3 × 10¹⁰ operations. CountSketch only needs one bucket and one sign per row, and one pass over D. So I would use CountSketch on the full D. It is just as accurate and much cheaper in time and memory.

---

## Appendix A: Q1(b, c, d), every run

| m | run | cost(D, B̂) | cost(D, B̂)/cost(D, B) | cost(D, B⁺) | cost(D, B⁺)/cost(D, B) | time (s) |
|---|---|---|---|---|---|---|
| 5 | 1 | 58,541.18 | 1.0546 | 4,094,781 | 73.77 | 1.10 |
| 5 | 2 | 56,047.11 | 1.0097 | 4,343,497 | 78.25 | 1.10 |
| 5 | 3 | 60,962.37 | 1.0983 | 4,296,939 | 77.41 | 1.03 |
| 5 | 4 | 71,882.24 | 1.2950 | 4,206,642 | 75.78 | 1.05 |
| 5 | 5 | 64,026.98 | 1.1535 | 4,341,917 | 78.22 | 1.07 |
| 10 | 1 | 55,260.80 | 0.9955 | 4,116,625 | 74.16 | 1.10 |
| 10 | 2 | 58,866.65 | 1.0605 | 3,972,908 | 71.57 | 1.21 |
| 10 | 3 | 54,057.40 | 0.9739 | 3,988,064 | 71.85 | 1.08 |
| 10 | 4 | 63,144.98 | 1.1376 | 4,113,782 | 74.11 | 1.15 |
| 10 | 5 | 61,921.46 | 1.1155 | 4,215,719 | 75.95 | 1.13 |
| 15 | 1 | 54,133.16 | 0.9752 | 3,975,860 | 71.63 | 1.18 |
| 15 | 2 | 53,762.37 | 0.9685 | 3,996,766 | 72.00 | 1.26 |
| 15 | 3 | 54,264.91 | 0.9776 | 3,738,079 | 67.34 | 1.37 |
| 15 | 4 | 56,870.58 | 1.0245 | 3,952,196 | 71.20 | 1.22 |
| 15 | 5 | 56,429.79 | 1.0166 | 3,713,874 | 66.91 | 1.33 |
| 20 | 1 | 56,784.88 | 1.0230 | 3,601,913 | 64.89 | 1.37 |
| 20 | 2 | 54,374.54 | 0.9796 | 3,760,910 | 67.75 | 1.29 |
| 20 | 3 | 53,728.90 | 0.9679 | 3,619,904 | 65.21 | 1.29 |
| 20 | 4 | 59,522.58 | 1.0723 | 3,659,912 | 65.93 | 1.32 |
| 20 | 5 | 56,357.50 | 1.0153 | 3,683,926 | 66.37 | 1.24 |
| 25 | 1 | 54,698.82 | 0.9854 | 3,468,022 | 62.48 | 1.41 |
| 25 | 2 | 54,855.83 | 0.9882 | 3,728,475 | 67.17 | 1.52 |
| 25 | 3 | 54,298.26 | 0.9782 | 3,460,081 | 62.33 | 1.37 |
| 25 | 4 | 61,548.17 | 1.1088 | 3,354,197 | 60.43 | 1.42 |
| 25 | 5 | 53,996.07 | 0.9728 | 3,452,221 | 62.19 | 1.41 |

## Appendix B: Q2(b, c), every run

| m | run | Loss(D, y, ã) | ratio to Loss(D, y, b) | sketched loss ‖SDã − Sy‖² | time (s) |
|---|---|---|---|---|---|
| 50 | 1 | 13,297.82 | 4.5081 | 9.7e-25 | 0.333 |
| 50 | 2 | 9,809.26 | 3.3254 | 1.5e-23 | 0.309 |
| 50 | 3 | 8,925.78 | 3.0259 | 1.8e-23 | 0.309 |
| 50 | 4 | 9,244.55 | 3.1340 | 7.8e-24 | 0.310 |
| 50 | 5 | 9,679.43 | 3.2814 | 1.7e-24 | 0.309 |
| 100 | 1 | 73,532.08 | 24.9279 | 4.1e-23 | 0.351 |
| 100 | 2 | 42,907.71 | 14.5460 | 4.1e-23 | 0.315 |
| 100 | 3 | 32,168.95 | 10.9055 | 1.9e-23 | 0.312 |
| 100 | 4 | 42,264.33 | 14.3279 | 5.7e-24 | 0.312 |
| 100 | 5 | 52,000.62 | 17.6286 | 1.6e-23 | 0.344 |
| 200 | 1 | 6,624.68 | 2.2458 | 1,365.88 | 0.318 |
| 200 | 2 | 5,995.91 | 2.0327 | 1,657.43 | 0.311 |
| 200 | 3 | 5,842.23 | 1.9806 | 1,385.52 | 0.312 |
| 200 | 4 | 6,383.28 | 2.1640 | 1,270.71 | 0.313 |
| 200 | 5 | 6,488.66 | 2.1997 | 1,354.89 | 0.312 |
| 500 | 1 | 3,856.69 | 1.3074 | 2,353.43 | 0.314 |
| 500 | 2 | 3,669.94 | 1.2441 | 2,277.81 | 0.313 |
| 500 | 3 | 3,811.82 | 1.2922 | 2,386.98 | 0.314 |
| 500 | 4 | 4,011.21 | 1.3598 | 2,438.07 | 0.315 |
| 500 | 5 | 3,800.87 | 1.2885 | 2,023.97 | 0.314 |
| 1000 | 1 | 3,280.21 | 1.1120 | 2,472.79 | 0.315 |
| 1000 | 2 | 3,286.37 | 1.1141 | 2,544.43 | 0.316 |
| 1000 | 3 | 3,253.74 | 1.1030 | 2,661.61 | 0.315 |
| 1000 | 4 | 3,328.69 | 1.1284 | 2,443.56 | 0.315 |
| 1000 | 5 | 3,427.49 | 1.1619 | 2,544.68 | 0.316 |
| 2000 | 1 | 3,148.83 | 1.0675 | 2,876.77 | 0.320 |
| 2000 | 2 | 3,145.50 | 1.0663 | 2,880.10 | 0.318 |
| 2000 | 3 | 3,116.53 | 1.0565 | 2,892.34 | 0.317 |
| 2000 | 4 | 3,147.20 | 1.0669 | 2,858.11 | 0.317 |
| 2000 | 5 | 3,099.52 | 1.0508 | 2,730.65 | 0.318 |
| 5000 | 1 | 3,007.54 | 1.0196 | 2,883.89 | 0.323 |
| 5000 | 2 | 3,035.93 | 1.0292 | 2,827.86 | 0.319 |
| 5000 | 3 | 3,024.20 | 1.0252 | 3,013.85 | 0.322 |
| 5000 | 4 | 3,016.92 | 1.0228 | 2,862.34 | 0.324 |
| 5000 | 5 | 3,023.97 | 1.0251 | 2,849.10 | 0.323 |
