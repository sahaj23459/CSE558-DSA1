# CSE558 Data Science Assignment (DSA1)

Sahaj Gupta, 2023459

Random projection (Johnson–Lindenstrauss) for k-means clustering and least-squares regression on the KDD Cup 1999 dataset.

- `DSA1.ipynb`: all code for Q1 and Q2, with outputs
- `DSA1.html`: the same notebook as a web page
- `figures/`: plots used in the report
- `results/`: numbers from every run, as CSV

## Run

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute DSA1.ipynb
```

All random seeds are fixed, so the reported numbers can be reproduced. Timings depend on the machine.
