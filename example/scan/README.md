# Scan

This example implements inclusive and exclusive scans.

## Inclusive Scan

Inclusive scan, or [prefix sum](https://en.wikipedia.org/wiki/Prefix_sum), converts an input vector $x$ to an output vector $y$, where $y_n = \sum_{i=0}^{n}x_i$, i.e., the sum of all entries in $x$ from $0$ to $n$.

## Exclusive Scan

Exclusive scan (or "all-prefix-sums") is the same as inclusive scan, but excludes the $n$-th index: $y_n = \sum_{i=0}^{n-1}x_i$.
