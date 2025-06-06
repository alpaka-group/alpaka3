# Scan

This example implements inclusive and exclusive scans. For details on the algorithm used, please refer to the relevant [CUDA blog entry](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).

## Inclusive Scan

Inclusive scan, or [prefix sum](https://en.wikipedia.org/wiki/Prefix_sum), converts an input vector $x$ to an output vector $y$, where $y_n = \sum_{i=0}^{n}x_i$, i.e., the sum of all entries in $x$ from $0$ to $n$.

To run, call the executable with `-t 0`:
```sh
$ ./scan -t 0
```

## Exclusive Scan

Exclusive scan (or "all-prefix-sums") is the same as inclusive scan, but excludes the $n$-th index: $y_n = \sum_{i=0}^{n-1}x_i$.

To run, call the executable with `-t 1`:
```sh
$ ./scan -t 1
```

## Running

To run this example use the following:
```sh
$ mkdir build && cd build
$ ccmake .. # configure your backends, configure and generate (make sure to enable examples)
$ cmake --build . --target=scan -j
$ ./example/scan/scan
```

The example is also added as a test, where it runs 2^24 elements and inclusive scan by default. To change the number of elements scanned, use `-n N`. To change between inclusive/exclusive scan, use `-t 0` or `-t 1`.
