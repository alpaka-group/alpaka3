# Tensor Collective Demo

This example demonstrates Alpaka's experimental NCCL collective provider.

- Configures a single-rank `CleanTensorOpContext` using the first CUDA device.
- Runs an in-place NCCL all-reduce to verify the provider wiring.
- Falls back cleanly when CUDA or NCCL is missing.
- Needs external process bootstrap (e.g., MPI) before it can scale to multiple ranks or nodes.

## Getting NCCL Running

1. **Detect NCCL at configure time**
    - Export `NCCL_ROOT`, `NCCL_PATH`, or extend `CMAKE_PREFIX_PATH` so CMake can locate the headers and library (`libnccl.so`).
    - Clear the CMake cache or reconfigure from a clean build directory.
    - Run `cmake -S .. -B build -Dalpaka_ENABLE_COLLECTIVES=ON -Dalpaka_ENABLE_NCCL=ON` and confirm the log prints the resolved NCCL include and library paths.

2. **Propagate headers and libraries to targets**
    - After configuration, rebuild `tensorCollectiveDemo` with `cmake --build build --target tensorCollectiveDemo VERBOSE=1`.
    - Check the `nvcc` command line for `-DALPAKA_HAS_NCCL` and the NCCL include directory; verify the link step lists `libnccl.so`.

3. **Ensure runtime visibility**
    - Add the NCCL library directory to `LD_LIBRARY_PATH` or rely on the rpath set by CMake so `libnccl.so` loads successfully.
    - Set environment tuning knobs (e.g., `NCCL_P2P_DISABLE=0`, `NCCL_SOCKET_IFNAME`) only if your fabric requires them.

4. **Scale beyond a single rank**
    - Provide an external bootstrap mechanism (MPI, TCP, etc.) to exchange NCCL unique IDs across processes before calling `configureCollectives`.
    - Launch one process per GPU, populate `GroupConfig.deviceIds`, `worldRank`, and `worldSize` for each rank, and rerun the demo to exercise multi-GPU communicators.
