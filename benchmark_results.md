# Julia CUDA vs C/C++ CUDA Benchmark Comparison

## Benchmarks Tested
- **backprop**: Neural network backpropagation training (2 kernels)
- **nn**: K-nearest neighbor search using Euclidean distance (1 kernel)

## Test Environment
- **GPU**: NVIDIA A100-SXM4-40GB (8x available)
- **CUDA**: 12.8 (runtime 13.0)
- **Julia**: 1.12.3 with CUDA.jl 5.9.5
- **C++ Compiler**: gcc with nvcc

---

## Results Summary

### GPU Kernel Execution Times (nsys profiler)

| Benchmark | Kernel | C/C++ CUDA | Julia CUDA | Winner |
|-----------|--------|------------|------------|--------|
| backprop | bpnn_layerforward_CUDA | 11.1 µs | 15.8 µs | C++ 1.4x faster |
| backprop | bpnn_adjust_weights_cuda | 10.3 µs | 11.3 µs | C++ 1.1x faster |
| backprop | **Total** | **21.5 µs** | **27.1 µs** | **C++ 1.26x faster** |
| nn | euclid | **5.3 µs** | **10.7 µs** | **C++ 2.0x faster** |

### End-to-End Execution Times

| Benchmark | C/C++ CUDA | Julia CUDA (cold) | Julia CUDA (warm) |
|-----------|------------|-------------------|-------------------|
| backprop | ~0.94 sec | ~37 sec | ~0.03 sec |
| nn | ~0.73 sec | ~37 sec | ~0.28 sec |

**Note**: Julia "cold" includes JIT compilation (~37 sec). Julia "warm" is after JIT within same session.

---

## Compilation Gotchas

### 1. CUDA Path Configuration

The default Makefile expects nvcc at `/usr/bin/nvcc`. Required creating `Make.user`:

```bash
# cuda/Make.user
CUDA_ROOT := /usr/local/cuda
```

### 2. Julia Random API Breaking Change

Julia 1.12 removed `Random.srand`. The original code in `common/julia/crand.jl`:

```julia
# BROKEN in Julia 1.12+
using Random
import Random: rand, srand  # srand no longer exists
```

**Fix applied:**
```julia
# WORKING
using Random
import Random: rand
# srand is defined locally, not imported from Random
```

### 3. NN Data File Path Issues

The generated data files use relative paths that assume execution from a specific directory:

```bash
# list640k_64.txt contains:
../../data/nn/cane640k_64_0.db

# Fix: Create local path version
sed 's|../../data/nn/||g' list640k_64.txt > list640k_64_local.txt
```

### 4. nvprof Not Supported on A100

```
======== Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
```

**Solution**: Use `nsys` (Nsight Systems) instead:
```bash
nsys profile --stats=true -f true -o /tmp/output ./program
```

---

## Code Comparison

### Backprop Kernel: Shared Memory Reduction

**C++ CUDA** (`backprop_cuda_kernel.cu`):
```cuda
__shared__ float input_node[HEIGHT];
__shared__ float weight_matrix[HEIGHT][WIDTH];

if (tx == 0)
    input_node[ty] = input_cuda[index_in];

__syncthreads();

weight_matrix[ty][tx] = input_hidden_cuda[index];

__syncthreads();

weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

__syncthreads();

for (int power_two = 2; power_two <= HEIGHT; power_two *= 2) {
    if (ty % power_two == 0)
        weight_matrix[ty][tx] =
            weight_matrix[ty][tx] + weight_matrix[ty + power_two / 2][tx];
    __syncthreads();
}
```

**Julia CUDA** (`backprop_cuda_kernel.jl`):
```julia
input_node = @cuStaticSharedMem(Float32, HEIGHT)
weight_matrix = @cuStaticSharedMem(Float32, (HEIGHT, WIDTH))

if tx == 1
    @inbounds input_node[ty] = input_cuda[index_in + 1]
end

sync_threads()

@inbounds weight_matrix[tx, ty] = input_hidden_cuda[index + 1]

sync_threads()

weight_matrix[tx, ty] *= input_node[ty]

sync_threads()

power_two = 2
while power_two <= HEIGHT
    if ty % power_two == 1
        @inbounds weight_matrix[tx, ty] += weight_matrix[tx, ty + power_two ÷ 2]
    end
    power_two *= 2
    sync_threads()
end
```

**Key Differences**:
- Julia uses 1-based indexing (requires `+1` adjustments)
- Julia uses `@cuStaticSharedMem` macro vs `__shared__` declaration
- Julia array indexing is `[tx, ty]` vs C's `[ty][tx]` (column-major vs row-major)
- Julia uses `sync_threads()` vs `__syncthreads()`

### NN Kernel: Simple Embarrassingly Parallel

**C++ CUDA** (`nn.cu`):
```cuda
__global__ void euclid(LatLong *d_locations, float *d_distances,
                       int numRecords, float lat, float lng) {
    int globalId = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
    LatLong *latLong = d_locations + globalId;
    if (globalId < numRecords) {
        float *dist = d_distances + globalId;
        *dist = (float)sqrt((lat - latLong->lat) * (lat - latLong->lat) +
                            (lng - latLong->lng) * (lng - latLong->lng));
    }
}
```

**Julia CUDA** (`nn.jl`):
```julia
function euclid(d_locations, d_distances, numRecords, lat, lng)
    globalId = threadIdx().x + blockDim().x *
                               (gridDim().x * (blockIdx().y - 1) + (blockIdx().x - 1))
    if globalId <= numRecords
        latLong = d_locations[globalId]
        d_distances[globalId] =
            CUDA.sqrt((lat - latLong.lat) * (lat - latLong.lat) +
                      (lng - latLong.lng) * (lng - latLong.lng))
    end
    return
end
```

**Key Differences**:
- Julia requires `CUDA.sqrt` instead of just `sqrt`
- 1-based indexing: `(blockIdx().y - 1)` vs `blockIdx.y`
- Julia accesses struct fields directly vs pointer arithmetic
- Julia requires explicit `return` statement

---

## Observations

### 1. Kernel Performance Favors C++

Both benchmarks show C++ kernels executing faster:
- **backprop**: 1.26x faster in C++
- **nn**: 2.0x faster in C++

The simpler nn kernel shows a larger relative difference, suggesting Julia's abstraction overhead is more significant for lightweight kernels.

### 2. Host Overhead Dominates Cold-Start

The ~0.9 sec C++ runtime is dominated by CUDA context initialization:
```
cudaMalloc: 136ms (first call initializes context)
```

Julia's 37 sec cold start includes:
- Package loading
- JIT compilation of all code paths
- CUDA.jl initialization

### 3. Warm Julia Appears Faster (Misleading)

When measuring within a warm Julia session:
- Context already initialized
- Memory pools pre-allocated
- JIT compilation complete

This gives Julia an apparent advantage that doesn't reflect kernel performance.

### 4. Memory Transfer Times Are Similar

Both implementations show similar memcpy overhead:
```
C++ nn:   Host-to-Device: 991 µs, Device-to-Host: 914 µs
Julia nn: Host-to-Device: 993 µs
```

### 5. Code Verbosity

Lines of code comparison:
| Component | C/C++ | Julia |
|-----------|-------|-------|
| backprop kernel | 105 lines | 73 lines |
| nn total | 373 lines | 200 lines |

Julia is more concise due to higher-level abstractions, but this comes at a runtime cost.

---

## Conclusion

**Neither benchmark benefits from Julia CUDA at the kernel execution level.**

C++ produces faster GPU code for both benchmarks. The apparent Julia advantage in warm benchmarks is due to amortized host-side initialization, not superior kernel performance.

**When to choose Julia CUDA:**
- Rapid prototyping
- Integration with Julia ecosystem
- Workflows where JIT cost is amortized over many iterations

**When to choose C++ CUDA:**
- Maximum kernel performance required
- Cold-start latency matters
- Production deployments with strict performance requirements
