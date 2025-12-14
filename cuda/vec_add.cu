#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

__global__ void addV1(int *output, const int *a, const int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n){
        output[i] = a[i] + b[i];
    }
}

__global__ void add(int *output, const int *a, const int *b, int n) {
    for (
        int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n; 
        i += blockDim.x * gridDim.x
    ) {
        output[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20;
    const size_t size = n * sizeof(int);

    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_output = (int*)malloc(size);

    int *d_a, *d_b, *d_output;
    CUDA_OK(cudaMalloc((void**)&d_a, size));
    CUDA_OK(cudaMalloc((void**)&d_b, size));
    CUDA_OK(cudaMalloc((void**)&d_output, size));

    // Initialize input
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    CUDA_OK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int numSMs;
    CUDA_OK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
    printf("num sms: %d\n", numSMs);

    add<<<32 * numSMs, 256>>>(d_output, d_a, d_b, n);

    CUDA_OK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Verify result
    for (int i = 0; i < n; ++i) {
        assert(h_output[i] == h_a[i] + h_b[i]);
    }

    printf("Add completed successfully.\n");

    free(h_a);
    free(h_b);
    free(h_output);
    CUDA_OK(cudaFree(d_a));
    CUDA_OK(cudaFree(d_b));
    CUDA_OK(cudaFree(d_output));

    return 0;
}
