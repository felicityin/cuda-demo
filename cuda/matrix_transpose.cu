#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

__global__ void transposeNaive(int *output, const int *input, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    size_t len = width * height;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t idx_in = (row + j) * width + col;
        size_t idx_out = col * height + row + j;
        if (idx_in < len && idx_out < len) {
            output[idx_out] = input[idx_in];
        }
    }
}

__global__ void transposeSharedMemory(int *output, const int *input, int width, int height) {
    __shared__ int s_mem[TILE_DIM][TILE_DIM+1];
        
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    size_t len = width * height;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t idx_in = (row + j) * width + col;
        if (idx_in < len) {
            s_mem[threadIdx.y + j][threadIdx.x] = input[idx_in];
        }
    }

    __syncthreads();

    col = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    row = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t idx_out = (row + j) * height + col;
        if (idx_out < len) {
            output[idx_out] = s_mem[threadIdx.x][threadIdx.y + j];
        }
    }
}

bool verify_result(const int *h_output, const int *h_input, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (h_output[j * height + i] != h_input[i * width + j]) {
                return false;
            }
        }
    }
    return true;
}

void print_matrix(const int *matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%d ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    const int width = 100;
    const int height = 1 << 10;
    const size_t size = width * height * sizeof(int);

    int *h_input = (int*)malloc(size);
    int *h_output = (int*)malloc(size);

    int *d_input, *d_output;
    CUDA_OK(cudaMalloc((void**)&d_input, size));
    CUDA_OK(cudaMalloc((void**)&d_output, size));

    // Initialize input matrix
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = i;
    }

    CUDA_OK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    float ms;

    // -------------- Transpose using naive kernel --------------

    dim3 blockSize(TILE_DIM, BLOCK_ROWS);
    dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        transposeNaive<<<gridSize, blockSize>>>(d_output, d_input, width, height);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[native] Average time per transpose: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    assert(verify_result(h_output, h_input, width, height));

    // -------------- Transpose using shared memory kernel --------------

    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        transposeSharedMemory<<<gridSize, blockSize>>>(d_output, d_input, width, height);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[shared memory] Average time per transpose: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    assert(verify_result(h_output, h_input, width, height));

    printf("Transpose completed successfully.\n");

    free(h_input);
    free(h_output);
    CUDA_OK(cudaFree(d_input));
    CUDA_OK(cudaFree(d_output));

    return 0;
}
