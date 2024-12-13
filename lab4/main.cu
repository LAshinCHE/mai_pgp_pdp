#include <stdio.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

/*
 n x n
 i, j: index = n * i + j
 index: j = index % j, i = index / n
*/

struct AbsComparator {
    __host__ __device__ bool operator()(double a, double b) {
        return abs(a) < abs(b);
    }
};


__global__ void SwapRows(double* matrix, int n, int row1, int row2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset_x = blockDim.x * gridDim.x;

    for (int x = idx; x < 2 * n; x += offset_x) {
        double tmp = matrix[x * n + row1];
        matrix[x * n + row1] = matrix[x * n + row2];
        matrix[x * n + row2] = tmp;
    }
}
__global__ void ZeroingLowerColumns(double* matrix, int n, int i) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int y = idy + i + 1; y < 2 * n; y += offset_y) {
        for (int x = idx + i + 1; x < n; x += offset_x){
            matrix[y * n + x] += matrix[y * n + i] * (-matrix[i * n + x] / matrix[i * n + i]);
        }
    }
}

__global__ void ZeroingUpperColumns(double* matrix, int n, int i) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int y = idy + i + 1; y < 2 * n; y += offset_y) {
        for (int x = i - 1 - idx; x >= 0; x -= offset_x) {
            matrix[y * n + x] += matrix[y * n + i] * (-matrix[i * n + x] / matrix[i * n + i]);
        }
    }
}

void PrintReverseMatrix(double* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = n; j < 2 * n; j++) {
            printf("%.10e ", matrix[i + j * n]);
        }
        printf("\n");
    }

}

void Normalize(double* matrix, int n) {
    for (int i = 0; i < n; i += 1) {
        for (int j = n; j < 2 * n; j += 1) {
            matrix[i + j * n] /= matrix[i * n + i];
        }
    }
}


const int SWAP_FUNC_BLOCKS = 32;
const int SWAP_FUNC_THREADS = 32;
const int ZEROING_FUNC_BLOCKS_X = 32;
const int ZEROING_FUNC_BLOCKS_Y = 32;
const int ZEROING_FUNC_THREADS_X = 32;
const int ZEROING_FUNC_THREADS_Y = 32;

int main() {
    int n, i, j;
    scanf("%d", &n);
    double* src = (double*)malloc(sizeof(double) * n * n * 2);

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            scanf("%lf", &src[i + j * n]);
        }
    }

    for (i = 0; i < n; i++){
        for (j = n; j < 2 * n; j++) {
            if (i == j - n) {
                src[i + j * n] = 1;
            }
            else {
                src[i + j * n] = 0;
            }
        }
    }


    double* dev_src;
    const AbsComparator comp;

    CSC(cudaMalloc(&dev_src, sizeof(double) * 2 * n * n));
    CSC(cudaMemcpy(dev_src, src, sizeof(double) * 2 * n * n, cudaMemcpyHostToDevice));


    dim3 X(ZEROING_FUNC_BLOCKS_X, ZEROING_FUNC_THREADS_X);
    dim3 Y(ZEROING_FUNC_BLOCKS_Y, ZEROING_FUNC_THREADS_Y);
    for (i = 0; i < n; i++) {
            const thrust::device_ptr<double> ptr = thrust::device_pointer_cast(dev_src + i * n);
            const thrust::device_ptr<double> maxPtr = thrust::max_element(ptr + i, ptr + n, comp);
        int maxIndex = maxPtr - ptr;

        if (maxIndex != i) {
            SwapRows << <SWAP_FUNC_BLOCKS, SWAP_FUNC_THREADS >> > (dev_src, n, i, maxIndex);
        }
        ZeroingLowerColumns << <X, Y >> > (dev_src, n, i);
    }
    CSC(cudaDeviceSynchronize());
    for (i = n - 1; i >= 0; i--) {
        ZeroingUpperColumns << <X, Y >> > (dev_src, n, i);
    }

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(src, dev_src, sizeof(double) * 2 * n * n, cudaMemcpyDeviceToHost));
    Normalize(src, n);
    PrintReverseMatrix(src, n);
    CSC(cudaFree(dev_src));
    free(src);
}