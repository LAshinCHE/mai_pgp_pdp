#include<stdio.h>

__global__ void kernel(double *arr1, double *arr2, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (idx < n) {
        arr1[idx] = (arr1[idx] < arr2[idx]) ? arr1[idx] : arr2[idx];
        idx += offset;
    }
}

#define BENCHMARK

const int BLOCKS = 32;
const int THREADS = 32;

void readVector(double* arr, int n){
    for (int i = 0; i < n; i++) {
        scanf("%lf", &arr[i]);
    }
}

void printVector(double* arr, int n){
    for (int i = 0; i < n; i++) {
        printf("%.10lf ", arr[i]);
    }
    printf("\n");
}

int main(){
    int n;
    scanf("%d", &n);

    double *arr1 = (double*)malloc(sizeof(double) * n);
    double *arr2 = (double*)malloc(sizeof(double) * n);
    readVector(arr1, n);
    readVector(arr2, n);

    double *dev_arr1, *dev_arr2;
    cudaMalloc(&dev_arr1, sizeof(double) * n);
    cudaMalloc(&dev_arr2, sizeof(double) * n);

    cudaMemcpy(dev_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice);

    #ifdef BENCHMARK
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    #endif /* BENCHMARK */


    kernel<<<BLOCKS, THREADS>>>(dev_arr1, dev_arr2, n);

    cudaDeviceSynchronize();

    #ifdef BENCHMARK
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("time = %f ms\n", time);
    #endif /* BENCHMARK */


    cudaMemcpy(arr1, dev_arr1, sizeof(double) * n, cudaMemcpyDeviceToHost);
    #ifndef BENCHMARK
    printVector(arr1,n);
    #endif

    free(arr1);
    free(arr2);
    cudaFree(dev_arr1);
    cudaFree(dev_arr2);

    return 0;
}