#include<stdio.h>

#define cudaCheckError() {                                                       \
    cudaError_t e=cudaGetLastError();                                            \
    if(e!=cudaSuccess) {                                                         \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                                 \
    }                                                                            \
}

__global__ void kernel(double *arr1, double *arr2, double *ans, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ans[idx] = (arr1[idx] < arr2[idx]) ? arr1[idx] : arr2[idx];
    }
}

void readVector(double* arr, int n){
    for (int i = 0; i < n; i++) {
        scanf("%lf", &arr[i]);
    }
}

void printVector(double* arr, int n){
    for (int i = 0; i < n; i++) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main(){
    int n;
    scanf("%d", &n);
    
    double *arr1 = (double*)malloc(sizeof(double) * n);
    double *arr2 = (double*)malloc(sizeof(double) * n);
    double *ans =  (double*)malloc(sizeof(double) * n);
    readVector(arr1, n);
    readVector(arr2, n);
    
    double *dev_arr1, *dev_arr2, *dev_ans;
    cudaMalloc(&dev_arr1, sizeof(double) * n); cudaCheckError();
    cudaMalloc(&dev_arr2, sizeof(double) * n); cudaCheckError();
    cudaMalloc(&dev_ans, sizeof(double) * n); cudaCheckError();
    
    cudaMemcpy(dev_arr1, arr1, sizeof(double) * n, cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(dev_arr2, arr2, sizeof(double) * n, cudaMemcpyHostToDevice); cudaCheckError();
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_arr1, dev_arr2, dev_ans, n);
    cudaCheckError();
    
    cudaDeviceSynchronize(); cudaCheckError();
    
    cudaMemcpy(ans, dev_ans, sizeof(double) * n, cudaMemcpyDeviceToHost); cudaCheckError();

    printVector(ans,n);

    free(arr1);
    free(arr2);
    free(ans);
    cudaFree(dev_arr1);
    cudaFree(dev_arr2);
    cudaFree(dev_ans);

    return 0;
}
