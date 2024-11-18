#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)


__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;
    uchar4 p;
    for(y = idy; y < height; y += offsety){
        for(x = idx; x < width; x += offsetx) {
            double w[3][3];

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    int xi = x + i - 1;
                    int yj = y + j - 1;
                    uchar4 p = tex2D<uchar4>(tex, xi, yj);
                    w[i][j] = 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
                }
            }

            double gx = w[0][2] + 2 * w[1][2] + w[2][2] - w[0][0] - 2 * w[1][0] - w[2][0];
            double gy = w[2][0] + 2 * w[2][ 1] + w[2][2] - w[0][0] - 2 * w[0][1] - w[0][2];
            int gf = min(255, int(sqrt(gx*gx + gy*gy)));

            out[y *width + x] = make_uchar4(gf, gf, gf, gf);
        }
    }
}




int main() {
    int w, h;
    std::string input;
    std::cin >> input;
    std::string output;
    std::cin >> output;
    FILE *fp = fopen(input.c_str(), "rb");
 	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
 	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<< dim3(16, 16), dim3(32, 32) >>>(tex, dev_out, w, h);
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(output.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

    free(data);
    return 0;
}