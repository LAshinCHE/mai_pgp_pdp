#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>


#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)



__global__ void kernel(cudaTextureObject_t texture, uchar4 *out, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;

    int Mx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int My[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    for(y = idy; y < height; y += offsety){
        for(x = idx; x < width; x += offsetx) {
            double Gx = 0.0, Gy = 0.0;
            
            double grayscaleMatrix[3][3];
            
            uchar4 p0 = tex2D<uchar4>(texture, x - 1, y - 1);
            grayscaleMatrix[0][0] = 0.299 * p0.x + 0.587 * p0.y + 0.114 * p0.z;

            uchar4 p1 = tex2D<uchar4>(texture, x, y - 1);
            grayscaleMatrix[0][1] = 0.299 * p1.x + 0.587 * p1.y + 0.114 * p1.z;

            uchar4 p2 = tex2D<uchar4>(texture, x + 1, y - 1);
            grayscaleMatrix[0][2] = 0.299 * p2.x + 0.587 * p2.y + 0.114 * p2.z;

            uchar4 p3 = tex2D<uchar4>(texture, x - 1, y);
            grayscaleMatrix[1][0] = 0.299 * p3.x + 0.587 * p3.y + 0.114 * p3.z;

            uchar4 p4 = tex2D<uchar4>(texture, x, y);
            grayscaleMatrix[1][1] = 0.299 * p4.x + 0.587 * p4.y + 0.114 * p4.z;

            uchar4 p5 = tex2D<uchar4>(texture, x + 1, y);
            grayscaleMatrix[1][2] = 0.299 * p5.x + 0.587 * p5.y + 0.114 * p5.z;

            uchar4 p6 = tex2D<uchar4>(texture, x - 1, y + 1);
            grayscaleMatrix[2][0] = 0.299 * p6.x + 0.587 * p6.y + 0.114 * p6.z;

            uchar4 p7 = tex2D<uchar4>(texture, x, y + 1);
            grayscaleMatrix[2][1] = 0.299 * p7.x + 0.587 * p7.y + 0.114 * p7.z;

            uchar4 p8 = tex2D<uchar4>(texture, x + 1, y + 1);
            grayscaleMatrix[2][2] = 0.299 * p8.x + 0.587 * p8.y + 0.114 * p8.z;


            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    Gx += grayscaleMatrix[i][j] * Mx[i][j];
                    Gy += grayscaleMatrix[i][j] * My[i][j];
                }
            }

            int Gf;

            if(int(sqrt(Gx * Gx + Gy * Gy)) < 255){
                Gf = int(sqrt(Gx * Gx + Gy * Gy));
            } else {
                Gf = 255;
            }

            out[y *width + x] = make_uchar4(Gf, Gf, Gf, Gf);
        }
    }
}


const int BLOCK_X= 16;
const int THREAD_X = 16;
const int BLOCK_Y = 16;
const int THREAD_Y = 16;

int main(){
    std::string input, output;
    int w, h;

    std::cin >> input >> output;

    FILE *fp = fopen(input.c_str(), "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);


    cudaArray *dev_arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&dev_arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(dev_arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t texture = 0;
    CSC(cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    dim3 X(BLOCK_X, THREAD_X);
    dim3 Y(BLOCK_Y, THREAD_Y);

    kernel<<<X, Y>>>(texture, dev_out, w, h);
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(texture));
    CSC(cudaFreeArray(dev_arr));
    CSC(cudaFree(dev_out));

    fp = fopen(output.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

    free(data);
    return 0;
}