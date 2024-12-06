#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <limits>


using namespace std;

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

typedef struct{
    int x;
    int y;
} pixel;


typedef struct{
    float red;
    float green; 
    float blue;
} pixelcolor;

const int NUM_BLOCKS = 32;
const int NUM_THREAD = 32;

__constant__ pixelcolor avg_dev[32];
__constant__ float norm_dev[32];

__device__ unsigned char calculateClass(int numberOfClasses, uchar4 p) {
    unsigned char maxClass = 0;
    float maxVal = 0;
    for(int i = 0; i < numberOfClasses; i++){
        float scalarProduct = p.x * avg_dev[i].red + p.y * avg_dev[i].green + p.z * avg_dev[i].blue;
        float pixelNorm = sqrt(float(p.x * p.x + p.y * p.y + p.z * p.z));
        float spec = scalarProduct / (pixelNorm * norm_dev[i]);
        if(spec > maxVal){
            maxVal = spec;
            maxClass = i;
        }
    }
    return maxClass;
}

__global__ void kernel(uchar4 *data, int width, int height, int numberOfClasses) {
    int id_x = blockDim.x * blockIdx.x + threadIdx.x;
    int offset_x = blockDim.x * gridDim.x;

    for (int x = id_x; x < width*height; x += offset_x) {
		data[x].w = calculateClass(numberOfClasses, data[x]);
	}
}


vector<pixelcolor> countAvg(vector<vector<pixel>>& classes, uchar4* data, int w){
    int numberOfClasses =  classes.size();
    vector<pixelcolor> avg(numberOfClasses);
    for(int i = 0; i < numberOfClasses; i++){
        avg[i].red = 0;
        avg[i].green = 0;
        avg[i].blue = 0;
        int numberOfPixels = classes[i].size();
        for(int j =0; j < numberOfPixels; j++ ){
            uchar4 pixel = data[classes[i][j].x + classes[i][j].y * w];
            avg[i].red += pixel.x;
            avg[i].green += pixel.y;
            avg[i].blue += pixel.z;
        }
        avg[i].red /= numberOfPixels;
        avg[i].green /= numberOfPixels;
        avg[i].blue /= numberOfPixels;
    }
    return avg;
}


vector<float> countNorm(vector<pixelcolor>& avg){
    int numberOfClasses = avg.size() ;
    vector<float> norm(numberOfClasses);
    for (int i = 0; i < numberOfClasses; i++){
        norm[i] =  sqrt(avg[i].red  * avg[i].red + avg[i].green * avg[i].green + avg[i].blue*avg[i].blue);
    }
    return norm;
}

int main() {

    int w, h;
    string input;
    cin >> input;
    string output;
    cin >> output;
    FILE *fp = fopen(input.c_str(), "rb");
 	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
 	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    int numberOfClasses;
    cin >> numberOfClasses;
    vector<vector<pixel>> classes(numberOfClasses);

    for(int i = 0; i < numberOfClasses; i++){
        int numberOfPixels;
        cin >> numberOfPixels;
        vector<pixel> classElements(numberOfPixels);
        for(int j = 0; j < numberOfPixels; j++){
            cin >> classElements[j].x;
            cin >> classElements[j].y;
        }
        classes[i] = classElements;
    }

    vector<pixelcolor> avg = countAvg(classes, data, w);
    vector<float> norm = countNorm(avg);
    // for(int i = 0;  i  < norm.size(); i++){
    //     cout << norm[i].red << " " << norm[i].green << " " << norm[i].blue << '\n';
    // }

    // for(int i = 0;  i  < avg.size(); i++){
    //     cout << avg[i].red << " " << avg[i].green << " " << avg[i].blue << '\n';
    // }

    uchar4* arr;
    CSC(cudaMalloc(&arr, sizeof(uchar4) * w*h));
    CSC(cudaMemcpy(arr, data, sizeof(uchar4) * w*h, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(avg_dev, avg.data(), sizeof(pixelcolor) * numberOfClasses, 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(norm_dev, norm.data(), sizeof(float) * numberOfClasses, 0, cudaMemcpyHostToDevice));

    kernel<<<NUM_BLOCKS, NUM_THREAD>>>(arr, w, h, numberOfClasses);
    CSC(cudaPeekAtLastError());

    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());


    CSC(cudaMemcpy(data, arr, sizeof(uchar4) * w*h, cudaMemcpyDeviceToHost));

    CSC(cudaFree(arr));

    fp = fopen(output.c_str(), "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

    free(data);
    return 0;
}