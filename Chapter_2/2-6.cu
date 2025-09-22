#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                                 \
{                                                                                   \
    const cudaError_t error = call;                                                 \
    if (error != cudaSuccess){                                                      \
        printf("Error : %s:%d, ", __FILE__, __LINE__);                              \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));         \
    exit(1);                                                                        \
    }                                                                               \
}   

void initialInt(int *ip, int size){
    for(int i = 0; i < size; i++){
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("\nMatrix: (%d,%d)\n",nx,ny);
    for(int iy = 0; iy < ny; iy++){
        for(int ix = 0; ix < nx; ix++){
            printf("3d%", ic[ix]);
        }
        ic = ic + nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = ix + iy * nx;
    printf("thread_id (%d,%d) block_id (%d,%d) cootdinate(%d,%d) globa lindex: %2d ival: %2d ",
            threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]); 
}

int main(){
    printf("Strating...\n");
    
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudeGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d: %s",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 6;
    int ny = 8;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    int *h_A;
    (int *)malloc(nBytes);

    initialInt(h_A,nxy);
    printMatrix(h_A,nx,ny);

    int *d_MatA;
    cudaMalloc((void **)&d_MatA,nBytes);

    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    dim3 block(2,4);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    printThreadIndex <<<block,grid>>>(d_MatA,nx,ny);

    cudaFree(d_MatA);
    free(h_A);

    cudaDeviceReset();
    return 0;

}