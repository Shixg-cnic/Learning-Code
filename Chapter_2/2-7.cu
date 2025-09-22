#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call)                                                                 \
{                                                                                   \
    const cudaError_t error = call;                                                 \
    if (error != cudaSuccess){                                                      \
        printf("Error : %s:%d, ", __FILE__, __LINE__);                              \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));         \
    exit(1);                                                                        \
    }                                                                               \
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;
    for(int i = 0; i < N; i++){
        if(hostRef[i] - gpuRef[i] > epsilon){
            match = 0;
            printf("Arrays do not match!\n ");
            printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
        }
            }
    if(match){
            printf("Arrays match.\n\n");
        }

}

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned) time(&t));
    
    for(int i = 0; i < size; i++){
        ip[i] = (float) ( rand() & 0xFF )/10.0f;

    }
}

void initialInt(int *ip, int size){
    for(int i = 0; i < size; i++){
        ip[i] = i;
    }
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void sumMatrixOnHost(float *A, float *B, float *C, int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for(int iy = 0; iy < ny; iy++){
        for(int ix = 0; ix < nx; ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = ix + iy * nx;
    if(ix < nx && iy < ny){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using Device %d: %s",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1<<14;
    int ny = 1<<15;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d",nx,ny);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    double iStart, iElaps;

    iStart = cpuSecond();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    iElaps = cpuSecond() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;

    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    cudaMemcpy(d_MatA, *h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, *h_B, nBytes, cudaMemcpyHostToDevice);

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx,dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    iStart = cpuSecond();
    sumMatrixOnGPU2D <<< grid, block >>> (d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU2D Execution configuration <<< (%d,%d),(%d,%d)>>> Time elepsed %f sec\n" ,grid.x, grid.y, block.x, block.y, iElaps);
    
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    
    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}   