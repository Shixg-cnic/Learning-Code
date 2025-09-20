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

void sumArrayOnHost(float *A, float *B, float *C, const int N){
    for(int i = 0; i < N; i++){
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C){

    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


int main(){
    printf("Strating...\n");
    
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s",dev,deviceProp.name);
    printf("Total global memory: %zu\n", deviceProp.totalGlobalMem);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    CHECK(cudaSetDevice(dev));

    int nElem = 1<<24;
    printf("Vector size %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);
    
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    double iStart, iElaps;

    iStart = cpuSecond();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = cpuSecond() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    istart = cpuSecond();
    sumArrayOnHost(h_A,h_B,hostRef,nElem);
    iElaps = cpuSecond() - iStart;

    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    int iLen = 1024;
    dim3 block (iLen);
    dim3 grid ((nElem+block.x-1)/block.x);

    istart = cpuSecond();
    sumArrayOnGPU<<< grid, block >>> (d_A, d_B, d_C);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - istart;
    printf("Execution configuration <<< %d,%d>>> Time elepsed %f sec\n" ,block.x, grid.x, iElaps);
    
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // sumArrayOnHost(h_A, h_B, hostRef, nElem);

    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}