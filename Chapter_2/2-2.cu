#include <stdio.h>
#include <cuda_runtime.h>

__global__ void checkIndex(void){
    printf("threadIdx.x: %d threadIdx.y: %d threadIdx.z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx.x: %d blockIdx.y: %d blockIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("blockDim.x: %d blockDim.y: %d, blockDim.z: %d\n", blockDim.x, blockIdx.y, blockIdx.z);
    printf("gridDim.x: %d gridDim.y: %d gridDim.z: %d\n", gridDim.x, gridDim.y, gridDim.z);

}


int main(){
    int Elem = 6;
    dim block (3);
    dim grid((nElem+block.x-1)/block.x);
    
    printf("grid x,y,z: %d %d %d", grid.x, gird.y, grid.z);
    printf("block x,y,z: %d %d %d", block.x, block.y, block.z);
    
    checkIndex <<<grid, block>>>();
    
    cudaDeviceReset();

    return 0;

}