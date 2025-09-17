#include <stdio.h>

__global__ void helloFromGPU(void){
    printf("Hello World from GPU!\n");
}

__global__ void helloFromGPU1(void){
    int a = threadIdx.x;
    printf("Hello World from GPU %d\n", a);
    
}

int main(void){
    printf("Hello World from CPU!\n");
    helloFromGPU1 <<<1,10>>>();
    cudaDeviceReset();
    return 0;
  
}
