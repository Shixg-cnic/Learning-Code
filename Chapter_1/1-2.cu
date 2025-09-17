#include <stdio.h>

_global_ void helloFromGPU(void){
    printf("Hello World from GPU!\n");
}

int main(void){
    printf("Hello World from CPU!\n");
    helloFromGPU <<<1,10>>>();
    cudaDeviceReset();
    return 0;
  
}
