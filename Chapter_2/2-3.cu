#include <stdio.h>
#include <cuda_runtime.h>

int main(){
    int nElem = 1024;

    dim3 block = (1024);
    dim3 grid = ((block.x+nElem-1)/block.x);

    printf("blockIdx: %d gridIdx: %d\n", block.x, grid.x);

    block.x = 512;
    grid.x = (block.x+nElem-1)/block.x;
    printf("blockIdx: %d gridIdx: %d\n", block.x, grid.x);


    block.x = 256;
    grid.x = (block.x+nElem-1)/block.x;
    printf("blockIdx: %d gridIdx: %d\n", block.x, grid.x);


    block.x = 128;
    grid.x = (block.x+nElem-1)/block.x;
    printf("blockIdx: %d gridIdx: %d\n", block.x, grid.x);

    
    reutrn 0;
}