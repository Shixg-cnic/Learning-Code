// sumArraysOnHost
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++){
        ip[i] = (float) (rand() & 0XFF )/ 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for(int idx = 0; idx < N; idx++){
        C[idx] = A[idx] + B[idx];
        printf("%f + %f = %f\n", A[idx], B[idx], C[idx]);
    }
}



int main(){
    int nElem = 10;
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *) malloc(nBytes);
    h_B = (float *) malloc(nBytes);
    h_C = (float *) malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    sumArraysOnHost(h_A, h_A, h_C, nElem);

   

    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);




    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

