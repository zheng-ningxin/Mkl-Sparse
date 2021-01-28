#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
using namespace std;



void random_init(float *ptr, int size, float sparsity)
{
    for (int i = 0; i < size; i++)
    {
        float pro = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (pro < sparsity)
        {
            ptr[i] = 0.0;
        }
        else
        {
            ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

void show(float * ptr, int size){
    for(int i=0;i<size;i++){
        printf("%f \n", ptr[i]);
    }
}

int main()
{
    float *A, *B, *C;
    int m, n, k, i, j;
    float alpha, beta;

    printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
            " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
            " alpha and beta are double precision scalars\n\n");

    m = 1024, k = 1024, n = 1024;
    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;

    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");
    A = (float *)mkl_malloc( m*k*sizeof( float ), 64 );
    B = (float *)mkl_malloc( k*n*sizeof( float ), 64 );
    C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }

    random_init(A, m*k, 0);
    random_init(B, k*n, 0);
    printf (" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    // warmup
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
    int niter = 10000;
    clock_t t_start = clock();
    for(int i=0;i<niter;i++){
        // printf("%d\n", i);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
    }
    show(A, 100);
    show(B, 100);
    show(C, 100);
    clock_t t_end = clock();
    double timecost = (t_end - t_start) * 1.0 / CLOCKS_PER_SEC * 1000 / niter;
    printf("Time Cost: %lf \n", timecost);

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    // printf (" Example completed. \n\n");
    return 0;
}