#define MAX_SM_SIZE 256

// CUDA runtime

#include <vector>
#include <time.h>
#include <typeinfo>
#include <stdio.h>
#include <iostream>
#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
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


pair<vector<void *>, vector<unsigned long>> convert_csr(float *src, int h, int w)
{
    vector<float> value;
    vector<MKL_INT> row_idx;
    vector<MKL_INT> col_idx;
    MKL_INT pos;
    for (MKL_INT i = 0; i < h; i++)
    {
        row_idx.push_back(value.size());
        for (MKL_INT j = 0; j < w; j++)
        {
            pos = i * w + j;
            if (src[pos] != 0.0)
            {
                value.push_back(src[pos]);
                col_idx.push_back(j);
            }
        }
    }
    float *ptr_v = (float *)mkl_malloc(sizeof(float) * value.size(), 64);
    MKL_INT *ptr_r = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * row_idx.size(), 64);
    MKL_INT *ptr_c = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * col_idx.size(), 64);
    if(ptr_v==NULL || ptr_r==NULL || ptr_c==NULL){
        throw "Host memory allocation failed!";
        printf("ptr_v is NULL!!!!\n");
    }
    // copy the data
    for(int i=0; i<value.size(); i++) ptr_v[i] = value[i];
    for(int i=0; i<row_idx.size(); i++) ptr_r[i] = row_idx[i];
    for(int i=0; i<col_idx.size(); i++) ptr_c[i] = col_idx[i];
    vector<void *> ptrs = {(void *)ptr_v, (void *)ptr_r, (void *)ptr_c};
    // printf("ptr_v:%x\n", ptr_v);
    // printf("ptrs[0]:%x\n",ptrs[0]);
    // printf(" cpu value length: %ld", sizeof(float) * value.size());
    vector<unsigned long> sizes = {value.size(), row_idx.size(), col_idx.size()};
    return make_pair(ptrs, sizes);
}

void show(float * ptr, int size){
    for(int i=0;i<size;i++){
        printf("%f \n", ptr[i]);
    }
}

int main(){
    float *A, *B, *C;
    MKL_INT m, n, k, i, j;
    float alpha, beta, sparsity;
    m = 1024;
    n = 1024;
    k = 1024;
    alpha = 1.0; beta = 0.0; sparsity = 0.5;
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
    random_init(A, m*k, sparsity);
    random_init(B, k*n, 0);
    
    pair<vector<void *>, vector<unsigned long>> csr = convert_csr(A, m, k);
    vector<void *> ptrs = csr.first;
    vector<unsigned long> sizes = csr.second;
    float * values = (float *)ptrs[0];
    MKL_INT * rowIndex = (MKL_INT *) ptrs[1];
    MKL_INT * columns = (MKL_INT *) ptrs[2];
    
    char		transa, uplo, nonunit;
    char		matdescra[6];

    transa = 't';
    matdescra[0] = 't';
    matdescra[1] = 'l';
    matdescra[2] = 'n';
    matdescra[3] = 'c';
    mkl_scsrmm(&transa, &m, &n, &m, &alpha, matdescra, values, columns, rowIndex, &(rowIndex[1]), B, &n,  &beta, C, &n);
    MKL_INT niter = 1;
    clock_t t_start = clock();
    for(MKL_INT iter_id=0; iter_id<niter; iter_id+=1){
        mkl_scsrmm(&transa, &m, &n, &m, &alpha, matdescra, values, columns, rowIndex, &(rowIndex[1]), B, &n,  &beta, C, &n);
    }
    clock_t t_end = clock();
    double timecost = (t_end - t_start) * 1.0 / CLOCKS_PER_SEC * 1000 / niter;

    show(A, 100);
    show(B, 100);
    show(C, 100);

    printf("Time Cost: %lf Sparsity: %f \n", timecost, sparsity);

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    printf("Finished!!\n");
    return 0;
}