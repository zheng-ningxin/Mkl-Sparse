/********************************************************************************
!                              INTEL CONFIDENTIAL
!   Copyright(C) 2006-2008 Intel Corporation. All Rights Reserved.
!   The source code contained  or  described herein and all documents related to
!   the source code ("Material") are owned by Intel Corporation or its suppliers
!   or licensors.  Title to the  Material remains with  Intel Corporation or its
!   suppliers and licensors. The Material contains trade secrets and proprietary
!   and  confidential  information of  Intel or its suppliers and licensors. The
!   Material  is  protected  by  worldwide  copyright  and trade secret laws and
!   treaty  provisions. No part of the Material may be used, copied, reproduced,
!   modified, published, uploaded, posted, transmitted, distributed or disclosed
!   in any way without Intel's prior express written permission.
!   No license  under any  patent, copyright, trade secret or other intellectual
!   property right is granted to or conferred upon you by disclosure or delivery
!   of the Materials,  either expressly, by implication, inducement, estoppel or
!   otherwise.  Any  license  under  such  intellectual property  rights must be
!   express and approved by Intel in writing.
!
!*******************************************************************************
!   Content : MKL Sparse BLAS C example
!
!*******************************************************************************
!
! Example program for using MKL Sparse BLAS Level 2 and 3
! for matrices represented in the compressed sparse row storage scheme.
! The following Sparse  Blas routines are used in the example:
!          MKL_DCSRSM  MKL_DCSRSV  MKL_DCSRMM  MKL_DCSRMV
!          MKL_CSPBLAS_DCSRGEMV    MKL_CSPBLAS_DCSRSYMV
!		   MKL_CSPBLAS_DCSRTRSV.
!
! Consider the matrix A (see Appendix 'Sparse Storage Formats for Sparse Blas
! level 2-3')
!
!                 |   1       -1      0   -3     0   |
!                 |  -2        5      0    0     0   |
!   A    =        |   0        0      4    6     4   |,
!                 |  -4        0      2    7     0   |
!                 |   0        8      0    0    -5   |
!
!
! decomposed as
!
!                      A = L + D + U,
!
!  where L is the strict lower triangle of A, U is the strictly  upper triangle
!  of A, D is the main diagonal. Namely
!
!        |   0    0   0    0     0   |       |  0   -1    0   -3   0   |
!        |  -2    0   0    0     0   |       |  0    0    0    0   0   |
!   L  = |   0    0   0    0     0   |,  U=  |  0    0    0    6   4   |
!        |  -4    0   2    0     0   |       |  0    0    0    0   0   |
!        |   0    8   0    0     0   |       |  0    0    0    0   0   |
!
!
!           |   1  0  0   0   0   |
!           |   0  5  0   0   0   |
!   D    =  |   0  0  4   0   0   |.
!           |   0  0  0   7   0   |
!           |   0  0  0   0  -5   |
!
!  The matrix A is represented in the zero-based compressed sparse row storage
!  scheme with the help of three arrays  (see Appendix 'Sparse Matrix Storage')
!  as follows:
!
!         values = (1 -1 -3 -2 5 4 6 4 -4 2 7 8 -5)
!         columns = (0 1 3 0 1 2 3 4 0 2 3 1 4)
!         rowIndex = (0  3  5  8  11 13)
!
!  It should be noted that two variations of the compressed sparse row storage
!  scheme are supported by Intel MKL Sparse Blas (see 'Sparse Storage Formats
!  for Sparse Blas level 2-3') :
!
!        1. variation accepted in the NIST Sparse Blas (zero-based modification)
!        2. variation accepted for many other libraries (zero-based documentation).
!
!  The representation of the matrix A  given above is the 2's variation. Two
!  integer arrays pointerB and pointerE instead of the array rowIndex are used in
!  the NIST variation of variation of the compressed sparse row format. Thus the
!  arrays values and columns are the same for the both variations. The arrays
!  pointerB and pointerE for the matrix A are defined as follows:
!                          pointerB = (0 3  5  8 11)
!                          pointerE = (3 5  8 11 13)
!  It's easy to see that
!                    pointerB[i]= rowIndex[i] for i=0, ..4;
!                    pointerE[i]= rowIndex[i+1] for i=0, ..4.
!
!
!  The purpose of the given example is to show
!
!             1. how to form the arrays pointerB and pointerE for the NIST's
!				 variation of the  compressed sparse row format using
!                the  array rowIndex
!             2. how to use minors of the matrix A by redefining the arrays
!				 pointerB and pointerE but the arrays values and columns are
!				 the same.
!
!  In what follows the symbol ' means taking of transposed.
!
!  The test performs the following operations :
!
!       1. The code computes (L+D)'*S = F using MKL_DCSRMM where S is a known
!		   5 by 2 matrix and then the code solves the system (L+D)'*X = F with
!		   the help of MKL_DCSRSM. It's evident that X should be equal to S.
!
!       2. The code computes (U+I)'*S = F using MKL_DCSRMV where S is a vector
!          and then the code calls MKL_CSPBLAS_DCSRTRSV solves the system
!		   (U+I)'*X = F with the single right hand side. It's evident
!			that X should be equal to S.
!
!       3. The code computes D*S = F using MKL_DCSRMV where S is a vector
!          and then the code solves the system D*X = F with the single
!		   right hand side. It's evident that X should be equal to S.
!
!       4. The next step is the computation (U-U') S = F using MKL_DCSRMV where
!          S is a vector. It is easy to see that U-U' is a skew-symmetric matrix.
!
!       5. The next step is the computation (L+D+L') S = F using
!		   MKL_CSPBLAS_DCSRSYMV where S is a vector. It is easy to see that
!		   L+D+L' is a symmetric matrix.
!
!       6. The next step is the computation A'* S = F using MKL_CSPBLAS_DCSRGEMV
!		   where S is a vector.
!
!       7. Let's T be the upper 3 by 3 minor of the matrix A. Namely, T is
!		   the following matrix
!
!                        |   1       -1      0   |
!          T    =        |  -2        5      0   |.
!                        |   0        0      4   |
!          The test performs the matrix-vector multiply T*S=F with the same
!		   arrays values, columns and pointerB used before for the whole
!		   matrix A. It is enough to change two values of array pointerE
!		   in order to use the minor under consideration. Then the test
!		   solves the system T*X =F using MKL_DCSRSV. The routine MKL_DCSRMV
!		   is used for getting matrix-vector multiply.
!
! The code given below uses only one sparse representation for the all operations.
!
!*******************************************************************************
*/
#include <stdio.h>
#include "mkl_types.h"
#include "mkl_spblas.h"

int main() {
//*******************************************************************************
//     Definition arrays for sparse representation of  the matrix A in
//     the compressed sparse row format:
//*******************************************************************************
#define M 5
#define NNZ 13
#define MNEW 3
		MKL_INT		m = M, nnz = NNZ, mnew = MNEW;
        double		values[NNZ]	  = {1.0, -1.0, -3.0, -2.0, 5.0, 4.0, 6.0, 4.0, -4.0, 2.0, 7.0, 8.0, -5.0};
		MKL_INT		columns[NNZ]  = {0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
		MKL_INT		rowIndex[M+1] = {0, 3,  5,  8,  11, 13};
        MKL_INT		pointerB[M] , pointerE[M];
//*******************************************************************************
//    Declaration of local variables :
//*******************************************************************************
#define N 2
        MKL_INT		n = N;
		double		sol[M][N]	= {1.0, 5.0, 1.0, 4.0, 1.0, 3.0, 1.0, 2.0, 1.0,1.0};
		double		rhs[M][N]	= {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		double		temp[M][N]	= {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		double		sol_vec[M]	= {1.0, 1.0, 1.0, 1.0, 1.0};
		double		rhs_vec[M]	= {0.0, 0.0, 0.0, 0.0, 0.0};
		double		temp_vec[M]	= {0.0, 0.0, 0.0, 0.0, 0.0};
        double		alpha = 1.0, beta = 0.0;
        MKL_INT		i, j, is;
		char		transa, uplo, nonunit;
		char		matdescra[6];

		printf("\n EXAMPLE PROGRAM FOR COMPRESSED SPARSE ROW FORMAT ROUTINES \n");
//*******************************************************************************
//Task 1.    Obtain matrix-matrix multiply (L+D)' *sol --> rhs
//    and solve triangular system   (L+D)'
//	  *temp = rhs with multiple right hand sides
//    Array temp must be equal to the array sol
//*******************************************************************************
        printf("                             \n");
        printf("   INPUT DATA FOR MKL_DCSRMM \n");
        printf("   WITH TRIANGULAR MATRIX    \n");
        printf("     M = %1.1i   N = %1.1i\n", m, n);
        printf("     ALPHA = %4.1f  BETA = %4.1f \n", alpha, beta);
        printf("     TRANS = '%c' \n", 'T');
        printf("   Input matrix              \n");
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				printf("%7.1f", sol[i][j]);
			};
			printf("\n");
		};

		transa = 't';
		matdescra[0] = 't';
		matdescra[1] = 'l';
		matdescra[2] = 'n';
		matdescra[3] = 'c';

		mkl_dcsrmm(&transa, &m, &n, &m, &alpha, matdescra, values, columns, rowIndex, &(rowIndex[1]), &(sol[0][0]), &n,  &beta, &(rhs[0][0]), &n);

		printf("                             \n");
        printf("   OUTPUT DATA FOR MKL_DCSRMM\n");
        printf("   WITH TRIANGULAR MATRIX    \n");
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				printf("%7.1f", rhs[i][j]);
			};
			printf("\n");
		};
        printf("-----------------------------------------------\n");
        printf("   Solve triangular system   \n");
        printf("   with obtained             \n");
        printf("   right hand side           \n");
		mkl_dcsrsm(&transa, &m, &n, &alpha, matdescra, values, columns, rowIndex, &(rowIndex[1]), &(rhs[0][0]), &n, &(temp[0][0]), &n);

        printf("                             \n");
        printf("   OUTPUT DATA FOR MKL_DCSRSM\n");
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				printf("%7.1f", temp[i][j]);
			};
			printf("\n");
		};
        printf("-----------------------------------------------\n");

//*******************************************************************************
// Task 2.    Obtain matrix-vector multiply (U+I)' *sol --> rhs
//    and solve triangular system   (U+I)' *temp = rhs with single
//    right hand sides. Array temp must be equal to the array sol.
//
//    Let us form the arrays pointerB and pointerE for the NIST's
//	  variation of the compressed sparse row format using the array
//    rowIndex.
//
//*******************************************************************************
		for (i = 0; i < m; i++) {
			pointerB[i] = rowIndex[i];
            pointerE[i] = rowIndex[i+1];
		};
        printf("                             \n");
        printf("   INPUT DATA FOR MKL_DCSRMV \n");
        printf("   WITH TRIANGULAR MATRIX    \n");
        printf("     ALPHA = %4.1f  BETA = %4.1f \n", alpha, beta);
        printf("     TRANS = '%c' \n", 'T');
        printf("   Input vector              \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", sol_vec[i]);
		};

		transa = 't';
		matdescra[0] = 't';
		matdescra[1] = 'u';
		matdescra[2] = 'u';
		matdescra[3] = 'c';

		mkl_dcsrmv(&transa, &m, &m, &alpha, matdescra, values, columns, pointerB, pointerE, sol_vec, &beta, rhs_vec);

		printf("                             \n");
        printf("   OUTPUT DATA FOR MKL_DCSRMV\n");
        printf("   WITH TRIANGULAR MATRIX    \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", rhs_vec[i]);
		};
        printf("-----------------------------------------------\n");
        printf("   Solve triangular system   \n");
        printf("   with obtained             \n");
        printf("   right hand side           \n");

		uplo		= 'u';
		nonunit		= 'u';
		mkl_cspblas_dcsrtrsv(&uplo, &transa, &nonunit, &m, values, rowIndex, columns, rhs_vec, temp_vec);

		printf("                             \n");
        printf("   OUTPUT DATA FOR           \n");
		printf("   MKL_CSPBLAS_DCSRTRSV      \n");
        printf("   WITH TRIANGULAR MATRIX    \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", temp_vec[i]);
		};
        printf("-----------------------------------------------\n");
//*******************************************************************************
// Task 3.   Obtain matrix-vector multiply D *sol --> rhs
//    and solve triangular system   D *temp = rhs with single right hand side
//    Array temp must be equal to the array sol
//*******************************************************************************
        printf("                             \n");
        printf("   INPUT DATA FOR MKL_DCSRMV \n");
        printf("   WITH DIAGONAL MATRIX    \n");
        printf("     M = %1.1i   N = %1.1i\n", m, n);
        printf("     TRANS = '%c' \n", 'T');
        printf("   Input vector              \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", sol_vec[i]);
		};

		transa = 'n';
		matdescra[0] = 'd';
		matdescra[1] = 'u';
		matdescra[2] = 'n';
		matdescra[3] = 'c';

        mkl_dcsrmv(&transa, &m, &m, &alpha, matdescra, values, columns, pointerB, pointerE, sol_vec, &beta, rhs_vec);
        printf("                             \n");
        printf("   OUTPUT DATA FOR MKL_DCSRMV\n");
        printf("   WITH DIAGONAL MATRIX      \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", rhs_vec[i]);
		};
        printf("-----------------------------------------------\n");
        printf("   Multiply by inverse      \n");
        printf("   matrix with the help     \n");
        printf("   of MKL_DCSRSV            \n");


        mkl_dcsrsv(&transa, &m, &alpha, matdescra, values, columns, pointerB, pointerE, rhs_vec, temp_vec);
        printf("                             \n");
        printf("   OUTPUT DATA FOR MKL_DCSRSV\n");
        printf("   WITH DIAGONAL MATRIX      \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", temp_vec[i]);
		};
        printf("-----------------------------------------------\n");

//*******************************************************************************
// Task 4.  Obtain matrix-vector multiply (U -U')*sol --> rhs
//    Array temp must be equal to the array sol
//*******************************************************************************
        printf("                             \n");
        printf("   INPUT DATA FOR MKL_DCSRMV \n");
        printf("   WITH SKEW-SYMMETRIC MATRIX\n");
        printf("     ALPHA = %4.1f  BETA = %4.1f \n", alpha, beta);
        printf("     TRANS = '%c' \n", 'N');
        printf("   Input vector              \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", sol_vec[i]);
		};

		transa = 'n';
		matdescra[0] = 'a';
		matdescra[1] = 'u';
		matdescra[2] = 'n';
		matdescra[3] = 'c';
        mkl_dcsrmv(&transa, &m, &m, &alpha, matdescra, values, columns, rowIndex, &(rowIndex[1]), sol_vec, &beta, rhs_vec);
        printf("                             \n");
        printf("   OUTPUT DATA FOR MKL_DCSRMV\n");
        printf("   WITH SKEW-SYMMETRIC MATRIX\n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", rhs_vec[i]);
		};
        printf("-----------------------------------------------\n");
//*******************************************************************************
// Task 5.    Obtain matrix-vector multiply (L+D+L')*sol --> rhs whith
//				the help of MKL_DCSRSYMV
//*******************************************************************************
        printf("                             \n");
        printf("   INPUT DATA FOR            \n");
		printf("   MKL_CSPBLAS_DCSRSYMV      \n");
        printf("   WITH SYMMETRIC MATRIX     \n");
        printf("     ALPHA = %4.1f  BETA = %4.1f \n", alpha, beta);
        printf("   Input vector              \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", sol_vec[i]);
		};
        uplo = 'l';

        mkl_cspblas_dcsrsymv(&uplo, &m, values, rowIndex, columns, sol_vec, rhs_vec);
        printf("                             \n");
        printf("   OUTPUT DATA FOR           \n");
		printf("   MKL_CSPBLAS_DCSRSYMV      \n");
        printf("   WITH SYMMETRIC MATRIX     \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", rhs_vec[i]);
		};
        printf("-----------------------------------------------\n");
//*******************************************************************************
// Task 6.   Obtain matrix-vector multiply A'*sol --> rhs whith
//				the help of MKL_DCSRGEMV
//*******************************************************************************
        printf("                             \n");
        printf("   INPUT DATA FOR            \n");
		printf("   MKL_CSPBLAS_DCSRGEMV      \n");
        printf("   WITH GENERAL MATRIX       \n");
        printf("     ALPHA = %4.1f  BETA = %4.1f \n", alpha, beta);
        printf("     TRANS = '%c' \n", 'T');
        printf("   Input vector              \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", sol_vec[i]);
		};

		transa = 't';

		mkl_cspblas_dcsrgemv(&transa, &m, values, rowIndex, columns, sol_vec, rhs_vec);
        printf("                             \n");
        printf("   OUTPUT DATA FOR           \n");
		printf("   MKL_CSPBLAS_DCSRGEMV      \n");
        printf("   WITH GENERAL MATRIX       \n");
		for (i = 0; i < m; i++) {
			printf("%7.1f\n", rhs_vec[i]);
		};
        printf("-----------------------------------------------\n");

//*******************************************************************************
// Task 7.  Obtain matrix-vector multiply T*sol --> rhs whith the help of
//	  MKL_DCSRMV where S is  3 by 3 minor of the matrix A starting with A(1,1)
//    Let's us redefine two elements of the array pointerE in order to identify
//    the needed minor. More precisely
//            pointerE(1) --> pointerE(1)-1
//            pointerE(3) --> pointerE(3)-2
//
//*******************************************************************************
        pointerE[0] = pointerE[0] - 1;
        pointerE[2] = pointerE[2] - 2;
        printf("                             \n");
        printf("   INPUT DATA FOR MKL_DCSRMV \n");
        printf("   WITH A MINOR OF GENERAL   \n");
		printf("   MATRIX                    \n");
        printf("     ALPHA = %4.1f  BETA = %4.1f \n", alpha, beta);
        printf("     TRANS = '%c' \n", 'T');
        printf("   Input vector              \n");
		for (i = 0; i < mnew; i++) {
			printf("%7.1f\n", sol_vec[i]);
		};

		transa = 'n';
		matdescra[0] = 't';
		matdescra[1] = 'l';
		matdescra[2] = 'n';
		matdescra[3] = 'c';
        mkl_dcsrmv(&transa, &mnew, &mnew, &alpha, matdescra, values, columns, pointerB, pointerE, sol_vec, &beta, rhs_vec);

        printf("                                       \n");
        printf("   OUTPUT DATA FOR MKL_DCSRMV          \n");
        printf("   WITH A MINOR OF GENERAL MATRIX      \n");
		for (i = 0; i < mnew; i++) {
			printf("%7.1f\n", rhs_vec[i]);
		};
        printf("-----------------------------------------------\n");

        printf("    Multiply by inverse to a    \n");
		printf("    minor of the matrix with    \n");
        printf("    the help of MKL_DCSRSV      \n");

        mkl_dcsrsv(&transa, &mnew, &alpha, matdescra, values, columns, pointerB, pointerE, rhs_vec, temp_vec);

        printf("                                 \n");
        printf("   OUTPUT DATA FOR MKL_DCSRSV    \n");
        printf("   WITH A MINOR OF GENERAL MATRIX\n");
		for (i = 0; i < mnew; i++) {
			printf("%7.1f\n", temp_vec[i]);
		};
        printf("-----------------------------------------------\n");
	return 0;
}

