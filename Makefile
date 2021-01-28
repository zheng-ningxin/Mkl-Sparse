spmm:
	g++ spmm.cpp -o spmm -lmkl_core -lmkl_rt

gemm:
	g++  gemm.cpp -o gemm -lmkl_core -lmkl_rt 

spmm_v2:
	g++ spmm_v2.cpp -o spmm_v2 -lmkl_core -lmkl_rt

all: spmm gemm spmm_v2

clean:
	rm spmm gemm spmm_v2