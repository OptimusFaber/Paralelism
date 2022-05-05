#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cub/cub.cuh>
#define N 1024


__global__ void first(double* u, double* up, int n) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i > 0 && i < n-1) && (j > 0 && j < n-1))
        up[i*n + j] = 0.25 * (u[i*n + j - 1] + u[i*n + j + 1] + u[(i - 1)*n + j] + u[(i + 1)*n + j]);
}


__global__ void second(double* u, double* up, double* arr, int n) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= 0 && i < n) && (j >= 0 && j < n))
        arr[i*n + j] = up[i*n + j] - u[i*n + j];
}


int main() {

    double* u = (double*)calloc(N*N, sizeof(double));
    double* up = (double*)calloc(N*N, sizeof(double));
    
    double x1 = 10.0;
    double x2 = 20.0;
    double y1 = 20.0;
    double y2 = 30.0;
    u[0] = up[0] = x1;
    u[N] = up[N] = x2;
    u[N * (N - 1) + 1] = up[N * (N - 1) + 1] = y1;
    u[N * N] = up[N * N] = y2;

    double step1 = 10.0/(N-1);

    for (int i = 1; i < N-1; i++) {
        u[i*N] = up[i*N] = x1 + i * step1;
        u[i] = up[i] = x1 + i * step1;
        u[(N - 1) * N + i] = up[(N - 1) * N + i] = y1 + i * step1;
        u[i * N + (N - 1)] = up[i * N + (N - 1)] = x2 + i * step1;
    }

    double* ux;
    double* upx;
    double* arrx;
        
    cudaMalloc(&ux, sizeof(double)*N*N);
    cudaMalloc(&upx, sizeof(double)*N*N);
    cudaMalloc(&arrx, sizeof(double)*N*N);
    
    cudaMemcpy(ux, u, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(upx, up, N*N*sizeof(double), cudaMemcpyHostToDevice);

    dim3 BLOCK_SIZE = dim3(32, 32);
    dim3 GRID_SIZE = dim3(ceil(N/32.), ceil(N/32.));

    int itter = 0;
    double* error = (double*)calloc(1,sizeof(double));
    *error = 1;
    double* errx;
    cudaMalloc(&errx, sizeof(double));
    void* store = NULL;
    size_t bytes = 0;

    cub::DeviceReduce::Max(store, bytes, arrx, errx, N*N);
    cudaMalloc(&store, bytes);

    itter = 0;
    int graphitt = 1000000/100;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bool flag = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    while(*error > 1e-6 && itter < graphitt)
    {	
	itter += 2;

        if(!flag)
        {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for(int i = 0; i < 100; i ++)
            {
                first<<<GRID_SIZE,BLOCK_SIZE,0,stream>>>(upx, ux, N);
                first<<<GRID_SIZE, BLOCK_SIZE,0,stream>>>(ux, upx, N);
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

            flag=true;
        }
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);

        printf("%d %e\n", itter*100, *error);
        *error = 0;
        double* buf = ux;
        ux = upx;
        upx = buf;

        second<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(upx, ux, arrx, N);
        cub::DeviceReduce::Max(store, bytes, arrx, errx, N*N, stream);
        cudaMemcpyAsync(error, errx, sizeof(double), cudaMemcpyDeviceToHost, stream);

        buf = ux;
        ux = upx;
        upx = buf;

        cudaStreamSynchronize(stream);
    }

    free(u);
    free(up);
    cudaFree(ux);
    cudaFree(upx);
    cudaFree(arrx);

    printf("%d %lf\n", itter, *error);

    return 0;

}
