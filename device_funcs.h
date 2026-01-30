#include <cuda_runtime.h>
#define SYNC_T __syncthreads();
#define SYNC_D cudaDeviceSynchronize();
#define BLOCK_SIZE 32
#define BLOCK_SPLIT_N 4

__device__ void dev_ker3_matmul(const double *A,
                                double A1[BLOCK_SIZE][BLOCK_SIZE],
                                double A2[BLOCK_SIZE][BLOCK_SIZE],
                                double Ares[BLOCK_SIZE][BLOCK_SIZE],
                                int m_size)
{
    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.y;

    double tmp = 0.0f;

    for (int i = 0; i < m_size / BLOCK_SIZE + (m_size % BLOCK_SIZE > 0); i++) {
        int A_col = i * BLOCK_SIZE + threadIdx.x;
        if (A_col < m_size && row < m_size) {
            A1[threadIdx.x][threadIdx.y] = A[m_size * row + A_col];

        } else {
            A1[threadIdx.x][threadIdx.y] = 0.0f;
        }

        int B_row = i * BLOCK_SIZE + threadIdx.y;
        if (B_row < m_size && col < m_size) {
            A2[threadIdx.x][threadIdx.y] = A[m_size * B_row + col];

        } else {
            A2[threadIdx.x][threadIdx.y] = 0.0f;
        }
        SYNC_T
        for (int j = 0; j < BLOCK_SIZE; j++) {
            tmp += A1[j][threadIdx.y] * A2[threadIdx.x][j];
        }
        SYNC_T
    }
    if(col < m_size && row < m_size)
    {
        Ares[threadIdx.x][threadIdx.y] = tmp;
    }

}

__device__ void dev_add_with_ks(double A[BLOCK_SIZE][BLOCK_SIZE], const double B[BLOCK_SIZE][BLOCK_SIZE], double k1, double k2, int m_size)
{
    int col = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    int row = threadIdx.y + blockIdx.y * BLOCK_SIZE;

    if(col < m_size && row < m_size)
    {
        A[threadIdx.y][threadIdx.x] = A[threadIdx.y][threadIdx.x] * k1 + B[threadIdx.y][threadIdx.x] * k2;
    }


}

