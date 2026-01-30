#include "device_funcs.h"

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)


__global__ void ker1(const double * A, const double * B, double * C, int m_size)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(col < m_size && row < m_size)
    {

        double tmp = 0;
        for(int k = 0; k < m_size; k++)
        {
            tmp += A[row * m_size + k] * B[k * m_size + col];
        }
        C[m_size * row + col] = tmp;

    }


}

__global__ void ker2(const double * A, const double * B, double * C, int m_size) {
    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x / BLOCK_SIZE;
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.x % BLOCK_SIZE;

    if(col < m_size && row < m_size)
    {

        double tmp = 0;
        for(int k = 0; k < m_size; k++)
        {
            tmp += A[row * m_size + k] * B[k * m_size + col];
        }
        C[m_size * row + col] = tmp;

    }

}

__global__ void ker3(const double * A, const double * B, double * C, int m_size) {
    __shared__ double s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_B[BLOCK_SIZE][BLOCK_SIZE];

    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.y;

    double tmp = 0.0f;

    for (int i = 0; i < m_size / BLOCK_SIZE + (m_size % BLOCK_SIZE > 0); i++) {
        int A_col = i * BLOCK_SIZE + threadIdx.x;
        if (A_col < m_size && row < m_size) {
            s_A[threadIdx.x][threadIdx.y] = A[m_size * row + A_col];

        } else {
            s_A[threadIdx.x][threadIdx.y] = 0.0f;
        }

        int B_row = i * BLOCK_SIZE + threadIdx.y;
        if (B_row < m_size && col < m_size) {
            s_B[threadIdx.x][threadIdx.y] = B[m_size * B_row + col];

        } else {
            s_B[threadIdx.x][threadIdx.y] = 0.0f;
        }
        SYNC_T
        for (int j = 0; j < BLOCK_SIZE; j++) {
            tmp += s_A[j][threadIdx.y] * s_B[threadIdx.x][j];
        }
        SYNC_T
    }
    if(col < m_size && row < m_size)
    {
        C[row * m_size + col] = tmp;
    }

}

__global__ void ker3_v2(const double * A, const double * B, double * C, int m_size) {
    __shared__ double s_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ double s_B[BLOCK_SIZE * BLOCK_SIZE];

    int col = threadIdx.x % BLOCK_SIZE;
    int row = threadIdx.x / BLOCK_SIZE;

    int cRow = blockIdx.x;
    int cCol = blockIdx.y;

    double tmp = 0.0f;


    for (int i = 0; i < m_size / BLOCK_SIZE + (m_size % BLOCK_SIZE > 0); i++) {
        int A_col = i * BLOCK_SIZE + col;
        if (row + BLOCK_SIZE * cRow < m_size && A_col < m_size) {
            s_A[col + row * BLOCK_SIZE] = A[m_size * (row + BLOCK_SIZE * cRow)  + A_col];

        } else {
            s_A[col + row * BLOCK_SIZE] = 0;
        }
        SYNC_T
        int B_row = i * BLOCK_SIZE + row;
        if (B_row < m_size && (col + BLOCK_SIZE * cCol) < m_size) {
            s_B[col + BLOCK_SIZE * row] = B[m_size * B_row + (col + BLOCK_SIZE * cCol)];

        } else {
            s_B[col + BLOCK_SIZE * row] = 0;
        }
        SYNC_T

        for (int j = 0; j < BLOCK_SIZE; j++) {
            tmp += s_A[j + row * BLOCK_SIZE] * s_B[col + j * BLOCK_SIZE];
        }
        SYNC_T

    }
    if(col + BLOCK_SIZE * cCol < m_size && row + BLOCK_SIZE * cRow < m_size)
    {
        C[m_size * (row + BLOCK_SIZE * cRow) + (col + BLOCK_SIZE * cCol)] = tmp;
    }


}

__global__ void ker4(const double * A, const double * B, double * C, int m_size)
{
    __shared__ double s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_B[BLOCK_SIZE][BLOCK_SIZE];

    double reg_C[BLOCK_SPLIT_N][BLOCK_SPLIT_N] = {0.0f};
    double reg_A[BLOCK_SPLIT_N];
    double reg_B[BLOCK_SPLIT_N];

    int b_col = blockIdx.x * BLOCK_SIZE + threadIdx.x * BLOCK_SPLIT_N;
    int b_row = blockIdx.y * BLOCK_SIZE + threadIdx.y * BLOCK_SPLIT_N;
    for(int tile_i = 0; tile_i < m_size / BLOCK_SIZE + (m_size % BLOCK_SIZE > 0); tile_i++)
    {
#pragma unroll
        for(int split_x = 0; split_x < BLOCK_SPLIT_N; split_x++)
        {
#pragma unroll
            for(int split_y = 0; split_y < BLOCK_SPLIT_N; split_y++)
            {
                int g_col = threadIdx.x * BLOCK_SPLIT_N + split_x + tile_i * BLOCK_SIZE;
                int g_row = (b_row + split_y);
                if (g_col < m_size && g_row < m_size)
                {
                    s_A[threadIdx.x * BLOCK_SPLIT_N + split_x][threadIdx.y * BLOCK_SPLIT_N + split_y] = A[g_col + m_size * g_row];
                }
                else
                {
                    s_A[threadIdx.x * BLOCK_SPLIT_N + split_x][threadIdx.y * BLOCK_SPLIT_N + split_y] = 0;
                }

            }

        }
#pragma unroll
        for(int split_x = 0; split_x < BLOCK_SPLIT_N; split_x++)
        {
#pragma unroll
            for(int split_y = 0; split_y < BLOCK_SPLIT_N; split_y++)
            {
                int g_col = (b_col + split_x);
                int g_row = threadIdx.y * BLOCK_SPLIT_N + split_y + tile_i * BLOCK_SIZE;
                if (g_col < m_size && g_row < m_size)
                {
                    s_B[threadIdx.x * BLOCK_SPLIT_N + split_x][threadIdx.y * BLOCK_SPLIT_N + split_y] = B[g_col + m_size * g_row];
                }
                else
                {
                    s_B[threadIdx.x * BLOCK_SPLIT_N + split_x][threadIdx.y * BLOCK_SPLIT_N + split_y] = 0;
                }

            }


        }
        SYNC_T
#pragma unroll
        for(int k = 0; k < BLOCK_SIZE; k++)
        {
#pragma unroll
            for(int i = 0; i < BLOCK_SPLIT_N; i++)
            {
                reg_A[i] = s_A[k][threadIdx.y * BLOCK_SPLIT_N + i];
                reg_B[i] = s_B[threadIdx.x * BLOCK_SPLIT_N + i][k];
            }
#pragma unroll
            for(int i = 0; i < BLOCK_SPLIT_N; i++)
            {
#pragma unroll
                for(int j = 0; j < BLOCK_SPLIT_N; j++)
                {
                    reg_C[j][i] += reg_A[i] * reg_B[j];
                }
            }

        }
        SYNC_T

    }
#pragma unroll
    for(int i = 0; i < BLOCK_SPLIT_N; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCK_SPLIT_N; j++)
        {
            if ((b_row + j) < m_size && (b_col + i) < m_size)
            {
                C[b_col + i + (b_row + j) * m_size]= reg_C[i][j];
            }
        }
    }


}

__global__ void ker4_v2(const double * A, const double * B, double * C, int m_size)
{
    __shared__ double s_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ double s_B[BLOCK_SIZE * BLOCK_SIZE];

    double reg_C[BLOCK_SPLIT_N * BLOCK_SPLIT_N] = {0.0f};
    double reg_A[BLOCK_SPLIT_N];
    double reg_B[BLOCK_SPLIT_N];

    int t_ind_x = threadIdx.x % BLOCK_SPLIT_N * BLOCK_SPLIT_N;
    int t_ind_y = threadIdx.x / BLOCK_SPLIT_N * BLOCK_SPLIT_N;
    int b_col = blockIdx.x * BLOCK_SIZE + t_ind_x;
    int b_row = blockIdx.y * BLOCK_SIZE + t_ind_y;

    for(int tile_i = 0; tile_i < m_size / BLOCK_SIZE + (m_size % BLOCK_SIZE > 0); tile_i++)
    {
#pragma unroll
        for(int split_x = 0; split_x < BLOCK_SPLIT_N; split_x++)
        {
#pragma unroll
            for(int split_y = 0; split_y < BLOCK_SPLIT_N; split_y++)
            {
                int g_col = t_ind_x + split_x + tile_i * BLOCK_SIZE;
                int g_row = (b_row + split_y);
                if (g_col < m_size && g_row < m_size)
                {
                    s_A[t_ind_x + split_x + BLOCK_SIZE * (t_ind_y + split_y)] = A[g_col + m_size * g_row];
                }
                else
                {
                    s_A[t_ind_x + split_x + BLOCK_SIZE * (t_ind_y + split_y)] = 0;
                }

            }

        }
#pragma unroll
        for(int split_x = 0; split_x < BLOCK_SPLIT_N; split_x++)
        {
#pragma unroll
            for(int split_y = 0; split_y < BLOCK_SPLIT_N; split_y++)
            {
                int g_col = (b_col + split_x);
                int g_row = t_ind_y + split_y + tile_i * BLOCK_SIZE;
                if (g_col < m_size && g_row < m_size)
                {
                    s_B[t_ind_x + split_x + BLOCK_SIZE * (t_ind_y + split_y)] = B[g_col + m_size * g_row];
                }
                else
                {
                    s_B[t_ind_x  + split_x + BLOCK_SIZE * (t_ind_y + split_y)] = 0;
                }

            }


        }
        SYNC_T
#pragma unroll
        for(int k = 0; k < BLOCK_SIZE; k++)
        {
#pragma unroll
            for(int i = 0; i < BLOCK_SPLIT_N; i++)
            {
                reg_A[i] = s_A[k + BLOCK_SPLIT_N * t_ind_y+ i];
                reg_B[i] = s_B[t_ind_x + i + k * BLOCK_SPLIT_N];
            }
#pragma unroll
            for(int i = 0; i < BLOCK_SPLIT_N; i++)
            {
#pragma unroll
                for(int j = 0; j < BLOCK_SPLIT_N; j++)
                {
                    reg_C[j + i * BLOCK_SPLIT_N] += reg_A[i] * reg_B[j];
                }
            }

        }
        SYNC_T

    }
#pragma unroll
    for(int i = 0; i < BLOCK_SPLIT_N; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCK_SPLIT_N; j++)
        {
            if ((b_row + j) < m_size && (b_col + i) < m_size)
            {
                C[b_col + i + (b_row + j) * m_size]= reg_C[i + BLOCK_SPLIT_N * j];
            }


        }
    }

}

__global__ void ker_add_with_ks(double * A, const double * B, double k1, double k2, int m_size)
{
    int col = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    int row = threadIdx.y + blockIdx.y * BLOCK_SIZE;

    if(col < m_size && row < m_size)
    {
        A[row * m_size + col] = A[row * m_size + col] * k1 + B[row * m_size + col] * k2;
    }


}

__global__ void Taylor_exp_ker(const double * A, double * B, int m_size)
{
    __shared__ double A_copy[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double A_t1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double A_t2[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double A_exp[BLOCK_SIZE][BLOCK_SIZE];

    int col = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    int row = threadIdx.y + blockIdx.y * BLOCK_SIZE;

    if(col < m_size && row < m_size)
    {
        A_t1[threadIdx.y][threadIdx.x] = A[m_size * row + col];
        A_copy[threadIdx.y][threadIdx.x] = A[m_size * row + col];
        A_exp[threadIdx.y][threadIdx.x] = B[m_size * row + col];
    }
    else
    {
        A_t1[threadIdx.y][threadIdx.x] = 0;
        A_exp[threadIdx.y][threadIdx.x] = 0;
    }

    SYNC_T

    dev_add_with_ks(A_exp, A_t1, 1, 1, m_size);
    dev_ker3_matmul(A, A_copy, A_t1, A_t2, m_size);


}


