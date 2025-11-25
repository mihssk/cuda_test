#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <string>
using namespace std;


#define BLOCK_SIZE 32
#define BLOCK_SPLIT_N 4
#define SYNC_T __syncthreads();
#define N 512

void print_matrix(double * m, int m_size)
{
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            printf("%.2f ", m[i * m_size + j]);
        }
        cout << endl;
    }

}

void check_matrix(const double * m1, const double * m2, int m_size, const string& nm1, const string& nm2, double precision = 0.001)
{
    bool ret = true;
    int p = 1;
    double err = 0;
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            err += m1[i * m_size + j] - m2[i * m_size + j];
            if(m1[i * m_size + j] - m2[i * m_size + j] > precision)
            {
                ret = false;
                if (p < 15)
                {
                    cout << i << " " << j << " " << m1[i * m_size + j] - m2[i * m_size + j] << "|";
                    p++;
                }
            }
        }
    }
    if (ret)
    {
        cout << "The matrices " << nm1 + " and " << nm2 << " are identical" << endl;
    }
    cout << "Error: " <<  err << endl;
}

void create_matrix(double * m, int m_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distr(1,6); // distribution in range [1, 6]

    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            m[i * m_size + j] = (double )distr(gen);
        }
    }
}

void cpu_matmul(const double * A, const double * B, double * C, int m_size)
{
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            double tmp = 0;
            for(int k = 0; k < m_size; k++)
            {
                tmp += A[m_size * i + k] * B[k * m_size + j];
            }
            C[m_size * i + j] = tmp;
        }
    }
}

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
    if (row < m_size && col < m_size) {
        for (int i = 0; i < m_size / BLOCK_SIZE + (m_size % BLOCK_SIZE > 0); i++) {
            int A_col = i * BLOCK_SIZE + threadIdx.x;
            if (row < m_size && A_col < m_size) {
                s_A[threadIdx.x][threadIdx.y] = A[m_size * row + A_col];

            } else {
                s_A[threadIdx.x][threadIdx.y] = 0;
            }

            int B_row = i * BLOCK_SIZE + threadIdx.y;
            if (row < m_size && col < m_size) {
                s_B[threadIdx.x][threadIdx.y] = B[m_size * B_row + col];

            } else {
                s_B[threadIdx.x][threadIdx.y] = 0;
            }
            SYNC_T

            for (int j = 0; j < BLOCK_SIZE; j++) {
                tmp += s_A[j][threadIdx.y] * s_B[threadIdx.x][j];
            }
            SYNC_T

        }
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
    if (row < m_size && col < m_size) {
        for (int i = 0; i < m_size / BLOCK_SIZE + (m_size % BLOCK_SIZE > 0); i++) {
            int A_col = i * BLOCK_SIZE + col;
            if (row < m_size && A_col < m_size) {
                s_A[col + row * BLOCK_SIZE] = A[m_size * (row + BLOCK_SIZE * cRow)  + A_col];

            } else {
                s_A[col + row * BLOCK_SIZE] = 0;
            }

            int B_row = i * BLOCK_SIZE + row;
            if (row < m_size && col < m_size) {
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
        C[(row + BLOCK_SIZE * cRow) * m_size + (col + BLOCK_SIZE * cCol)] = tmp;
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
        for(int split_x = 0; split_x < BLOCK_SPLIT_N; split_x++)
        {
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

        for(int split_x = 0; split_x < BLOCK_SPLIT_N; split_x++)
        {
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

        for(int k = 0; k < BLOCK_SIZE; k++)
        {
            for(int i = 0; i < BLOCK_SPLIT_N; i++)
            {
                reg_A[i] = s_A[k][threadIdx.y * BLOCK_SPLIT_N + i]; //TODO чекнуть последовательность
                reg_B[i] = s_B[threadIdx.x * BLOCK_SPLIT_N + i][k];
            }

            for(int i = 0; i < BLOCK_SPLIT_N; i++)
            {
                for(int j = 0; j < BLOCK_SPLIT_N; j++)
                {
                    reg_C[j][i] += reg_A[i] * reg_B[j];
                }
            }

        }
        SYNC_T

    }
    for(int i = 0; i < BLOCK_SPLIT_N; i++)
    {
        for(int j = 0; j < BLOCK_SPLIT_N; j++)
        {
            if ((b_row + j) < m_size && (b_col + i) < m_size)
            {
                C[b_col + i + (b_row + j) * m_size]= reg_C[i][j];
            }


        }
    }


}

int main() {


//    int N = 8;
    auto* h_A = new double [N * N];
    auto* h_B = new double [N * N];
    auto* h_C = new double [N * N];
    auto* d_C_res = new double [N * N];

    double * d_A;
    double * d_B;
    double * d_C;

    cudaMalloc(&d_A, sizeof(double ) * N * N);
    cudaMalloc(&d_B, sizeof(double ) * N * N);
    cudaMalloc(&d_C, sizeof(double ) * N * N);


    dim3 grid_size( (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    chrono::high_resolution_clock cl;
    cout << "grid: " << grid_size.x << "x" << grid_size.y << endl;
    cout << "block: " << block_size.x << "x" << block_size.y << endl;
    auto start = chrono::high_resolution_clock::now();
    create_matrix(h_A, N);
    create_matrix(h_B, N);
    cudaMemcpy(d_A, h_A, sizeof(double ) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(double ) * N * N, cudaMemcpyHostToDevice);


    {
        for(int i = 0; i < 1; i++)
        {
            cpu_matmul(h_A, h_B, h_C, N);
        }
//        print_matrix(h_C, N);


    }
    auto end = chrono::high_resolution_clock::now();
    double time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    cout << "CPU time: "<< time_elapsed / 1000000 << endl << endl;

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker1<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
        }
        cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);
//        print_matrix(d_C_res, N);

    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();


    cout << "ker1 time: " << time_elapsed / 1000000 << endl << endl;
    check_matrix(d_C_res, h_C, N, "cpu", "ker1");

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker2<<<grid_size, dim3(BLOCK_SIZE * BLOCK_SIZE)>>>(d_A, d_B, d_C, N);
        }
        cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);


//        print_matrix(d_C_res, N);

    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();


    cout << "ker2 time: " << time_elapsed / 1000000 << endl << endl;
    check_matrix(d_C_res, h_C, N, "cpu", "ker2");

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker3<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
        }
        cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);

//        print_matrix(d_C_res, N);


    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    cout << "ker3 time: " << time_elapsed / 1000000 << endl << endl;
    check_matrix(d_C_res, h_C, N, "cpu", "ker3");

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker3_v2<<<grid_size, BLOCK_SIZE * BLOCK_SIZE>>>(d_A, d_B, d_C, N);
        }
        cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);

//        print_matrix(d_C_res, N);
    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    cout << "ker3_v2 time: " << time_elapsed / 1000000 << endl << endl;
    check_matrix(d_C_res, h_C, N, "cpu", "ker3_v2");

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker4<<<grid_size, dim3(BLOCK_SIZE / BLOCK_SPLIT_N, BLOCK_SIZE / BLOCK_SPLIT_N)>>>(d_A, d_B, d_C, N);
        }
        cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);

//        print_matrix(d_C_res, N);
    }

    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    cout << "ker4 time: " << time_elapsed / 1000000 << endl << endl;
    check_matrix(d_C_res, h_C, N, "cpu", "ker4");

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] d_C_res;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}