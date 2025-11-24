#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
using namespace std;


#define BLOCK_SIZE 16
#define THREAD_TILE_M 4
#define THREAD_TILE_N 4

void print_matrix(float* m, int m_size)
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

void check_matrix(const float* m1, const float* m2, int m_size, float precision = 0.0001)
{
    bool ret = true;
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            if(m1[i * m_size + j] - m2[i * m_size + j] > precision)
            {
                ret = false;
            }
        }
    }
    if (ret)
    {
        cout << "The matrices are identical";
    }
    cout << endl;
}

void create_matrix(float* m, int m_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distr(1,6); // distribution in range [1, 6]

    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            m[i * m_size + j] = (float)distr(gen);
        }
    }
}

void cpu_matmul(const float* A, const float* B, float* C, int m_size)
{
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            float tmp = 0;
            for(int k = 0; k < m_size; k++)
            {
                tmp += A[m_size * i + k] * B[k * m_size + j];
            }
            cout << m_size / BLOCK_SIZE + (m_size % BLOCK_SIZE > 0);
            C[m_size * i + j] = tmp;
        }
    }
}

__global__ void ker1(const float* A, const float* B, float* C, int m_size)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(col < m_size && row < m_size)
    {

        float tmp = 0;
        for(int k = 0; k < m_size; k++)
        {
            tmp += A[row * m_size + k] * B[k * m_size + col];
        }
        C[m_size * row + col] = tmp;

    }


}

__global__ void ker2(const float* A, const float* B, float* C, int m_size) {
    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x / BLOCK_SIZE;
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.x % BLOCK_SIZE;

    if(col < m_size && row < m_size)
    {

        float tmp = 0;
        for(int k = 0; k < m_size; k++)
        {
            tmp += A[row * m_size + k] * B[k * m_size + col];
        }
        C[m_size * row + col] = tmp;

    }

}

__global__ void ker3(const float* A, const float* B, float* C, int m_size) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int col = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int row = BLOCK_SIZE * blockIdx.y + threadIdx.y;

    float tmp = 0;

    for(int i = 0; i < m_size / BLOCK_SIZE + (m_size % BLOCK_SIZE > 0); i++)
    {
        int A_col = i * BLOCK_SIZE + threadIdx.x;
        if (row < m_size && A_col < m_size)
        {
            s_A[threadIdx.x][threadIdx.y] = A[m_size * row + A_col];

        }
        else
        {
            s_A[threadIdx.x][threadIdx.y] = 0;
        }

        int B_row = i * BLOCK_SIZE + threadIdx.y;
        if (B_row < m_size && col < m_size)
        {
            s_B[threadIdx.x][threadIdx.y] = B[m_size * B_row + col];

        }
        else
        {
            s_B[threadIdx.x][threadIdx.y] = 0;
        }

        for(int j = 0; j < BLOCK_SIZE; j++)
        {
            tmp = s_A[j][threadIdx.y] * s_B[threadIdx.x][j];
        }

    }

    C[row * m_size + col] = tmp;


}


int main() {


    int N = 4;
    auto* h_A = new float[N * N];
    auto* h_B = new float[N * N];
    auto* h_C = new float[N * N];
    auto* d_C_res = new float[N * N];

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, sizeof(float) * N * N);
    cudaMalloc(&d_B, sizeof(float) * N * N);
    cudaMalloc(&d_C, sizeof(float) * N * N);


    dim3 grid_size( (N + BLOCK_SIZE) / BLOCK_SIZE, (N + BLOCK_SIZE) / BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    chrono::high_resolution_clock cl;

    auto start = chrono::high_resolution_clock::now();
    create_matrix(h_A, N);
    create_matrix(h_B, N);
    cudaMemcpy(d_A, h_A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * N * N, cudaMemcpyHostToDevice);


    {
        for(int i = 0; i < 1; i++)
        {
            cpu_matmul(h_A, h_B, h_C, N);
        }
        print_matrix(h_C, N);


    }
    auto end = chrono::high_resolution_clock::now();
    float time_elapsed = (float)chrono::duration_cast<chrono::microseconds >(end - start).count();
    cout << "CPU time: "<< time_elapsed << endl << endl;

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker1<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
        }
        cudaMemcpy(d_C_res, d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        print_matrix(d_C_res, N);

    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (float)chrono::duration_cast<chrono::microseconds >(end - start).count();
    cout << "ker1 time: " << time_elapsed << endl << endl;

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker2<<<grid_size, dim3(BLOCK_SIZE * BLOCK_SIZE)>>>(d_A, d_B, d_C, N);
        }
        cudaMemcpy(d_C_res, d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        print_matrix(d_C_res, N);


    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (float)chrono::duration_cast<chrono::microseconds >(end - start).count();
    cout << "ker2 time: " << time_elapsed << endl << endl;


    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker3<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
        }
        cudaMemcpy(d_C_res, d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
        print_matrix(d_C_res, N);


    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (float)chrono::duration_cast<chrono::microseconds >(end - start).count();
    cout << "ker3 time: " << time_elapsed << endl << endl;

    cudaMemcpy(d_C_res, d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

//    check_matrix(d_C_res, h_C, N);



    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] d_C_res;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}