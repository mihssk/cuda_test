#include "kernels.h"
#include "matr_methods.h"

int main() {

    int N = 4000;
    int N_b = 17;
    bool load = true;
    bool save = true;

    ofstream out;
    out.open("../Results/Results" + to_string(N) + ".txt");

    string A_fname = "../bit_matrs/matrA" + to_string(N) + ".bin";
    string B_fname = "../bit_matrs/matrB" + to_string(N) + ".bin";


    out << "Begin check matmul for " + to_string(N) + "x" + to_string(N) + " matrices" << endl;
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

    out << "grid: " << grid_size.x << "x" << grid_size.y << endl;
    out << "block: " << block_size.x << "x" << block_size.y << endl << endl;
    auto startf = chrono::high_resolution_clock::now();

    if(load)
    {
        bin_load_matrix(h_A, N, A_fname);
        bin_load_matrix(h_B, N, B_fname);

    }
    else
    {
        create_random_matrix(h_A, N);
        create_random_matrix(h_B, N);
        if(save)
        {
            bin_save_matrix(h_A, N, A_fname);
            bin_save_matrix(h_B, N, B_fname);
        }
    }
    auto endf = chrono::high_resolution_clock::now();
    double time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(endf - startf).count();
    cout << "save/load time " << time_elapsed / 1000000 << endl;



    if(N < N_b) {
        cout << "A" << endl;
        print_matrix(h_A, N);
        cout << "B" << endl;
        print_matrix(h_B, N);
    }
    startf = chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, sizeof(double ) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(double ) * N * N, cudaMemcpyHostToDevice);
    endf = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(endf - startf).count();
    cout << "cuda_memcpy to device time " << time_elapsed / 1000000 << endl;

    auto start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 1; i++)
        {
            cpu_matmul(h_A, h_B, h_C, N);
        }

    }
    auto end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    out << "CPU time: "<< time_elapsed / 1000000 << endl << endl;
    if(N < N_b)
    {
        cout << "CPU C" << endl;
        print_matrix(h_C, N);
    }
    SYNC_D
    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker1<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
        }
    }
    end = chrono::high_resolution_clock::now();
    SYNC_D

    startf = chrono::high_resolution_clock::now();
    cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);
    endf = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(endf - startf).count();
    cout << "cuda_memcpy to host time: " << time_elapsed / 1000000 << endl;

    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    out << "ker1 time: " << time_elapsed / 1000000 << endl;
    if(N < N_b)
    {
        cout << "Ker1 C" << endl;

        print_matrix(d_C_res, N);
    }
    startf = chrono::high_resolution_clock::now();
    check_matrix(d_C_res, h_C, N, "cpu", "ker1", out);
    endf = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(endf - startf).count();
    cout << "check time: " << time_elapsed / 1000000 << endl;

    SYNC_D
    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker2<<<grid_size, dim3(BLOCK_SIZE * BLOCK_SIZE)>>>(d_A, d_B, d_C, N);
        }
    }
    end = chrono::high_resolution_clock::now();
    cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    out << "ker2 time: " << time_elapsed / 1000000 << endl;
    if(N < N_b)
    {
        cout << "Ker2 C" << endl;
        print_matrix(d_C_res, N);
    }
    check_matrix(d_C_res, h_C, N, "cpu", "ker2", out);
    SYNC_D

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker3<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
        }
    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);
    out << "ker3 time: " << time_elapsed / 1000000 << endl;
    check_matrix(d_C_res, h_C, N, "cpu", "ker3", out);
    if(N < N_b)
    {
        cout << "Ker3 C" << endl;
        print_matrix(d_C_res, N);
    }
    SYNC_D

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker3_v2<<<grid_size, BLOCK_SIZE * BLOCK_SIZE>>>(d_A, d_B, d_C, N);
        }
    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);
    out << "ker3_v2 time: " << time_elapsed / 1000000 << endl;
    check_matrix(d_C_res, h_C, N, "cpu", "ker3_v2", out);
    if(N < N_b)
    {
        cout << "Ker3_v2 C" << endl;
        print_matrix(d_C_res, N);
    }
    SYNC_D

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker4<<<grid_size, dim3(BLOCK_SIZE / BLOCK_SPLIT_N, BLOCK_SIZE / BLOCK_SPLIT_N)>>>(d_A, d_B, d_C, N);
        }
    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);
    out << "ker4 time: " << time_elapsed / 1000000 << endl;
    check_matrix(d_C_res, h_C, N, "cpu", "ker4", out);
    if(N < N_b)
    {
        cout << "Ker4 C" << endl;
        print_matrix(d_C_res, N);
    }
    SYNC_D

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            ker4_v2<<<grid_size, dim3(BLOCK_SIZE / BLOCK_SPLIT_N * BLOCK_SIZE / BLOCK_SPLIT_N)>>>(d_A, d_B, d_C, N);
        }
    }
    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    cudaMemcpy(d_C_res, d_C, sizeof(double ) * N * N, cudaMemcpyDeviceToHost);
    out << "ker4_v2 time: " << time_elapsed / 1000000 << endl;
    check_matrix(d_C_res, h_C, N, "cpu", "ker4_v2", out);
    if(N < N_b)
    {
        cout << "Ker4_v2 C" << endl;
        print_matrix(d_C_res, N);
    }
    SYNC_D

    out << "End check matmul for " + to_string(N) + "x" + to_string(N) + " matrices" << endl;

    out.close();
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] d_C_res;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);



    return 0;
}