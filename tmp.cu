#include "kernels.h"
#include "matr_methods.h"
#include "myBLASXT/include/myBLAS.h"

int main() {

    size_t N = 333;
    int N_b = 17;

    ofstream out;
    out.open("../Results/Results" + to_string(N) + ".txt");

    string A_fname = "../bit_matrs/matrA" + to_string(N) + ".bin";
    string B_fname = "../bit_matrs/matrB" + to_string(N) + ".bin";
    bool load = true;
    bool save = true;

    out << "Begin check matmul for " + to_string(N) + "x" + to_string(N) + " matrices" << endl;
    auto *h_A = new float[N * N];
    auto *h_B = new float[N * N];
    auto *h_C = new float[N * N];
    auto *d_C_res = new float[N * N];

//    float *d_A;
//    float *d_B;
//    float *d_C;
//
//    cudaMalloc(&d_A, sizeof(float) * N * N);
//    cudaMalloc(&d_B, sizeof(float) * N * N);
//    cudaMalloc(&d_C, sizeof(float) * N * N);
//
//    dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
//    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
//
//    out << "grid: " << grid_size.x << "x" << grid_size.y << endl;
//    out << "block: " << block_size.x << "x" << block_size.y << endl << endl;
    auto startf = chrono::high_resolution_clock::now();

    if (load) {
        bin_load_matrix(h_A, N, A_fname);
        bin_load_matrix(h_B, N, B_fname);

    } else {
        create_random_matrix(h_A, N);
        create_random_matrix(h_B, N);
        if (save) {
            bin_save_matrix(h_A, N, A_fname);
            bin_save_matrix(h_B, N, B_fname);
        }
    }
    auto endf = chrono::high_resolution_clock::now();
    float time_elapsed = (float) chrono::duration_cast<chrono::microseconds>(endf - startf).count();
    cout << "save/load time " << time_elapsed / 1000000 << endl;


    if (N < N_b) {
        cout << "A" << endl;
        print_matrix(h_A, N);
        cout << "B" << endl;
        print_matrix(h_B, N);
    }
//    startf = chrono::high_resolution_clock::now();
//    cudaMemcpy(d_A, h_A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B, h_B, sizeof(float) * N * N, cudaMemcpyHostToDevice);
//    endf = chrono::high_resolution_clock::now();
//    time_elapsed = (float) chrono::duration_cast<chrono::microseconds>(endf - startf).count();
//    cout << "cuda_memcpy to device time " << time_elapsed / 1000000 << endl;

    auto start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 1; i++)
        {
            cpu_matmul(h_A, h_B, h_C, N);
        }

    }
    auto end = chrono::high_resolution_clock::now();
    time_elapsed = (float )chrono::duration_cast<chrono::microseconds >(end - start).count();
    out << "CPU time: "<< time_elapsed / 1000000 << endl << endl;
    if(N < N_b)
    {
        cout << "CPU C" << endl;
        print_matrix(h_C, N);
    }

    start = chrono::high_resolution_clock::now();
    {
        for(int i = 0; i < 20; i++)
        {
            multiGPUsGEMM(N, N, N, h_A, h_B, d_C_res, 0, 2, 1, 0);
        }
    }
    end = chrono::high_resolution_clock::now();
//    startf = chrono::high_resolution_clock::now();
//    cudaMemcpy(d_C_res, d_C, sizeof(float ) * N * N, cudaMemcpyDeviceToHost);
//    endf = chrono::high_resolution_clock::now();
//    time_elapsed = (float )chrono::duration_cast<chrono::microseconds >(endf - startf).count();
//    cout << "cuda_memcpy to host time: " << time_elapsed / 1000000 << endl;

    time_elapsed = (float )chrono::duration_cast<chrono::microseconds >(end - start).count();
    out << "ker1 time: " << time_elapsed / 1000000 << endl;
    if(N < N_b)
    {
        cout << "Ker1 C" << endl;

        print_matrix(d_C_res, N);
    }
    startf = chrono::high_resolution_clock::now();
    check_matrix(d_C_res, h_C, N, "cpu", "ker1", out);
    endf = chrono::high_resolution_clock::now();
    time_elapsed = (float )chrono::duration_cast<chrono::microseconds >(endf - startf).count();
    cout << "check time: " << time_elapsed / 1000000 << endl;

}