#include "matr_methods.h"
#include "kernels.h"

class Timer {
public:
    Timer(const std::string& name = "")
            : start(std::chrono::high_resolution_clock::now()),
              timer_name(name) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << timer_name << " time: "
                  << duration.count() << " mu_s ("
                  << duration.count() / 1000.0 << " ms, "
                  << duration.count() / 1000000.0 << " sec)" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::string timer_name;
};

int main() {

    bool load = true;
    bool save = true;
    int N = 4000;
    int N_b = 17;

    string A_fname = "../bit_matrs/matr" + to_string(N) + ".bin";

    ofstream out;
    out.open("../Results/Results_matr_exp" + to_string(N) + ".txt");

    auto *h_A = new double [N * N];
    auto *h_Aexp = new double [N * N];
    auto *h_Aexp_res = new double [N * N];
    auto *h_A_ind = new double [N * N];
    double h;

    double *d_A_cumulative;
    double *d_A_cumulative2;
    double *d_A;
    double *d_A_exp;
    double *d_A_ind;

    CUDA_CHECK(cudaMalloc((void**) &d_A_cumulative, sizeof(double) * N * N));
    cudaMalloc((void**) &d_A_cumulative2, sizeof(double) * N * N);
    cudaMalloc((void**) &d_A, sizeof(double) * N * N);
    cudaMalloc((void**) &d_A_exp, sizeof(double) * N * N);
    cudaMalloc((void**) &d_A_ind, sizeof(double) * N * N);

    create_identical_matrix(h_Aexp, N);
    create_identical_matrix(h_A_ind, N);
    {
        Timer t1("save/load");
        if(load)
        {
            bin_load_matrix(h_A, N, A_fname);
        }
        else
        {
            create_random_matrix(h_A, N);
            if(save)
            {
                bin_save_matrix(h_A, N, A_fname);
            }
        }
    }


    if(N < N_b)
    {
        print_matrix(h_A, N);
    }
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 block_size_for_ker4(BLOCK_SIZE / BLOCK_SPLIT_N, BLOCK_SIZE / BLOCK_SPLIT_N);
    dim3 grid_size((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    out << "grid: " << grid_size.x << "x" << grid_size.y << endl;
    out << "block: " << block_size.x << "x" << block_size.y << endl << endl;

    {
        Timer t1("cuda_memcpy");
        cudaMemcpy(d_A, h_A, sizeof(double ) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_cumulative, h_A, sizeof(double ) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_exp, h_Aexp, sizeof(double ) * N * N, cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaMemcpy(d_A_ind, h_A_ind, sizeof(double ) * N * N, cudaMemcpyHostToDevice));
    }



    auto start = chrono::high_resolution_clock::now();
    cpu_matr_exp(h_A, h_Aexp, N);
    auto end = chrono::high_resolution_clock::now();
    double time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    out << "cpu time " << time_elapsed / 1000000 << endl;
    if(N < N_b)
    {
        cout << "CPU exp" << endl;
        print_matrix(h_Aexp, N);
    }


    SYNC_D

    start = chrono::high_resolution_clock::now();
    {
        Timer t1("ker_add");
        ker_add_with_ks<<<grid_size, block_size>>>(d_A_exp, d_A_cumulative, 1, 1, N);
    }
    {
        Timer t1("ker4");
        ker4<<<grid_size, block_size_for_ker4>>>(d_A, d_A_cumulative, d_A_cumulative2, N);
    }

    ker_add_with_ks<<<grid_size, block_size>>>(d_A_exp, d_A_cumulative2, 1, 1.0/2, N);

    ker4<<<grid_size, block_size_for_ker4>>>(d_A, d_A_cumulative2, d_A_cumulative, N);
    ker_add_with_ks<<<grid_size, block_size>>>(d_A_exp, d_A_cumulative, 1, 1.0/6, N);

    ker4<<<grid_size, block_size_for_ker4>>>(d_A, d_A_cumulative, d_A_cumulative2, N);
    ker_add_with_ks<<<grid_size, block_size>>>(d_A_exp, d_A_cumulative2, 1, 1.0/24, N);

    SYNC_D
    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    out << "gpu time " << time_elapsed / 1000000 << endl;

    {
        Timer t1("cudaMemcpy dev_to_host");
        cudaMemcpy(h_Aexp_res, d_A_exp, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
    }
    if(N < N_b)
    {
        cout << "GPU exp" << endl;
        print_matrix(h_Aexp_res, N);
    }
    check_matrix(h_Aexp, h_Aexp_res, N, "cpu_exp", "gpu_exp", out);

    cudaMemcpy(d_A_cumulative, h_A, sizeof(double ) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_exp, h_A_ind, sizeof(double ) * N * N, cudaMemcpyHostToDevice);

    SYNC_D
    start = chrono::high_resolution_clock::now();
    //k1 add
    ker_add_with_ks<<<grid_size, block_size>>>(d_A_exp, d_A_cumulative, 1, 1.0/6, N);

    // k2 eval
    ker_add_with_ks<<<grid_size, block_size>>>( d_A_cumulative, d_A_ind, 1.0/2, 1, N);
    ker4<<<grid_size, block_size_for_ker4>>>(d_A, d_A_cumulative, d_A_cumulative2, N);
    // k2 add
    ker_add_with_ks<<<grid_size, block_size>>>(d_A_exp, d_A_cumulative2, 1, 1.0/3, N);

    // k3 eval
    ker_add_with_ks<<<grid_size, block_size>>>( d_A_cumulative2, d_A_ind, 1.0/2, 1, N);
    ker4<<<grid_size, block_size_for_ker4>>>(d_A, d_A_cumulative2, d_A_cumulative, N);
    // k3 add
    ker_add_with_ks<<<grid_size, block_size>>>(d_A_exp, d_A_cumulative, 1, 1.0/3, N);

    // k4 eval
    ker_add_with_ks<<<grid_size, block_size>>>(d_A_cumulative, d_A_ind, 1, 1, N);
    ker4<<<grid_size, block_size_for_ker4>>>(d_A, d_A_cumulative, d_A_cumulative2, N);
    // k4 add
    ker_add_with_ks<<<grid_size, block_size>>>(d_A_exp, d_A_cumulative2, 1, 1.0/6, N);
    SYNC_D

    end = chrono::high_resolution_clock::now();
    time_elapsed = (double )chrono::duration_cast<chrono::microseconds >(end - start).count();
    out << "gpu runge time " << time_elapsed / 1000000 << endl;

    cudaMemcpy(h_Aexp_res, d_A_exp, sizeof(double)* N * N, cudaMemcpyDeviceToHost);
    if(N < N_b)
    {
        cout << "GPU exp runge" << endl;
        print_matrix(h_Aexp_res, N);
    }
    check_matrix(h_Aexp, h_Aexp_res, N, "cpu_exp", "gpu_exp_runge", out);




    delete[] h_A;
    delete[] h_Aexp;
    delete[] h_A_ind;
    delete[] h_Aexp_res;
    cudaFree(d_A_cumulative);
    cudaFree(d_A_cumulative2);
    cudaFree(d_A);
    cudaFree(d_A_exp);
    cudaFree(d_A_ind);
    out.close();

    return 0;
}