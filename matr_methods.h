#include <iostream>
#include <chrono>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>

using namespace std;

template<typename T>
void print_matrix(T * m, int m_size)
{
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            cout << setprecision(4) << to_string(m[i * m_size + j]) << " ";
        }
        cout << endl;
    }

}

template<typename T>
void check_matrix(const T * m1, const T * m2, int m_size, const string& nm1, const string& nm2, ofstream& f, T precision = 0.0001)
{
    bool ret = true;
    int p = 1;
    double err = 0;
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            err += m1[i * m_size + j] - m2[i * m_size + j];
            if(abs(m1[i * m_size + j] - m2[i * m_size + j]) > precision)
            {
                ret = false;
                if (p < 15)
                {
                    f << i << " " << j << " " << m1[i * m_size + j] - m2[i * m_size + j] << "|";
                    p++;
                }
            }
        }
    }
    if (ret)
    {
        f << "The matrices " << nm1 + " and " << nm2 << " are identical" << endl;
    }
    else
    {
        f << endl << "ERROR! The matrices " << nm1 + " and " << nm2 << " are not identical" << endl;
    }

    f << "Error: " <<  err << endl << endl;
}

template<typename T>
void create_random_matrix(T * m, int m_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distr(1,2);

    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            m[i * m_size + j] = (T )distr(gen);
        }
    }


}

template<typename T>
void create_identical_matrix(T * m, int m_size)
{
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            if(i == j)
            {
                m[i * m_size + j] = 1;

            }
            else
            {
                m[i * m_size + j] = 0;

            }

        }

    }
}

template<typename T>
void cpu_matmul(const T * A, const T * B, T * C, int m_size)
{
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            T tmp = 0;
            for(int k = 0; k < m_size; k++)
            {
                tmp += A[m_size * i + k] * B[k * m_size + j];
            }
            C[m_size * i + j] = tmp;
        }
    }
}

template<typename T>
void bin_save_matrix(T *m, int m_size, const string& fname)
{
    ofstream save_f(fname, ios::binary);
    save_f.write(reinterpret_cast<char*>(m), m_size * m_size * sizeof(T));
    save_f.close();
}

template<typename T>
void bin_load_matrix(T *m, int m_size, const string& fname)
{
    ifstream save_f(fname, ios::binary);
    save_f.read(reinterpret_cast<char*>(m), m_size * m_size * sizeof(T));
    save_f.close();
}

void cpu_matr_exp(const double * A, double * C, int m_size)
{
    int power = 1;
    int denum = 1;
    auto *tmpA = new double [m_size * m_size];
    auto *tmpA2 = new double [m_size * m_size];

    // Create I and exp = I
    create_identical_matrix(C, m_size);
    // Copy of A
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            tmpA[i * m_size + j] = A[i * m_size + j];
        }
    }

    // exp = I + A
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            C[i * m_size + j] = C[i * m_size + j] + tmpA[i * m_size + j] / denum;
        }
    }
    denum *= ++power;
    // A**2 (in tmpA2)
    cpu_matmul(A, tmpA, tmpA2, m_size);
    // exp = exp + A ** 2 / 2
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            C[i * m_size + j] = C[i * m_size + j] + tmpA2[i * m_size + j] / denum;
        }
    }

    denum *= ++power;
    // A**3 (in tmpA)
    cpu_matmul(A, tmpA2, tmpA, m_size);
    // exp = exp + A ** 3 / 6
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            C[i * m_size + j] = C[i * m_size + j] + tmpA[i * m_size + j] / denum;
        }
    }

    denum *= ++power;
    // A**4 (in tmpA2)
    cpu_matmul(A, tmpA, tmpA2, m_size);
    // exp = exp + A ** 4 / 24
    for(int i = 0; i < m_size; i++)
    {
        for(int j = 0; j < m_size; j++)
        {
            C[i * m_size + j] = C[i * m_size + j] + tmpA2[i * m_size + j] / denum;
        }
    }

    delete[] tmpA;
    delete[] tmpA2;
}
