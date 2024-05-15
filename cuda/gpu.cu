// #include <iostream>
// #include <cmath>
// #include <chrono>
// #include <cuda_runtime.h>

// struct Node
// {
//     double price, optionvalue;
// };

// __global__ void calculatePrices(Node *d_tree, double upfactor, int n)
// {
//     int i = blockIdx.x + 1; // Adjusting for zero-based indexing
//     int j = threadIdx.x;

//     if (i < n)
//     {
//         if (j <= i)
//         {
//             if (j == 0)
//             {
//                 d_tree[i * (i + 1) / 2].price = d_tree[(i - 1) * i / 2].price / upfactor;
//             }
//             else
//             {
//                 d_tree[i * (i + 1) / 2 + j].price = d_tree[(i - 1) * i / 2 + j - 1].price * upfactor;
//             }
//         }
//     }
// }

// __global__ void calculateOptionValues(Node *d_tree, int n, double K, double discountFactor)
// {
//     int i = blockIdx.x;
//     int j = threadIdx.x;

//     if (i == n - 1)
//     {
//         if (j < n)
//         {
//             d_tree[i * (i + 1) / 2 + j].optionvalue = fmax(d_tree[i * (i + 1) / 2 + j].price - K, 0.0);
//         }
//     }
//     else
//     {
//         if (j <= i)
//         {
//             double g1 = d_tree[(i + 1) * (i + 2) / 2 + j + 1].optionvalue;
//             double g2 = d_tree[(i + 1) * (i + 2) / 2 + j].optionvalue;
//             d_tree[i * (i + 1) / 2 + j].optionvalue = 0.5 * discountFactor * (g1 + g2);
//         }
//     }
// }

// class BinomialTree
// {
// private:
//     Node *h_tree;
//     Node *d_tree;
//     int n;
//     double S, volatility, upfactor, tfin, tstep;

// public:
//     BinomialTree(double S, double volatility, int n, double tstep);
//     ~BinomialTree();
//     double getValue(double K, double R);
//     void print();
// };

// BinomialTree::BinomialTree(double price, double vol, int _n, double _tstep)
// {
//     n = _n;
//     S = price;
//     volatility = vol;
//     tstep = _tstep;
//     tfin = n * tstep;
//     upfactor = exp(volatility * sqrt(tstep));

//     size_t treeSize = n * (n + 1) / 2 * sizeof(Node);

//     h_tree = new Node[n * (n + 1) / 2];
//     for (int i = 0; i < n * (n + 1) / 2; i++)
//     {
//         h_tree[i].price = 0.0;
//         h_tree[i].optionvalue = 0.0;
//     }
//     h_tree[0].price = S;

//     cudaMalloc(&d_tree, treeSize);
//     cudaMemcpy(d_tree, h_tree, treeSize, cudaMemcpyHostToDevice);
// }

// BinomialTree::~BinomialTree()
// {
//     delete[] h_tree;
//     cudaFree(d_tree);
// }

// double BinomialTree::getValue(double K, double R)
// {
//     dim3 gridDim(n);
//     dim3 blockDim(n);

//     for (int i = 1; i < n; i++)
//     {
//         calculatePrices<<<1, i + 1>>>(d_tree, upfactor, n);
//     }

//     double discountFactor = exp(-R * tstep);

//     for (int i = n - 1; i >= 0; i--)
//     {
//         calculateOptionValues<<<1, i + 1>>>(d_tree, n, K, discountFactor);
//     }

//     cudaMemcpy(h_tree, d_tree, n * (n + 1) / 2 * sizeof(Node), cudaMemcpyDeviceToHost);
//     return h_tree[0].optionvalue;
// }

// void BinomialTree::print()
// {
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j <= i; j++)
//         {
//             std::cout << "[" << h_tree[i * (i + 1) / 2 + j].price << ", " << h_tree[i * (i + 1) / 2 + j].optionvalue << "]\t";
//         }
//         std::cout << std::endl;
//     }
// }

// int main()
// {
//     double S, V, K, T, R, N;
//     S = 127.2;
//     V = 0.2;
//     K = 252;
//     T = 12;
//     R = 0.001;
//     N = 50;

//     auto start_time = std::chrono::steady_clock::now();
//     BinomialTree bt(S, V, N, T / N);
//     double value = bt.getValue(K, R);

//     auto end_time = std::chrono::steady_clock::now();
//     std::chrono::duration<double> diff = end_time - start_time;
//     double seconds = diff.count();
//     std::cout << "Simulation Time GPU = " << seconds << "\n";
//     // bt.print();
//     std::cout << "OPTION VALUE = " << value << std::endl;
//     return 0;
// }

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

// Kernel configuration parameters
#define NUM_STEPS 40064 // For demonstration; should be set based on actual needs and divisible by THREADBLOCK_SIZE
#define MAX_OPTIONS 1   // Processing a single option for simplicity
#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS / THREADBLOCK_SIZE)

// Option data structure
struct OptionData
{
    float S;
    float X;
    float vDt;
    float pu;
    float pd;
};
__constant__ OptionData d_OptionData[MAX_OPTIONS];
__device__ float d_CallValue[MAX_OPTIONS];

__global__ void binomialOptionsKernel()
{
    int tid = threadIdx.x;
    OptionData opt = d_OptionData[blockIdx.x];

    float dt = 1.0f / NUM_STEPS; // Small time delta per step
    float r = 0.001f;            // Risk-free rate
    float u = exp(opt.vDt * sqrt(dt));
    float d = 1.0f / u;
    float pu = (exp(r * dt) - d) / (u - d); // Risk-neutral up probability
    float pd = 1.0f - pu;                   // Down probability
    float discount = exp(-r * dt);          // Discount factor per step

    float call[NUM_STEPS + 1] = {0};
    for (int i = 0; i <= NUM_STEPS; ++i)
    {
        float stockPrice = opt.S * powf(u, i) * powf(d, NUM_STEPS - i);
        call[i] = fmaxf(stockPrice - opt.X, 0.0f);
    }

    for (int i = NUM_STEPS - 1; i >= 0; --i)
    {
        for (int j = 0; j <= i; ++j)
        {
            call[j] = (pu * call[j + 1] + pd * call[j]) * discount;
        }
    }

    if (tid == 0)
    {
        d_CallValue[blockIdx.x] = call[0];
    }
}

// Main function to setup and invoke the kernel
int main()
{
    OptionData h_OptionData[MAX_OPTIONS] = {{127.2f, 252.0f, 0.2f}};
    auto start_time = std::chrono::steady_clock::now();
    cudaMemcpyToSymbol(d_OptionData, h_OptionData, sizeof(OptionData) * MAX_OPTIONS);

    dim3 blocks(MAX_OPTIONS);
    dim3 threads(1); // Only 1 thread for simplicity in the example
    binomialOptionsKernel<<<blocks, threads>>>();

    float callValue[MAX_OPTIONS];
    cudaMemcpyFromSymbol(callValue, d_CallValue, sizeof(float) * MAX_OPTIONS, 0, cudaMemcpyDeviceToHost);
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();
    std::cout << "Simulation Time GPU = " << seconds << "\n";

    std::cout << "Calculated call value: " << callValue[0] << std::endl;

    cudaDeviceReset();
    return 0;
}

// auto end_time = std::chrono::steady_clock::now();
// std::chrono::duration<double> diff = end_time - start_time;
// double seconds = diff.count();
// std::cout << "Simulation Time GPU = " << seconds << "\n";
// auto start_time = std::chrono::steady_clock::now();
