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

// #include <iostream>
// #include <cuda_runtime.h>
// #include <cmath>
// #include <chrono>

// // Kernel configuration parameters
// #define NUM_STEPS 40064 // For demonstration; should be set based on actual needs and divisible by THREADBLOCK_SIZE
// #define MAX_OPTIONS 1   // Processing a single option for simplicity
// #define THREADBLOCK_SIZE 128
// #define ELEMS_PER_THREAD (NUM_STEPS / THREADBLOCK_SIZE)

// // Option data structure
// struct OptionData
// {
//     float S;
//     float X;
//     float vDt;
//     float pu;
//     float pd;
// };
// __constant__ OptionData d_OptionData[MAX_OPTIONS];
// __device__ float d_CallValue[MAX_OPTIONS];

// __global__ void binomialOptionsKernel()
// {
//     int tid = threadIdx.x;
//     OptionData opt = d_OptionData[blockIdx.x];

//     float dt = 1.0f / NUM_STEPS; // Small time delta per step
//     float r = 0.001f;            // Risk-free rate
//     float u = exp(opt.vDt * sqrt(dt));
//     float d = 1.0f / u;
//     float pu = (exp(r * dt) - d) / (u - d); // Risk-neutral up probability
//     float pd = 1.0f - pu;                   // Down probability
//     float discount = exp(-r * dt);          // Discount factor per step

//     float call[NUM_STEPS + 1] = {0};
//     for (int i = 0; i <= NUM_STEPS; ++i)
//     {
//         float stockPrice = opt.S * powf(u, i) * powf(d, NUM_STEPS - i);
//         call[i] = fmaxf(stockPrice - opt.X, 0.0f);
//     }

//     for (int i = NUM_STEPS - 1; i >= 0; --i)
//     {
//         for (int j = 0; j <= i; ++j)
//         {
//             call[j] = (pu * call[j + 1] + pd * call[j]) * discount;
//         }
//     }

//     if (tid == 0)
//     {
//         d_CallValue[blockIdx.x] = call[0];
//     }
// }

// // Main function to setup and invoke the kernel
// int main()
// {
//     OptionData h_OptionData[MAX_OPTIONS] = {{127.2f, 252.0f, 0.2f}};
//     auto start_time = std::chrono::steady_clock::now();
//     cudaMemcpyToSymbol(d_OptionData, h_OptionData, sizeof(OptionData) * MAX_OPTIONS);

//     dim3 blocks(MAX_OPTIONS);
//     dim3 threads(1); // Only 1 thread for simplicity in the example
//     binomialOptionsKernel<<<blocks, threads>>>();

//     float callValue[MAX_OPTIONS];
//     cudaMemcpyFromSymbol(callValue, d_CallValue, sizeof(float) * MAX_OPTIONS, 0, cudaMemcpyDeviceToHost);
//     auto end_time = std::chrono::steady_clock::now();
//     std::chrono::duration<double> diff = end_time - start_time;
//     double seconds = diff.count();
//     std::cout << "Simulation Time GPU = " << seconds << "\n";

//     std::cout << "Calculated call value: " << callValue[0] << std::endl;

//     cudaDeviceReset();
//     return 0;
// }

// auto end_time = std::chrono::steady_clock::now();
// std::chrono::duration<double> diff = end_time - start_time;
// double seconds = diff.count();
// std::cout << "Simulation Time GPU = " << seconds << "\n";
// auto start_time = std::chrono::steady_clock::now();

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

struct Node
{
    double price;
    double optionvalue;
};

__global__ void initializeTreeLevel(Node *d_tree, double S, double upfactor, int level)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > level)
        return; // Ensure we do not go out of bounds

    int node_idx = (level * (level + 1)) / 2 + idx;

    if (level == 0 && idx == 0)
    {
        d_tree[node_idx].price = S;
    }
    else if (idx == 0)
    {
        int parent_idx = ((level - 1) * level) / 2;
        d_tree[node_idx].price = d_tree[parent_idx].price / upfactor;
    }
    else
    {
        int parent_idx = ((level - 1) * level) / 2 + (idx - 1);
        d_tree[node_idx].price = d_tree[parent_idx].price * upfactor;
    }
}

__global__ void setMaturityOptionValues(Node *d_tree, int n, double K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n)
        return; // Ensure we do not go out of bounds
    n = n - 1;

    int node_idx = (n * (n + 1)) / 2 + idx;
    d_tree[node_idx].optionvalue = max(d_tree[node_idx].price - K, 0.0);
}

__global__ void computeOptionValuesLevel(Node *d_tree, int level, double discountFactor, int n, double K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > level)
        return; // Ensure we do not go out of bounds

    int node_idx = (level * (level + 1)) / 2 + idx;
    if (level == n - 1)
    {
        d_tree[node_idx].optionvalue = max(d_tree[node_idx].price - K, 0.0);
    }
    else
    {
        int next_level_idx1 = ((level + 1) * (level + 2)) / 2 + idx;
        int next_level_idx2 = next_level_idx1 + 1;

        double g1 = d_tree[next_level_idx1].optionvalue;
        double g2 = d_tree[next_level_idx2].optionvalue;
        d_tree[node_idx].optionvalue = 0.5 * discountFactor * (g1 + g2);
    }
}

class BinomialTree
{
public:
    BinomialTree(double price, double vol, int _n, double _tstep);
    ~BinomialTree(); // Destructor to free memory
    double getValue(double K, double R);
    void print();

private:
    int n;
    double S;
    double volatility;
    double tstep;
    double tfin;
    double upfactor;
    Node *tree;
    Node *d_tree; // Device tree
};

BinomialTree::BinomialTree(double price, double vol, int _n, double _tstep)
{
    n = _n;
    S = price;
    volatility = vol;
    tstep = _tstep;
    tfin = n * tstep;
    upfactor = exp(volatility * sqrt(tstep));
    cout << "upfactor" << upfactor << endl;

    // Check for potential overflow and memory limits
    // if (n <= 0 || n > 500000)
    // { // Adjust the upper limit based on your system's capacity
    //     throw std::runtime_error("Invalid number of steps (n) specified.");
    // }

    // Calculate the size of the tree
    int64_t treeSize = static_cast<int64_t>(n) * (n + 1) / 2;
    // if (treeSize <= 0 || treeSize > 1e8)
    // { // Adjust the upper limit based on your system's capacity
    //     throw std::runtime_error("Invalid tree size");
    // }

    tree = new Node[treeSize];
    cudaMalloc((void **)&d_tree, treeSize * sizeof(Node));

    for (int level = 0; level < n; ++level)
    {
        int numThreads = min(level + 1, 256);
        int numBlocks = (level + numThreads - 1) / numThreads;
        if (numBlocks < 1)
        {
            numBlocks = 1;
        }
        initializeTreeLevel<<<numBlocks, numThreads>>>(d_tree, S, upfactor, level);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(tree, d_tree, treeSize * sizeof(Node), cudaMemcpyDeviceToHost);
}

BinomialTree::~BinomialTree()
{
    delete[] tree;
    cudaFree(d_tree);
}

double BinomialTree::getValue(double K, double R)
{
    double discountFactor = exp(-R * tstep);
    int64_t treeSize = static_cast<int64_t>(n) * (n + 1) / 2;

    // Set option values at maturity
    int numThreads = min(n + 1, 256);
    int numBlocks = (n + numThreads - 1) / numThreads;
    setMaturityOptionValues<<<numBlocks, numThreads>>>(d_tree, n, K);
    cudaDeviceSynchronize();

    // Calculate option values at earlier times
    for (int level = n - 1; level >= 0; --level)
    {
        numThreads = min(level + 1, 256);
        numBlocks = (level + numThreads - 1) / numThreads;
        if (numBlocks < 1)
        {
            numBlocks = 1;
        }
        computeOptionValuesLevel<<<numBlocks, numThreads>>>(d_tree, level, discountFactor, n, K);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(tree, d_tree, treeSize * sizeof(Node), cudaMemcpyDeviceToHost);

    return tree[0].optionvalue;
}

void BinomialTree::print()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            int idx = (i * (i + 1)) / 2 + j;
            cout << "[" << tree[idx].price << ", " << tree[idx].optionvalue << "]\t";
        }
        cout << endl;
    }
}

int main()
{
    double S, V, K, T, R, N;
    S = 127.2;
    V = 0.2;
    K = 252;
    T = 12;
    R = 0.001;
    N = 100;

    try
    {
        auto start_time = std::chrono::steady_clock::now();
        BinomialTree bt(S, V, N, T / N);
        double value = bt.getValue(K, R);

        auto end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        double seconds = diff.count();
        std::cout << "Simulation Time CUDA = " << seconds << "\n";
        // bt.print();
        cout << "OPTION VALUE = " << value << endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}