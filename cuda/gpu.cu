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
    // cout << "upfactor" << upfactor << endl;

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

// int main()
// {
//     double S, V, K, T, R, N;
//     S = 127.2;
//     V = 0.2;
//     K = 252;
//     T = 12;
//     R = 0.001;
//     N = 100;

//     try
//     {
//         auto start_time = std::chrono::steady_clock::now();
//         BinomialTree bt(S, V, N, T / N);
//         double value = bt.getValue(K, R);

//         auto end_time = std::chrono::steady_clock::now();
//         std::chrono::duration<double> diff = end_time - start_time;
//         double seconds = diff.count();
//         std::cout << "Simulation Time CUDA = " << seconds << "\n";
//         // bt.print();
//         cout << "OPTION VALUE = " << value << endl;
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }

//     return 0;
// }