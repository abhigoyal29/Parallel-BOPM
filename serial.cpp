#include <iostream>
#include <cmath>
#include <chrono>
#include "impl.h"
using namespace std;

BinomialTree::BinomialTree(double price, double vol, int _n, double _tstep)
{
    n = _n;
    S = price;
    volatility = vol;
    tstep = _tstep;
    tfin = n * tstep;
    upfactor = exp(volatility * sqrt(tstep));
    tree = new Node *[n];
    for (int i = 0; i < n; i++)
        tree[i] = new Node[i + 1];

    tree[0][0].price = S;
    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            if (j == 0)
                tree[i][j].price = tree[i - 1][j].price / upfactor;
            else
                tree[i][j].price = tree[i - 1][j - 1].price * upfactor;
        }
    }
}

double BinomialTree::getValue(double K, double R)
{
    double discountFactor = exp(-R * tstep);

    // Set option values at maturity
    for (int j = 0; j < n; j++)
    {
        tree[n - 1][j].optionvalue = max(tree[n - 1][j].price - K, 0.0);
    }

    // Calculate option values at earlier times
    for (int i = n - 2; i >= 0; i--)
    {
        for (int j = 0; j <= i; j++)
        {
            double g1 = tree[i + 1][j + 1].optionvalue;
            double g2 = tree[i + 1][j].optionvalue;
            tree[i][j].optionvalue = 0.5 * discountFactor * (g1 + g2);
        }
    }

    return tree[0][0].optionvalue;
}

void BinomialTree::print()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            cout << "[" << tree[i][j].price << ", " << tree[i][j].optionvalue << "]\t";
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
//     N = 50000;

//     auto start_time = std::chrono::steady_clock::now();
//     BinomialTree bt(S, V, N, T / N);
//     double value = bt.getValue(K, R);

//     auto end_time = std::chrono::steady_clock::now();
//     std::chrono::duration<double> diff = end_time - start_time;
//     double seconds = diff.count();
//     cout << "Simulation Time Serial = " << seconds << endl;
//     cout << "OPTION VALUE = " << value << endl;
//     return 0;
// }