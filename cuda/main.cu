#include <chrono>
#include <cmath>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "impl.h"
using namespace std;

#define num_inputs 1000

// =================
// Helper Functions
// =================

// Command Line Option Processing
int find_arg_idx(int argc, char **argv, const char *option)
{
  for (int i = 1; i < argc; ++i)
  {
    if (strcmp(argv[i], option) == 0)
    {
      return i;
    }
  }
  return -1;
}

int find_int_arg(int argc, char **argv, const char *option, int default_value)
{
  int iplace = find_arg_idx(argc, argv, option);

  if (iplace >= 0 && iplace < argc - 1)
  {
    return std::stoi(argv[iplace + 1]);
  }

  return default_value;
}

char *find_string_option(int argc, char **argv, const char *option, char *default_value)
{
  int iplace = find_arg_idx(argc, argv, option);

  if (iplace >= 0 && iplace < argc - 1)
  {
    return argv[iplace + 1];
  }

  return default_value;
}

vector<double> S_vals;
vector<double> V_vals;
vector<double> K_vals;
vector<double> T_vals;
vector<double> R_vals;

void init_inputs()
{
  S_vals.resize(num_inputs);
  V_vals.resize(num_inputs);
  K_vals.resize(num_inputs);
  T_vals.resize(num_inputs);
  R_vals.resize(num_inputs);

  ifstream inFile;
  inFile.open("/global/homes/a/ag894/CS5220-HW/Parallel-BOPM/benchmarking/MSFT_benchmark.txt");

  if (inFile.fail())
  {
    cerr << "unable to open file for reading" << endl;
    exit(1);
  }

  string str;
  int i = 0;
  double S, V, K, T, R;
  while (inFile >> S >> V >> K >> T >> R)
  {
    if (i >= num_inputs)
    {
      break;
    }
    S_vals[i] = S;
    V_vals[i] = V;
    K_vals[i] = K;
    T_vals[i] = T;
    R_vals[i] = R;

    i++;
  }
}

// ==============
// Main Function
// ==============

int main(int argc, char **argv)
{

  init_inputs();

  int N = find_int_arg(argc, argv, "-n", 100);

  // file to write output for correctness
  char *filename = find_string_option(argc, argv, "-o", nullptr);
  ofstream output(filename);

  auto start_time = std::chrono::steady_clock::now();

  for (int i = 0; i < num_inputs; i++)
  {
    BinomialTree bt(S_vals[i], V_vals[i], N, T_vals[i] / N);
    double value = bt.getValue(K_vals[i], R_vals[i]);

    if (output.good())
      output << value << endl;
  }

  auto end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  double seconds = diff.count();
  cout << "Simulation Time for " << N << " time steps and " << num_inputs << " contracts = " << seconds << endl;
  // cout << "OPTION VALUE = " << value << endl;
}
