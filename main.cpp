#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <serial.cpp>

// =================
// Helper Functions
// =================

// I/O routines
void save(std::ofstream &fsave, particle_t *parts, int num_parts, double size)
{
  static bool first = true;

  if (first)
  {
    fsave << num_parts << " " << size << "\n";
    first = false;
  }

  for (int i = 0; i < num_parts; ++i)
  {
    fsave << parts[i].x << " " << parts[i].y << "\n";
  }

  fsave << std::endl;
}

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

// ==============
// Main Function
// ==============

int main(int argc, char **argv)
{
#ifdef _OPENMP
#pragma omp parallel default(shared)
#endif
  {
    for (int step = 0; step < nsteps; ++step)
    {
      simulate_one_step(parts, num_parts, size);

      // Save state if necessary
#ifdef _OPENMP
#pragma omp master
#endif
      if (fsave.good() && (step % savefreq) == 0)
      {
        save(fsave, parts, num_parts, size);
      }
    }
  }

  auto end_time = std::chrono::steady_clock::now();

  std::chrono::duration<double> diff = end_time - start_time;
  double seconds = diff.count();

  // Finalize
  std::cout << "Simulation Time = " << seconds << " seconds for " << num_parts << " particles.\n";
  fsave.close();
  delete[] parts;
}
