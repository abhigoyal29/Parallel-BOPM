#ifndef IMPL_H
#define IMPL_H

#include <cmath>

struct Node
{
  double price;
  double optionvalue;
};

class BinomialTree
{
private:
  Node **tree;
  int n;
  double S, volatility, upfactor, tfin, tstep;

  void initNode(int level, int node);

public:
  BinomialTree(double S, double volatility, int n, double tstep);
  double getValue(double K, double R);
  void print();
};

#endif
