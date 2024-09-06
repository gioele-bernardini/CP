#include <iostream>

using namespace std;

// Overloaded swap function for int
void swap(int& a, int& b) {
  int temp = a;
  a = b;
  b = temp;
}

// Overloaded swap function for double
void swap(double& a, double& b) {
  double temp = a;
  a = b;
  b = temp;
}

int main() {
  int m = 5, n = 10;

  cout << "Inputs: " << m << ", " << n << endl;
  swap(m, n);
  cout << "Outputs: " << m << ", " << n << endl;

  double x = 5.3, y = 10.6;

  cout << "Double inputs: " << x << ", " << y << endl;
  swap(x, y);
  cout << "Double outputs: " << x << ", " << y << endl;

  return 0;
}

