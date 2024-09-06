#include <stdio.h>

void swap_double(double* a, double* b) {
  double temp = *a;

  *a = *b;
  *b = temp;
}

void swap(int* a, int* b) {
  int temp = *a;

  *a = *b;
  *b = temp;
}

int main() {
  int m = 5, n = 10;

  printf("Inputs: %d, %d\n", m, n);
  swap(&m, &n);
  printf("Outputs: %d, %d\n", m, n);

  double x = 5.3, y = 10.6;

  printf("Double inputs: %lf, %lf\n", x, y);
  swap_double(&x, &y);
  printf("Double outputs: %lf, %lf\n", x, y);

  return 0;
}

