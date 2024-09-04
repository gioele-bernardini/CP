#include <stdio.h>

int main() {
  unsigned value = 1;

  char* pointer = (char*) &value;

  if (pointer) {
    printf("Little Endian\n");
  } else {
    printf("Big Endian\n");
  }

  return 0;
}

