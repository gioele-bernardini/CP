#include <stdio.h>

int main() {
  unsigned int value = 1;

  // Casting the address of value to a char pointer
  char* pointer = (char*) &value;

  // Checking the value of the first byte
  if (*pointer) {
    printf("Little Endian\n");
  } else {
    printf("Big Endian\n");
  }

  return 0;
}

