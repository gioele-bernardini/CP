#include <stdio.h>

int main() {
  // Try to open the file in "rb" mode (read binary)
  FILE* file = fopen("counter.txt", "rb");

  int counter = 0;

  // If the file doesn't exist, create it
  if (!file) {
    fprintf(stderr, "File not found, creating new one...\n");
    file = fopen("counter.txt", "wb");
    if (!file) {
      fprintf(stderr, "Error while creating the file\n");
      return 1;
    }
    // Write the initial counter value (0) to the file
    fwrite(&counter, sizeof(counter), 1, file);
    fclose(file);
  } else {
    // If the file exists, read the current counter value
    fread(&counter, sizeof(counter), 1, file);
    fclose(file);
  }

  // Increment the counter
  counter++;

  // Reopen the file in "wb" mode (write binary) to update the value
  file = fopen("counter.txt", "wb");
  if (!file) {
    fprintf(stderr, "Error while opening the file for writing\n");
    return 1;
  }

  // Write the updated counter value to the file
  fwrite(&counter, sizeof(counter), 1, file);

  // Flush the buffer to ensure data is written to disk (optional but good practice)
  fflush(file);

  // Close the file
  fclose(file);

  // Print the updated counter value for debugging purposes
  printf("Counter updated to: %d\n", counter);

  return 0;
}

