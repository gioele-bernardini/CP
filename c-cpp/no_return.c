#include <ncurses.h> // Library for ncurses functions
#include <stdio.h>
#include <string.h>

int main() {
  char buffer[5] = {0}; // Buffer to store the input
  int index = 0;

  // Initialize ncurses
  initscr(); // Start curses mode
  cbreak();  // Disable line buffering, pass every character
  noecho();  // Don't echo characters on screen

  printw("Type 'ciao' for an immediate response:\n");
  refresh();

  while (1) {
    char c = getch(); // Get a single character input without waiting for Enter
    printw("%c", c);  // Print the character to the screen
    buffer[index++] = c;

    // If the user typed "ciao"
    if (index == 4 && strncmp(buffer, "ciao", 4) == 0) {
      printw("\nYou typed 'ciao'! Greetings!\n");
      break;
    }

    // Reset the buffer if more than 4 characters are entered
    if (index == 4) {
      index = 0;
    }

    refresh(); // Refresh the screen to update the output
  }

  // End ncurses mode
  endwin();
  return 0;
}

