// Question:
// Define a class which has at least two methods:
// getString: to get a string from console input
// printString: to print the string in upper case.
// Also please include simple test function to test the class methods.

// Hints:
// Use __init__ method to construct some parameters

package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func printToUpper(input string) {
	fmt.Println("Output :", strings.ToUpper(input))
}

func main() {
	// Read whole line until the first newline character
	reader := bufio.NewReader(os.Stdin)

	fmt.Print("Please insert a string > ")

	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	printToUpper(input)
}
