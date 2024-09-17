// Question:
// Write a program which can compute the factorial of a given numbers.
// The results should be printed in a comma-separated sequence on a single line.
// Suppose the following input is supplied to the program:
// 8
// Then, the output should be:
// 40320

// Hints:
// In case of input data being supplied to the question, it should be assumed to be a console input.

package main

import "fmt"

func factorial(n int) int {
	if n == 0 || n == 1 {
		return 1
	}

	return n * factorial(n-1)
}

func main() {
	result := factorial(10)
	fmt.Println("Output :", result)
}
