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

func factorial(number int) int {
	if number == 0 || number == 1 {
		return 1
	}

	return number * factorial(number-1)
}

func main() {
	// var target int
	// fmt.Print("Please insert a number : ")
	// fmt.Scan(&target)

	target := 6
	var result = factorial(target)

	fmt.Printf("Output : %d\n", result)
}
