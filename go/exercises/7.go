// Question:
// Write a program which takes 2 digits, X,Y as input and generates a 2-dimensional array. The element value in the i-th row and j-th column of the array should be i*j.
// Note: i=0,1.., X-1; j=0,1,¡­Y-1.
// Example
// Suppose the following inputs are given to the program:
// 3,5
// Then, the output of the program should be:
// [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]

// Hints:
// Note: In case of input data being supplied to the question, it should be assumed to be a console input in a comma-separated form.

package main

import (
	"fmt"
	"strings"
)

func calculateMatrix(x, y int) [][]int {
	var result [x][j]int

	for i := 0; i < x; i++ {
		for j := 0; j < y; j++ {
			result[i][j] = i * j
		}
	}

	return result
}

func main() {
	var input string

	fmt.Println("Please insert two numbers > ")
	fmt.Scanln(&input)

	parts := strings.Split(input, ",")
	if len(parts) != 2 {
		fmt.Println("Error: input must contain exactly 2 numbers")
		return
	}

	x := strings.TrimSpace(parts[0])
	y := strings.TrimSpace(parts[1])

	output := calculateMatrix(x, y)
	fmt.Println("Output :", output)
}
