// Question:
// Write a program that calculates and prints the value according to the given formula:
// Q = Square root of [(2 * C * D)/H]
// Following are the fixed values of C and H:
// C is 50. H is 30.
// D is the variable whose values should be input to your program in a comma-separated sequence.
// Example
// Let us assume the following comma separated input sequence is given to the program:
// 100,150,180
// The output of the program should be:
// 18,22,24

package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

const C = 50
const H = 30

func calculateValue(input string) string {
	list := strings.Split(input, ",")
	var results []string

	for _, numStr := range list {
		num, err := strconv.Atoi(numStr)
		if err != nil {
			fmt.Println("Error accurred converting :", numStr)
			continue
		}
		Q := int(math.Sqrt(float64(2*C*num) / H))
		results = append(results, strconv.Itoa(Q))
	}

	return strings.Join(results, ",")
}

func main() {
	input := "100,150,180"
	output := calculateValue(input)

	fmt.Println(output)
}
