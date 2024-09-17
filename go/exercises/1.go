// Question:
// Write a program which will find all such numbers which are divisible by 7 but are not a multiple of 5, between 2000 and 3200 (both included).
// The numbers obtained should be printed in a comma-separated sequence on a single line.

// Hints:
// Consider use range(#begin, #end) method

package main

import (
	"fmt"
	"strings"
)

func findNumbers() string {
	var result []string

	for i := 2000; i <= 3200; i++ {
		if i%7 == 0 && i%5 != 0 {
			result = append(result, fmt.Sprint(i))
		}
	}

	return strings.Join(result, ",")
}

func main() {
	result := findNumbers()
	fmt.Println(result)
}
