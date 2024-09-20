// Question:
// Write a program which accepts a sequence of comma-separated numbers from console and generate a list and a tuple which contains every number.
// Suppose the following input is supplied to the program:
// 34,67,55,33,12,98
// Then, the output should be:
// ['34', '67', '55', '33', '12', '98']
// ('34', '67', '55', '33', '12', '98')

// Hints:
// In case of input data being supplied to the question, it should be assumed to be a console input.
// tuple() method can convert list to tuple

package main

import (
	"fmt"
	"strconv"
	"strings"
)

func generateListAndTuple(input string) ([]int, []int) {
	// Split the input string by commas
	strList := strings.Split(input, ",")

	// Create a slice of integers with the same length as the string slice
	list := make([]int, len(strList))

	// Convert each string in the list to an integer
	for i, value := range strList {
		num, err := strconv.Atoi(strings.TrimSpace(value)) // Trim spaces and convert to int
		if err != nil {
			fmt.Println("Errore di conversione:", err)
			return nil, nil
		}
		list[i] = num
	}

	// In Go, tuples don't exist. Here we simulate it by returning the same list twice.
	return list, list
}

func main() {
	var input string

	fmt.Println("Please insert a comma-separated list of numbers:")
	fmt.Scanln(&input)

	// Generate list and tuple
	list, tuple := generateListAndTuple(input)

	// Print the results
	fmt.Printf("List: %v\n", list)
	fmt.Printf("Tuple: %v\n", tuple)
}
