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
	"strings"
)

func generateListAndTuple(input string) ([]string, []string) {
	list := strings.Split(input, ",")

	tuple := make([]string, len(list))
	copy(tuple, list)

	return list, tuple
}

func main() {
	var userInput string

	fmt.Printf("Please insert your string > ")
	fmt.Scanln(&userInput)

	tuple, list := generateListAndTuple(userInput)
	fmt.Println("Tuple :", tuple)
	fmt.Println("List : ", list)
}
