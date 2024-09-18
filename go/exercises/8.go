// Question:
// Write a program that accepts a comma separated sequence of words as input and prints the words in a comma-separated sequence after sorting them alphabetically.
// Suppose the following input is supplied to the program:
// without,hello,bag,world
// Then, the output should be:
// bag,hello,without,world

// Hints:
// In case of input data being supplied to the question, it should be assumed to be a console input.

package main

import (
	"fmt"
	"strings"
)

func SortAndPrint(input string) {
	list := strings.Split(input, ",")

	var result []string

	for index := range len(list) {

	}

}

func main() {
	var input string

	fmt.Println("Please insert a comma separated sequence of words > ")
	fmt.Scanln(&input)

}
