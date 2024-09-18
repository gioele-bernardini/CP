// Question:
// Write a program that accepts a sequence of whitespace separated words as input and prints the words after removing all duplicate words and sorting them alphanumerically.
// Suppose the following input is supplied to the program:
// hello world and practice makes perfect and hello world again
// Then, the output should be:
// again and hello makes perfect practice world

// Hints:
// In case of input data being supplied to the question, it should be assumed to be a console input.
// We use set container to remove duplicated data automatically and then use sorted() to sort the data.

package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
)

func removeDuplicatesAndSort(input string) string {
	// Split the input into words
	words := strings.Fields(input)

	// Use a map to remove duplicates
	wordMap := make(map[string]bool)
	for _, word := range words {
		wordMap[word] = true
	}

	// Collect the unique words into a slice
	var uniqueWords []string
	for word := range wordMap {
		uniqueWords = append(uniqueWords, word)
	}

	// Sort the words alphanumerically
	sort.Strings(uniqueWords)

	// Join the words back into a single string
	return strings.Join(uniqueWords, " ")
}

func main() {
	// Read input from the user
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Enter a sequence of words:")
	input, _ := reader.ReadString('\n')

	// Remove duplicates and sort
	result := removeDuplicatesAndSort(strings.TrimSpace(input))

	// Print the result
	fmt.Println(result)
}
