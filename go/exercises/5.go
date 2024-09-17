// Question:
// Define a class which has at least two methods:
// getString: to get a string from console input
// printString: to print the string in upper case.
// Also please include simple test function to test the class methods.

// Hints:
// Use __init__ method to construct some parameters

package main

import (
	"fmt"
	"strings"
	"testing"
)

type MyClass struct {
	myString string
}

func (c *MyClass) setString(input string) {
	c.myString = input
}

func (c *MyClass) getUpperString() string {
	return strings.ToUpper(c.myString)
}

func TestMyClass(t *testing.T) {
	obj := MyClass{}

	obj.setString("ciao")
	result := obj.getUpperString()
	expected := "CIAO"

	if result != expected {
		t.Errorf("Expected %s but got %s", expected, result)
	}

	obj.setString("test")
	result = obj.getUpperString()
	expected = "TEST"

	if result != expected {
		t.Errorf("Expected %s but got %s", expected, result)
	}
}

func main() {
	fmt.Println("No buona prassi..")

	t := new(testing.T)
	TestMyClass(t)

	if t.Failed() {
		fmt.Println("I test hanno fallito")
	} else {
		fmt.Println("Tutti i test sono passati")
	}
}
