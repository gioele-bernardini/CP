package main

import (
	"fmt"
	"packages/utils"
)

func main() {
	result := utils.Add(3, 4)
	fmt.Println("Result: ", result)

	utils.Quote()
}
