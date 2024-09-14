package main

import (
	"fmt"
	"strings"
)

var pl = fmt.Println

func main() {
	sV1 := "A word"

	replace := strings.NewReplacer("A", "Another")
	sV2 := replace.Replace(sV1)
	pl(sV2)
}
