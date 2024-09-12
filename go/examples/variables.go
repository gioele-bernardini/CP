/**
 * The live server raises conflicts due to multiple
 * instances of 'main' and 'pl' identifiers existing
 * across different files within the same folder.
 */

package main

import (
	"fmt"
	"reflect"
	"strconv"
)

var pl = fmt.Println

func main() {
	// var vName string = "Derek"
	// var v1, v2 = 1.2, 3.4
	// var v3 = "Hello!"
	// var v4 float64 = 2.3
	// v4 := 2.4 // Automatically determines the type
	// v4 = 5.4
	// castedValue = int(v4)

	// Propagation of error + casting
	x := "5000"
	y, err := strconv.Atoi(x)
	// z := strconv.Itoa(y)
	pl(x, err, reflect.TypeOf(y))

	// Int, float64, bool
	// They got default values!

	// Reflection can let the program modify itself
	// under some circumstances on runtime!
	pl(reflect.TypeOf(25))
}
