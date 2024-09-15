package main

import "fmt"

var pl = fmt.Println

func main() {
	ch := make(chan int)

	// Goroutine for sending data
	go func() {
		ch <- 42
	}()

	val := <-ch
	pl(val)

	// Check if the channel is closed
	val2, ok := <-ch
	if !ok {
		fmt.Println("Channel closed.")
	}
}
