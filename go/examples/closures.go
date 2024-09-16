package main

import "fmt"

func makeCounter() func() int {
	// Una variabile locale per mantenere lo stato del contatore
	count := 0

	// Restituiamo una closure
	return func() int {
		count++
		return count
	}
}

func main() {
	counter1 := makeCounter()
	counter2 := makeCounter()

	fmt.Println(counter1()) // Stampa: 1
	fmt.Println(counter1()) // Stampa: 2
	fmt.Println(counter2()) // Stampa: 1
	fmt.Println(counter2()) // Stampa: 2
}
