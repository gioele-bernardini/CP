package main

import (
	"fmt"
	"math/rand"
	"time"
)

// Function simulating a sensor that sends periodic readings
func sensor(sensorID int, ch chan int) {
	for {
		// Simulate a sensor reading
		reading := rand.Intn(100)
		fmt.Printf("Sensor %d sends reading: %d\n", sensorID, reading)
		ch <- reading                                         // Send the reading to the channel
		time.Sleep(time.Duration(rand.Intn(2)) * time.Second) // Simulate delay
	}
}

// Manager function that collects and processes readings from multiple sensors
func manager(ch1, ch2, ch3 chan int, done chan bool) {
	count := 0
	for {
		select {
		case reading := <-ch1:
			fmt.Printf("Manager received from sensor 1: %d\n", reading)
		case reading := <-ch2:
			fmt.Printf("Manager received from sensor 2: %d\n", reading)
		case reading := <-ch3:
			fmt.Printf("Manager received from sensor 3: %d\n", reading)
		}

		count++
		if count >= 10 { // After processing 10 readings, stop the manager
			fmt.Println("Manager has received enough readings, terminating.")
			done <- true // Signal that the manager is done
			return
		}
	}
}

func main() {
	// Create buffered channels for sensors
	ch1 := make(chan int, 5)
	ch2 := make(chan int, 5)
	ch3 := make(chan int, 5)

	done := make(chan bool) // Channel to signal manager termination

	// Start three goroutines simulating the sensors
	go sensor(1, ch1)
	go sensor(2, ch2)
	go sensor(3, ch3)

	// Start the manager in a goroutine
	go manager(ch1, ch2, ch3, done)

	// Wait for the manager to finish
	<-done

	// Close the channels (not strictly necessary here since the program will exit, but good practice)
	close(ch1)
	close(ch2)
	close(ch3)
	close(done)

	fmt.Println("Program terminated.")
}
