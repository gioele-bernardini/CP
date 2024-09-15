package main

import (
	"fmt"
	"math/rand"
	"time"
)

// Funzione che simula un sensore che invia letture periodicamente
func sensor(sensorID int, ch chan int) {
	for {
		// Simula una lettura del sensore
		reading := rand.Intn(100)
		fmt.Printf("Sensore %d invia lettura: %d\n", sensorID, reading)
		ch <- reading                                         // Invio della lettura sul canale
		time.Sleep(time.Duration(rand.Intn(2)) * time.Second) // Simuliamo un ritardo
	}
}

// Funzione gestore che raccoglie e processa le letture da più sensori
func manager(ch1, ch2, ch3 chan int, done chan bool) {
	count := 0
	for {
		select {
		case reading := <-ch1:
			fmt.Printf("Gestore ha ricevuto dal sensore 1: %d\n", reading)
		case reading := <-ch2:
			fmt.Printf("Gestore ha ricevuto dal sensore 2: %d\n", reading)
		case reading := <-ch3:
			fmt.Printf("Gestore ha ricevuto dal sensore 3: %d\n", reading)
		}

		count++
		if count >= 10 { // Dopo aver processato 10 letture, il gestore si ferma
			fmt.Println("Gestore ha ricevuto abbastanza letture, terminazione.")
			done <- true // Segnala che il gestore ha terminato
			return
		}
	}
}

func main() {
	// Creazione di canali bufferizzati per i sensori
	ch1 := make(chan int, 5)
	ch2 := make(chan int, 5)
	ch3 := make(chan int, 5)

	done := make(chan bool) // Canale per segnalare la terminazione del gestore

	// Avvia tre goroutine per simulare i sensori
	go sensor(1, ch1)
	go sensor(2, ch2)
	go sensor(3, ch3)

	// Avvia il gestore in una goroutine
	go manager(ch1, ch2, ch3, done)

	// Aspetta che il gestore termini
	<-done

	// Chiudiamo i canali (in questo caso non è strettamente necessario poiché il programma termina, ma è una buona pratica)
	close(ch1)
	close(ch2)
	close(ch3)
	close(done)

	fmt.Println("Programma terminato.")
}
