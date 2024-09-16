package main

import (
	"log"
	"net/http"
)

type TodoList struct {
	ToDoCount int
	ToDos     []string
}

func errorCheck(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func write(writer http.ResponseWriter, mst string) {
	_, err := writer.Write([]byte(msg))
	errorCheck()
}

func englishHandler(writer http.ResponseWriter, request *http.Request) {
	write(writer, "Hello, Internet!")
}
