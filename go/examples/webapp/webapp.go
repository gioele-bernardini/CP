package main

import (
	"bufio"
	"fmt"
	"log"
	"net/http"
	"os"
	"text/template"
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

func getStrings(fileName string) []string {
	var lines []string
	file, err := os.Open(fileName)

	if os.IsNotExist(err) {
		return nil
	}
	errorCheck(err)
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	errorCheck(scanner.Err())

	return lines
}

func englishHandler(writer http.ResponseWriter, request *http.Request) {
	write(writer, "Hello, Internet!")
}

func SpanishHandler(writer http.ResponseWriter, request *http.Request) {
	write(writer, "Hola, Internet!")
}

func interactHandler(writer http.ResponseWriter, request *http.Request) {
	todoVals := getStrings("todo.txt")
	fmt.Printf("%#v\n", todoVals)
	tmpl, err := template.ParseFiles("view.html")
	errorCheck(err)

	todos := TodoList{
		ToDoCount: len(todoVals),
		ToDos:     todoVals,
	}
	err = tmpl.Execute(writer, todos)
}

func newHandler(writer http.ResponseWriter, request *http.Request) {
	tmpl, err := template.ParseFiles("new.html")
	errorCheck(err)
	err = tmpl.Execute(writer, nil)
}
