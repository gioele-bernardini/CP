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

func write(writer http.ResponseWriter, msg string) {
	_, err := writer.Write([]byte(msg))
	errorCheck(err) // Pass the error to the errorCheck function
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

func spanishHandler(writer http.ResponseWriter, request *http.Request) {
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
	errorCheck(err) // Check error here as well
}

func newHandler(writer http.ResponseWriter, request *http.Request) {
	tmpl, err := template.ParseFiles("new.html")
	errorCheck(err)
	err = tmpl.Execute(writer, nil)
	errorCheck(err) // Check error here as well
}

func createHandler(writer http.ResponseWriter, request *http.Request) {
	todo := request.FormValue("todo")
	options := os.O_WRONLY | os.O_APPEND | os.O_CREATE
	file, err := os.OpenFile("todo.txt", options, os.FileMode(0600))
	errorCheck(err)

	_, err = fmt.Fprintln(file, todo)
	errorCheck(err) // Pass the error to the errorCheck function
	err = file.Close()
	errorCheck(err)
	http.Redirect(writer, request, "/interact", http.StatusFound)
}

func main() {
	http.HandleFunc("/hello", englishHandler)
	http.HandleFunc("/hola", spanishHandler)
	http.HandleFunc("/interact", interactHandler)
	http.HandleFunc("/new", newHandler)
	http.HandleFunc("/create", createHandler)

	err := http.ListenAndServe("localhost:8080", nil)
	log.Fatal(err)
}
