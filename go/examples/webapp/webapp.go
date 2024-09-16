package main

import (
	"bufio"
	"fmt"
	"log"
	"net/http"
	"os"
	"text/template"
)

// TodoList struct represents the data passed to the HTML template, including
// the number of todos and the list of todos themselves.
type TodoList struct {
	ToDoCount int
	ToDos     []string
}

// errorCheck is a helper function that checks if an error occurred.
// If there's an error, it will log the error and stop the program.
func errorCheck(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// write is a helper function to write a message to the HTTP response.
// It uses the writer interface to send a string message.
func write(writer http.ResponseWriter, msg string) {
	_, err := writer.Write([]byte(msg))
	errorCheck(err) // If there's an error in writing, it will log the error.
}

// getStrings reads a file line by line and returns a slice of strings
// containing each line from the file. If the file doesn't exist, it returns nil.
func getStrings(fileName string) []string {
	var lines []string
	file, err := os.Open(fileName)

	// If the file doesn't exist, return nil (no todos).
	if os.IsNotExist(err) {
		return nil
	}
	errorCheck(err)
	defer file.Close()

	// Use a scanner to read the file line by line.
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	errorCheck(scanner.Err()) // Check for scanning errors.

	return lines
}

// englishHandler handles the "/hello" route and responds with "Hello, Internet!".
func englishHandler(writer http.ResponseWriter, request *http.Request) {
	write(writer, "Hello, Internet!")
}

// spanishHandler handles the "/hola" route and responds with "Hola, Internet!".
func spanishHandler(writer http.ResponseWriter, request *http.Request) {
	write(writer, "Hola, Internet!")
}

// interactHandler handles the "/interact" route. It reads the list of todos
// from the "todo.txt" file, then passes the list to the "view.html" template for rendering.
func interactHandler(writer http.ResponseWriter, request *http.Request) {
	todoVals := getStrings("todo.txt") // Get the todos from the file.
	fmt.Printf("%#v\n", todoVals)      // Print todos to the console for debugging.

	// Parse the HTML template for rendering the todos.
	tmpl, err := template.ParseFiles("view.html")
	errorCheck(err)

	// Create a TodoList struct to pass to the template.
	todos := TodoList{
		ToDoCount: len(todoVals),
		ToDos:     todoVals,
	}
	// Execute the template and send the rendered HTML back to the browser.
	err = tmpl.Execute(writer, todos)
	errorCheck(err) // Check if template execution fails.
}

// newHandler handles the "/new" route and renders the "new.html" template
// to allow the user to add a new todo.
func newHandler(writer http.ResponseWriter, request *http.Request) {
	// Parse the HTML template for creating a new todo.
	tmpl, err := template.ParseFiles("new.html")
	errorCheck(err)
	// Execute the template (no data is passed here).
	err = tmpl.Execute(writer, nil)
	errorCheck(err) // Check if template execution fails.
}

// createHandler handles the form submission from the "new.html" page.
// It appends the new todo to the "todo.txt" file and then redirects the user
// back to the "/interact" page to see the updated list.
func createHandler(writer http.ResponseWriter, request *http.Request) {
	todo := request.FormValue("todo") // Get the new todo from the form submission.

	// Open the file in append mode, or create it if it doesn't exist.
	options := os.O_WRONLY | os.O_APPEND | os.O_CREATE
	file, err := os.OpenFile("todo.txt", options, os.FileMode(0600))
	errorCheck(err)

	// Write the new todo to the file.
	_, err = fmt.Fprintln(file, todo)
	errorCheck(err) // Check for any errors in writing to the file.

	// Close the file.
	err = file.Close()
	errorCheck(err)

	// Redirect the user to the "/interact" page to see the updated todo list.
	http.Redirect(writer, request, "/interact", http.StatusFound)
}

// main function sets up the HTTP server and routes. It maps different URL paths
// to their respective handler functions and starts the web server on localhost:8080.
func main() {
	// Set up routes for handling different paths.
	http.HandleFunc("/hello", englishHandler)     // Route for "/hello" (English greeting).
	http.HandleFunc("/hola", spanishHandler)      // Route for "/hola" (Spanish greeting).
	http.HandleFunc("/interact", interactHandler) // Route for viewing the todo list.
	http.HandleFunc("/new", newHandler)           // Route for creating a new todo.
	http.HandleFunc("/create", createHandler)     // Route for submitting the new todo form.

	// Start the web server on localhost at port 8080.
	err := http.ListenAndServe("localhost:8080", nil)
	log.Fatal(err) // Log any errors that occur when starting the server.
}
