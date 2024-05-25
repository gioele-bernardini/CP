#! /usr/bin/env python3

from http.server import BaseHTTPRequestHandler, HTTPServer  # Importa i moduli necessari per creare un server HTTP

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    self.send_response(200)  # Invia una risposta HTTP con lo status code 200 (OK)
    self.send_header("Content-type", "text/html")  # Aggiunge l'intestazione HTTP "Content-type" con valore "text/html"
    self.end_headers()  # Termina l'invio delle intestazioni della risposta
    self.wfile.write(b"Hello, world!")  # Scrive il corpo della risposta, "Hello, world!", come byte

def run_server(port=8000):
  server_address = ("127.0.0.1", port)  # Indirizzo IP e porta su cui il server sarà in ascolto
  httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)  # Crea un'istanza del server HTTP utilizzando l'indirizzo e il gestore delle richieste
  print(f"Server running at http://127.0.0.1:{port}")  # Stampa un messaggio indicando che il server è in esecuzione
  httpd.serve_forever()  # Avvia il server e lo fa continuare ad ascoltare per le richieste HTTP

if __name__ == "__main__":
  run_server()  # Avvia il server quando lo script viene eseguito direttamente