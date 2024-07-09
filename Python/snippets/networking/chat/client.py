#!/usr/bin/env python3

import socket

def main():
    host = '127.0.0.1'  # Indirizzo IP del server
    port = 12345        # Porta del server

    # Creazione del socket TCP/IP
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connessione al server
    client_socket.connect((host, port))
    print(f"Connesso al server su {host}:{port}")

    while True:
        # Invio di un messaggio al server
        message = input("Inserisci il messaggio: ")
        client_socket.send(message.encode())

        # Ricezione della risposta dal server
        data = client_socket.recv(1024).decode()
        print(f"Risposta dal server: {data}")

    # Chiusura del socket
    client_socket.close()

if __name__ == "__main__":
    main()
