#!/usr/bin/env python3

import socket

def main():
    host = '127.0.0.1'  # Indirizzo IP del server
    port = 12345        # Porta su cui il server ascolta

    # Creazione del socket TCP/IP
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Collegamento del socket all'indirizzo e alla porta
    server_socket.bind((host, port))

    # Il server inizia ad ascoltare
    server_socket.listen(1)

    print(f"Server in ascolto su {host}:{port}")

    # Accettazione della connessione
    client_socket, client_address = server_socket.accept()
    print(f"Connessione accettata da {client_address}")

    while True:
        # Ricezione dei dati dal client
        data = client_socket.recv(1024).decode()
        if not data:
            break
        print(f"Ricevuto dal client: {data}")

        # Invio di una risposta al client
        message = input("Inserisci la risposta: ")
        client_socket.send(message.encode())

    # Chiusura del socket
    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    main()
