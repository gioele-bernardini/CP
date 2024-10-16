#include <ncurses.h> // Libreria per le funzioni ncurses
#include <stdio.h>
#include <string.h>

#define TARGET "ciao"
#define TARGET_LEN 4

int main() {
    char buffer[TARGET_LEN + 1] = {0}; // Buffer per memorizzare l'input, incluso il terminatore nullo
    int index = 0;

    // Inizializza ncurses
    initscr();          // Avvia la modalità curses
    cbreak();           // Disabilita il buffering della linea, passa ogni carattere
    noecho();           // Non visualizza i caratteri digitati
    keypad(stdscr, TRUE); // Abilita l'input di tasti speciali

    printw("Digita 'ciao' per una risposta immediata:\n");
    refresh();

    while (1) {
        int ch = getch();  // Ottiene un singolo carattere senza attendere l'Invio

        // Gestione dell'input speciale (es. tasti freccia)
        if (ch == KEY_RESIZE) {
            // Puoi gestire il ridimensionamento del terminale qui se necessario
            continue;
        }

        // Ignora input non stampabili
        if (ch == ERR || ch < 32 || ch > 126) {
            continue;
        }

        char c = (char)ch;
        printw("%c", c);   // Stampa il carattere sullo schermo
        refresh();         // Aggiorna lo schermo

        // Aggiungi il carattere al buffer
        buffer[index] = c;
        index = (index + 1) % TARGET_LEN;
        buffer[TARGET_LEN] = '\0'; // Assicura la terminazione nulla

        // Costruisci la stringa corrente dal buffer
        char temp[TARGET_LEN + 1] = {0};
        for (int i = 0; i < TARGET_LEN; i++) {
            temp[i] = buffer[(index + i) % TARGET_LEN];
        }

        // Confronta la stringa corrente con "ciao"
        if (strcmp(temp, TARGET) == 0) {
            printw("\nHai digitato 'ciao'! Saluti!\n");
            refresh(); // Assicura che il messaggio sia visualizzato

            // Attende una pressione di tasto prima di uscire
            printw("Premi un tasto qualsiasi per uscire.\n");
            refresh();
            getch(); // Attende l'input dell'utente
            break;
        }
    }

    // Termina la modalità ncurses
    endwin();
    return 0;
}
