#include <stdio.h>
#include <string.h>

#define MAX_LEN 256

void remove_duplicates(char str[]) {
    int seen[256] = {0}; // Array per tracciare i caratteri visti (valori ASCII)
    int i;

    for (i = 0; str[i] != '\0'; i++) {
        // Ottieni il valore ASCII del carattere corrente
        unsigned char current_char = str[i];

        // Se il carattere non è stato visto prima, stampalo
        if (!seen[current_char]) {
            printf("%c", current_char);
            seen[current_char] = 1; // Marca il carattere come visto
        }
    }
}

int main() {
    char input[MAX_LEN];

    // Leggi la stringa dall'input
    printf("Inserisci una stringa (max 255 caratteri): ");
    fgets(input, MAX_LEN, stdin);

    // Rimuovi il newline (se presente)
    input[strcspn(input, "\n")] = '\0';

    printf("Stringa senza duplicati: ");
    remove_duplicates(input);

    printf("\n");
    return 0;
}
