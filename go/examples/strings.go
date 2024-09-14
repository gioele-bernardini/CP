package main

import (
	"fmt"
	"strings"
)

var pl = fmt.Println

func main() {
	sV1 := "A word"

	replace := strings.NewReplacer("A", "Another")
	sV2 := replace.Replace(sV1)
	pl(sV2)
	pl("Contains Another:", strings.Contains(sV2, "Another"))
	// strings.Index(sV2, "o")

	sV3 := "\nSome Words\n"
	sV3 = strings.TrimSpace(sV3)
	pl("Split:", strings.Split("a-b-c-d", "-"))
	// strings.ToLower("Ciao!")
}

// Definizione della struct Replacer
// struct Replacer {
//     from: String,
//     to: String,
// }

// Implementazione della struct Replacer
// impl Replacer {
//     // Funzione per creare un nuovo oggetto Replacer
//     fn new(from: &str, to: &str) -> Replacer {
//         Replacer {
//             from: from.to_string(),
//             to: to.to_string(),
//         }
//     }

//     // Metodo per sostituire le occorrenze della stringa 'from' con 'to'
//     fn replace(&self, input: &str) -> String {
//         input.replace(&self.from, &self.to)
//     }
// }

// fn main() {
//     let s_v1 = "A word";  // Stringa originale

//     // Crea un nuovo Replacer che sostituisce "A" con "Another"
//     let replacer = Replacer::new("A", "Another");

//     // Applica la sostituzione
//     let s_v2 = replacer.replace(s_v1);

//     // Stampa il risultato
//     println!("{}", s_v2);  // Stampa: "Another word"
// }
