#!/usr/bin/env python3

def print_even_words(period):
  words = period.split()

  i = 1
  # for index, word in enumerate(words):
  for word in words:
    if i % 2 == 0:
      print(f'{word} ')

    i += 1

string = 'ciao sono una frase normalissima'
print_even_words(string)