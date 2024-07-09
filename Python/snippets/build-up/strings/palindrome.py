#!/usr/bin/env python3

def is_palindrome3(word):
  half = len(word) // 2

  # codice per verificare se sia SIMMETRICA
  # NON PALINDROMA
  first_str = word[:half]

  if len(word) % 2 == 0:
    second_str = word[half:]
  else:
    second_str = word[half +1:]
  # return first_str == second_str

  return first_str == second_str[::-1]

def is_palindrome2(word):
  return word == ''.join(reversed(word))

def is_palindrome(word):
  for i in range(len(word) // 2):
    if word[i] != word[-i -1]:
      return False
    
  return True

word = input('Insert your word => ')
print(is_palindrome3(word))