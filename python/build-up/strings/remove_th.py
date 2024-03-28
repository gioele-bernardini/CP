#!/usr/bin/env python3

def remove_th(string, i):
  if i < 0 or i >= len(string):
    return 'Index out of range'
  
  return string[:i] + string[i+1:]
  # return string.replace(string[i], '')

user_string = input('Hi! Please provide a string >>> ')
user_index = int(input('Gotcha! Now please provide the number >>> '))

print(remove_th(user_string, user_index))
