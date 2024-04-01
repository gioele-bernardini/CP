#!/usr/bin/env python3

def count_elements_recursion(list):
  if not list:
    return 0
  
  return 1 + count_elements_recursion(list[1:])

list = [10, 20, 30]
print(f'List length is => [{len(list)}]')

counter = 0
for el in list:
  counter += 1

print(counter)

# Stampiamo la lista con str(lista)
print('The list is:' + str(list))

list_len = sum(1 for _ in list)
print(list_len)