#!/usr/bin/env python3

def summation1(test_sup: tuple):
  test = list(test_sup)
  count = 0

  for _ in test:
    count += 1

  return count

def summation2(test_tup: tuple):
  # map serve per sommare eventuali valori enlla tupla
  # che siano a loro volta liste o sottotuple
  res = sum(list(map(sum, list(test_tup))))
