#!/usr/bin/env python3

K = 2

def max_min1(test_tup: tuple):
  print('Origin tuple is >>> ' + str(test_tup))

  res = []
  test_tup = list(sorted(test_tup))

  for index, val in enumerate(test_tup):
    if index < K or index >= len(test_tup) -K:
      res.append(val)

  res = tuple(res)

  print("The extracted values >>> " + str(res))

def max_min2(test_tup: tuple):
  test_tup = list(test_tup)
  temp = sorted(test_tup)
  
  res = tuple(temp[:K] + temp[-K:])

  print("The extracted values >>> " + str(res))


if __name__ == '__main__':
  test_tup = (5, 20, 3, 7, 6, 8)

  max_min1(test_tup)
