#!/usr/bin/env python3

def reverse_period(period):
  out = period.split()[::-1]
  l = []

  for i in out:
    l.append(i)

  print(' '.join(l))

period = input('Please insert your period: \n')
reverse_period(period)