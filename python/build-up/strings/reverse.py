#!/usr/bin/env python3

import sys

def reverse_period(period):
  out = period.split()[::-1]
  l = []

  for i in out:
    l.append(i)

  print(' '.join(l))

period = input('Please insert your period: \n')
if len(period.strip()) == 0:
  sys.stderr.write('Please provide a valid period\n')
  exit(1)

reverse_period(period)