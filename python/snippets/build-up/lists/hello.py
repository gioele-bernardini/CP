#!/usr/bin/env python3

import sys

if len(sys.argv) != 2:
  sys.stderr.write(f'Usage: {sys.argv[0]}, <name>\n')
  exit(1)

print(f'Hello, {sys.argv[1]}!')