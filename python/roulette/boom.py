#!/usr/bin/env python3

import time
import os
from colorama import init, Fore, Back, Style

ascii_draw = open('ascii.txt', 'r').read()

def shutdown():
  print(Style.BRIGHT + Fore.RED + Back.BLACK + ascii_draw + Style.RESET_ALL)

  time.sleep(2)
  os.system('shutdown 0')
