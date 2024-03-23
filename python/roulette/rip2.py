#!/usr/bin/env python3

import random
from colorama import *
import sys
import argparse

import chat


LIST = ['sasso', 'carta', 'forbice']

def winner(usr, pc):
  selections = {'sasso': 0, 'carta': 1, 'forbice': 2}
  if usr == pc:
      print('TIE')
  
  # l'elemento successivo (modulo) vince su quello prima
  if (selections[usr] == (selections[pc] + 1) % 3):
      print(Fore.GREEN + 'LUCKY BOY' + Style.RESET_ALL)
  else:
      print(Fore.RED + 'YOU LOSE' + Style.RESET_ALL)
      chat.bye_bye()
    
def main():
  parser = argparse.ArgumentParser(description='russian roulette')
  parser.add_argument('--no-pussy', action='store_true', help='Start the game')
  args = parser.parse_args()

  if not args.no_pussy:
    print('Non e` un gioco per pussy, torna quando sarai un uomo')
    exit(0)

  usr = input('Scegli sasso, carta o forbice >>> ')

  if usr not in LIST:
    sys.stderr.write(Fore.RED + 'Scelta non valida!\n' + Style.RESET_ALL)
    exit(1)
  else:
    # pc = random.choice(LIST)
    pc = LIST[1]
    print(f'PC => {pc}\nUSR =>{usr}')
    print(winner(usr, pc))

if __name__ == '__main__':
   main()
