#!/usr/bin/env python3

import os

def update_counter():
    counter_file = "counter.txt"
    
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as file:
            file.write("0")
    
    with open(counter_file, "r") as file:
        count = int(file.read().strip())
    
    count += 1
    
    with open(counter_file, "w") as file:
        file.write(str(count))
    
    print(f"CTF risolte: {count}")

if __name__ == "__main__":
    update_counter()

