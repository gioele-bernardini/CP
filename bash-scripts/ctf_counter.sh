#!/usr/bin/env bash

counter_file="counter.txt"

if [ ! -f "$counter_file" ]; then
  echo 0 > "$counter_file"
fi

count=$(cat "$counter_file")

count=$((count +1))

echo "$count" > "$counter_file"

echo "CTF solved: $count"

