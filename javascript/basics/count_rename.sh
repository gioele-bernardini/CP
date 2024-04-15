#!/usr/bin/env bash

count=1

for file in start*.js; do
  mv "$file" "basics${count}.js"

  ((count++))
done
