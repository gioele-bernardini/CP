#!/usr/bin/env python3

class Solution:
  def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
    return sum(all(letter not in brokenLetters for letter in word) for word in text.split())
