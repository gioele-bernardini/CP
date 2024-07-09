#!/usr/bin/env python3

import math

class Solution:
  @staticmethod
  def k_divider(n: int, k: int) -> int:
    max = int(math.sqrt(n) +1)
    for i in range(1, max):
      if n % i == 0:
        k -= 1
        if k == 0:
          return i
 
if __name__ == '__main__':
  target = 20
  k = 3

  out = Solution.k_divider(target, k)
  print(out)
