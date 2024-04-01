#!/usr/bin/env python3

def swapList(newList):
  size = len(newList)

  # Swapping
  temp = newList[0]
  newList[0] = newList[size -1]
  newList[size - 1] = temp

  # Second approach
  # newList[0], newList[-1] = newList[-1], newList[0]
  # return newList

  # Third approach
  # get = list[-1], list[0]
  # list[0], list[-1] = get
  # return list

  return newList

# Driver code
newList = [12, 35, 9, 56, 24]

print(swapList(newList))