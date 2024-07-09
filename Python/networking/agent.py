#! /usr/bin/env python3

import requests

def send_request():
  url = "http://127.0.0.1:8000"

  try:
    response = requests.get(url)
    if response.status_code == 200:
      print("Server answer >>> ", response.text)
    else:
      print("Error: ", response.status_code)
  except requests.RequestException as e:
    print("Request error >>> ", e)

if __name__ == "__main__":
  send_request()

