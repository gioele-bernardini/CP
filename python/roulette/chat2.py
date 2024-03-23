#!/usr/bin/env python3

import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# Credenziali per il login
username = 'tocav10931'
# password = open('password.txt', 'r').read().strip()
password = 'balenciaga10'
base_url = 'https://instagram.com'
target = 'patrick_perini'
message = 'NO CAPPIN`'

chrome_options = Options()
# chrome_options.add_argument("--headless")

driver = webdriver.Chrome(options=chrome_options)

driver.get("https://www.instagram.com/")

driver.find_element(By.XPATH, '/html/body/div[4]/div[1]/div/div[2]/div/div/div/div/div[2]/div/button[2]').click()

username_form = driver.find_element(
  By.XPATH,
  '/html/body/div[2]/div/div/div[2]/div/div/div[1]/section/main/article/div[2]/div[1]/div[2]/form/div/div[1]/div/label/input'
)
password_form = driver.find_element(
  By.XPATH,
  '/html/body/div[2]/div/div/div[2]/div/div/div[1]/section/main/article/div[2]/div[1]/div[2]/form/div/div[2]/div/label/input'
)

time.sleep(4)
username_form.clear(); password_form.clear()
username_form.send_keys(username)
password_form.send_keys(password); password_form.send_keys(Keys.RETURN)

time.sleep(4)
driver.find_element(
  By.XPATH,
  '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[2]/section/main/div/div/div/div'
).click()

time.sleep(4)
driver.find_element(
  By.XPATH,
  '/html/body/div[3]/div[1]/div/div[2]/div/div/div/div/div[2]/div/div/div[3]/button[2]'
).click()

time.sleep(4)
driver.find_element(
  By.XPATH,
  '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[1]/div/div/div/div/div[2]/div[5]/div/div/div/span'
).click()

time.sleep(4)
driver.find_element(
  By.XPATH,
  '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[2]/section/div/div/div/div[1]/div/div[2]/div/div/div/div[4]/div'
).click()

time.sleep(3)
target_form = driver.find_element(
  By.XPATH,
  '/html/body/div[6]/div[1]/div/div[2]/div/div/div/div/div/div/div[1]/div/div[2]/div/div[2]/input'
)
target_form.clear(); target_form.send_keys(target)

time.sleep(2)
driver.find_element(
  By.XPATH,
  '/html/body/div[6]/div[1]/div/div[2]/div/div/div/div/div/div/div[1]/div/div[3]/div/div/div/div[1]/div/div/div[2]'
).click()

time.sleep(1)
driver.find_element(
  By.XPATH,
  '/html/body/div[6]/div[1]/div/div[2]/div/div/div/div/div/div/div[1]/div/div[4]/div'
).click()

time.sleep(2)
message_form = driver.find_element(
  By.XPATH,
  '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[2]/section/div/div/div/div[1]/div/div[2]/div/div/div/div/div/div/div[2]/div/div/div[2]/div/div/div[2]/div/div[1]/p'
)
message_form.clear(); message_form.send_keys(message)

time.sleep(4)
driver.find_element(
  By.XPATH,
  '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[2]/section/div/div/div/div[1]/div/div[2]/div/div/div/div/div/div/div[2]/div/div/div[2]/div/div/div[3]'
).click()



input('')
