from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options = Options()
# chrome_options.add_argument("--headless")

driver = webdriver.Chrome(options=chrome_options)

driver.get("https://www.instagram.com/")

# Attendi fino a quando il popup dei cookie non è visibile
try:
    cookie_popup = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, "//button[text()='Decline optional cookies']")))
    cookie_popup.click()
except:
    print("Popup dei cookie non trovato o timeout scaduto")


input('')
driver.quit()
