from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.service import Service # Import Service
import time

# Path to Geckodriver
geckodriver_path = r"C:\Users\03aay\Downloads\geckodriver-v0.34.0-win64\geckodriver.exe"
firefox_options = Options()
firefox_options.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

service = Service(geckodriver_path)
driver = webdriver.Firefox(service=service, options=firefox_options)

driver.get('https://go.drugbank.com/food-interaction-checker#results')
try:
    # Wait for the input field to be present
    wait = WebDriverWait(driver, 10)  # Increased timeout to 20 seconds
    input_box = wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/main/div/div[3]/div/div[1]/form/div[1]/div/span/span[1]/span/ul/li/input')))
    
    # Input the text (for example, "Aspirin")
    text_to_input = 'tylenol'  # Change this to the text you want to input
    input_box.clear()  # Clear the input field before typing
    input_box.send_keys(text_to_input)
    time.sleep(10)
    input_box.send_keys(Keys.RETURN) 

    # Wait for the submit button to be clickable
    submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/main/div/div[3]/div/div[1]/form/div[5]/button')))
    
    # Scroll the submit button into view to ensure it's not covered
    driver.execute_script("arguments[0].scrollIntoView(true);", submit_button)
    
    # Click the submit button
    submit_button.click()

    # Wait for the results to be visible
    result_xpath = '/html/body/main/div/div[3]/div/div[2]'
    wait.until(EC.visibility_of_element_located((By.XPATH, result_xpath)))

    # Extract and print the result
    result_element = driver.find_element(By.XPATH, result_xpath)
    print(result_element.text)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the browser after scraping
    driver.quit()




