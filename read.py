import requests
from bs4 import BeautifulSoup
import csv

# URL of the website
url = "http://research.bmh.manchester.ac.uk/informall/allergenic-foods/"

# Send a GET request to the URL
response = requests.get(url)
response.raise_for_status()  # Raise an error if the request fails

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

food_list = soup.select("div:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(2) > div > ul > li > a")

# Prepare the data for the CSV file
data = []
for item in food_list:
    food_name = item.text.strip()
    food_link = item.get("href")
    data.append([food_name, food_link])

# Write the data to a CSV file
output_file = "food_links.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["food", "link"])
    writer.writerows(data)

print(f"Data has been successfully written to {output_file}")
