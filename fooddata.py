import requests
from bs4 import BeautifulSoup
import csv

base_url = "http://research.bmh.manchester.ac.uk"
main_url = f"{base_url}/informall/allergenic-foods/"

response = requests.get(main_url)
response.raise_for_status()  # Raise an error if the request fails

soup = BeautifulSoup(response.text, "html.parser")

food_list = soup.select("div:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(2) > div > ul > li > a")

data = []
for item in food_list:
    food_name = item.text.strip()
    relative_food_link = item.get("href")
    food_link = f"{base_url}{relative_food_link}" if relative_food_link else "N/A"

    if food_link != "N/A":
        try:
            food_response = requests.get(food_link)
            food_response.raise_for_status()
            food_soup = BeautifulSoup(food_response.text, "html.parser")

            allergy_info = food_soup.select_one("div:nth-of-type(3) > div > div:nth-of-type(1) > ul:nth-of-type(1) > li:nth-of-type(4) > span:nth-of-type(2)")
            if allergy_info:

                allergy_texts = [p.text.strip() for p in allergy_info.find_all("p")]
                allergy_text = " ".join(allergy_texts) if allergy_texts else "N/A"
            else:
                allergy_text = "N/A"
        except Exception as e:
            allergy_text = f"Error: {e}"
    else:
        allergy_text = "N/A"

    data.append([food_name, food_link, allergy_text])

output_file = "food_links_with_allergy.csv"
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["food", "link", "allergy"])
    writer.writerows(data)

print(f"Data has been successfully written to {output_file}")
