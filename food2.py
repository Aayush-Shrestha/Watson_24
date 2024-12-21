import csv
import requests
from lxml import html

def scrape_allergy(link):
    try:
        response = requests.get(link)
        
        if response.status_code == 200:
            tree = html.fromstring(response.content)

            xpath = "//span[contains(text(),'Allergy')]/ancestor::li/span[2]"
            result = tree.xpath(xpath)

            if result:
                return result[0].text_content().strip()
            else:
                return None
        else:
            print(f"Failed to fetch {link} with status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred while processing {link}: {e}")
        return None

input_csv = 'food_links_with_allergy.csv'
output_csv = 'food_links_with_allergy_updated.csv'

with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    
    fieldnames = reader.fieldnames
    
    updated_rows = []
    
    for row in reader:
        link = row['link']  
        
        allergy_info = scrape_allergy(link)
        
        row['allergy'] = allergy_info if allergy_info else "No allergy information found"
        
        updated_rows.append(row)

with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(updated_rows)

print(f"Scraped data saved to {output_csv}")
