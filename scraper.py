"""
SHL Assessment Catalog Scraper
Scrapes individual test solutions from SHL product catalog
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict
import re
import os

max_pages = 32

class SHLScraper:
    def __init__(self):
        self.base_url = "https://www.shl.com"
        self.catalog_url = f"{self.base_url}/solutions/products/product-catalog/?type=1"
        self.assessments = []

    def scrape_catalog(self) -> List[Dict]:
        """
        Scrape all individual test solutions from SHL catalog
        Returns list of assessment dictionaries
        """
        print("Starting to scrape SHL catalog...")
        
        page = 1
        total_collected = 0
        while True:
            if page > max_pages:
                print(f"Reached page limit ({max_pages}). Stopping.")
                break
            
            print(f"Scraping catalog page {page}...")
            url = f"{self.base_url}/solutions/products/product-catalog/?start={(page-1)*12}&type=1"
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                print(f"Failed to load page {page}")
                break

            soup = BeautifulSoup(response.content, "html.parser")
            
            tables = soup.find_all("table")
            individual_table = None
            for t in tables:
                header_cell = t.find("th")
                if header_cell and "Individual Test Solutions" in header_cell.get_text(strip=True):
                    individual_table = t
                    break
                
            if not individual_table:
                print("No 'Individual Test Solutions' table found on this page.")
                break
            
            rows = individual_table.find_all("tr")[1:]  # Skip header row

            # Collect assessment links
            assessment_urls = []
            for row in rows:
                a_tag = row.find("a", href=True)
                if not a_tag:
                    continue
                href = a_tag["href"].strip()
                if href.startswith("/products/product-catalog/view/"):
                    full_url = f"{self.base_url}{href}"
                    assessment_urls.append(full_url)

            if not assessment_urls:
                break

            print(f"Found {len(assessment_urls)} assessments on page {page}")

            # Visit each assessment page
            for url in assessment_urls:
                print(f"  → Scraping: {url}")
                data = self.scrape_assessment_page(url)
                if data:
                    self.assessments.append(data)
                    total_collected += 1
                time.sleep(0.5)

            # Check if there's a “Next” link
            pagination = soup.find("ul", class_="pagination")
            next_item = pagination.find("li", class_="pagination__item -arrow -next") if pagination else None
            if next_item and next_item.find("a", href=True):
                page += 1
            else:
                break

        print(f"Total assessments scraped: {total_collected}")
        self.save_assessments()
        return self.assessments

    def scrape_assessment_page(self, url: str) -> Dict:
        """Scrape an individual assessment page"""
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            name_tag = soup.find("h1")
            name = name_tag.text.strip() if name_tag else "Unknown"

            # Description
            desc_block = soup.find("div", class_="product-catalogue-training-calendar__row typ")
            description = ""
            if desc_block:
                p = desc_block.find("p")
                description = p.text.strip() if p else ""

            # Duration
            duration = "Unknown"
            dur_row = soup.find("h4", string=re.compile("Assessment length", re.I))
            if dur_row:
                dur_p = dur_row.find_next("p")
                if dur_p:
                    duration = dur_p.text.strip()

            # Test Types
            test_types = []
            test_type_paragraphs = soup.find_all("p", class_="product-catalogue__small-text")
            for p in test_type_paragraphs:
                if "Test Type" in p.get_text():
                    for sp in p.find_all("span", class_="product-catalogue__key"):
                        t = sp.text.strip()
                        if t in ["K", "P", "C"]:
                            test_types.append(t)
                    break  
        
            test_type = ",".join(set(test_types))
            if not test_type:
                return None

            # Skills (commented because very few skills are described there)
            #skills = self.extract_skills(description)

            return {
                "name": name,
                "url": url,
                "description": description,
                "test_type": test_type or "Unknown",
                "duration": duration,                      # can add skill too
            }

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def extract_skills(self, text: str) -> List[str]:
        """Simple keyword-based skill extraction"""
        if not text:
            return []
        skill_keywords = [
            "leadership", "communication", "sales", "negotiation", "management",
            "teamwork", "customer", "technical", "cognitive", "analytical",
            "problem solving", "interpersonal", "adaptability", "collaboration",
            "java", "python", "excel", "accounting", "finance"
        ]
        found = [s for s in skill_keywords if s.lower() in text.lower()]
        return found

    def save_assessments(self):
        """Save scraped assessments to a JSON file"""
        os.makedirs("data", exist_ok=True)
        with open("data/assessments.json", "w", encoding="utf-8") as f:
            json.dump(self.assessments, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(self.assessments)} assessments to data/assessments.json")


def main():
    scraper = SHLScraper()
    scraper.scrape_catalog()


if __name__ == "__main__":
    main()
