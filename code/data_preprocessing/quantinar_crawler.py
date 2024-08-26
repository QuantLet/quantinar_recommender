import os
import time
import uuid
from firecrawl.firecrawl import FirecrawlApp
import json

# Get API key from environment variable
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')

if not FIRECRAWL_API_KEY:
    raise ValueError("FIRECRAWL_API_KEY environment variable is not set")

# Initialize the Firecrawl app
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

def crawl_with_retry(url, params, max_retries=3, delay=60):
    for attempt in range(max_retries):
        try:
            # Generate a unique idempotency key
            idempotency_key = str(uuid.uuid4())
            
            # Initiate the crawl job
            print(f"Attempting to start crawl job (attempt {attempt + 1}/{max_retries})")
            crawl_result = app.crawl_url(url, params=params, wait_until_done=True, poll_interval=2, idempotency_key=idempotency_key)
            return crawl_result
        except Exception as e:
            print(f"Error occurred (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached.")
                raise

# Define the URL and parameters for crawling
url = 'https://quantinar.com/course'
params = {
    'crawlerOptions': {
        'limit':100,  # Increase the limit to ensure all pages are crawled
        'includes': ['/course'],  # Ensure it includes all pages under /course
        'depth': 2  # Adjust the depth to crawl linked pages from the main course page
    },
    'pageOptions': {
        'onlyMainContent': True,
        'followLinks': True  # Ensure the crawler follows links to other pages
    }
}

try:
    # Start the crawling process
    crawl_result = crawl_with_retry(url, params)

    if not crawl_result or not isinstance(crawl_result, list):
        print(f"Debug: Received response: {crawl_result}")
        raise ValueError("Received an invalid or empty response from the API")

    # Structure the result to include "docs" with "content", "markdown", and "metadata"
    structured_result = [{
        "docs": [
            {
                "content": doc.get("content", ""),
                "markdown": doc.get("markdown", ""),
                "metadata": doc.get("metadata", {}),
                "linksOnPage": doc.get("linksOnPage", [])
            } 
            for doc in crawl_result
        ]
    }]

    # Save the structured data to a JSON file
    json_filename = os.path.join('data', 'raw', 'quantinar_courses_raw.json')
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)

    with open(json_filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(structured_result, jsonfile, ensure_ascii=False, indent=4)

    print(f"Crawl completed. Structured JSON data saved to {json_filename}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check the troubleshooting steps or consult the Firecrawl documentation.")
