from bs4 import BeautifulSoup
import requests

prof_ids = {"6313": ["1377273", "2324103", "3105814"],
            "6350": ["2038564"],
            "6363": ["194919", "918305", "462041", "2092131", "2516353"],
            "6375": ["1837370", "2566277", "3103949", "2369676"],
            "6320": ["2844042", "3077846"],
            "6360": ["1530329", "1936866", "2712933"],
            }

course_to_prof = {}
def fetch_page_content(url):
    """
    Fetches the HTML content from a given URL, using headers
    to mimic a real browser and avoid a 403 Forbidden error.
    """
    # These headers are crucial for this specific website
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.ratemyprofessors.com/',
        'Connection': 'keep-alive'
    }

    try:
        # Add the headers to your request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching page content: {e}")
        return None
    
def parse_html(content):
    """
    Parses the HTML to find the overall rating.
    
    NOTE: The class name you provided might change.
    You will need to update it if the script breaks.
    """
    soup = BeautifulSoup(content, 'html.parser')
    
    # This is the class name you provided.
    # Use your browser's "Inspect" tool to find the new one if this fails.
    class_rating = 'RatingValue__Numerator-qw8sqy-2 duhvlP'
    review_class = 'FeedbackItem__FeedbackNumber-uof32n-1 ecFgca'
    name_class = 'NameTitle__NameWrapper-dowf0z-2 cSXRap'
    
    try:
        rating_element = soup.find('div', class_=class_rating)

        if rating_element:
            print(f"Found rating: {rating_element.text}")
        else:
            print(f"Could not find an element with class: {class_rating}")
            print("The website's HTML may have changed. Please use 'Inspect' to find the new class name.")
            
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
    
    try:
        review_class_element = soup.find_all('div', class_=review_class)
        reviews = [elem.text for elem in review_class_element]
        would_take = reviews[0] if len(reviews) > 0 else "N/A"
        difficulty = reviews[1] if len(reviews) > 1 else "N/A"
        print(f"Found review elements: {reviews}")
    except Exception as e:
        print(f"An error occurred while fetching review elements: {e}")
    
    try:
        name_element = soup.find('h1', class_=name_class)
        if name_element:    
            print(f"Found name: {name_element.text}")
    except Exception as e:
        print(f"An error occurred while fetching name element: {e}")
    
    return name_element.text, rating_element.text, would_take, difficulty


# --- Main execution ---
for course_id, prof_id in prof_ids.items():
    print(f"Scraping course ID: {course_id}")
    for prof_id in prof_id:
        url_to_scrape = f"https://www.ratemyprofessors.com/professor/{prof_id}"
        print(f"Attempting to scrape: {url_to_scrape}")

        content = fetch_page_content(url_to_scrape)

        if content:
            print("Page content fetched successfully. Parsing...")
            name, rating, would_take, difficulty = parse_html(content)
            course_to_prof.setdefault(course_id, []).append({
                "prof_id": prof_id,
                "name": name,
                "rating": rating,
                "would_take_again": would_take,
                "difficulty": difficulty
            })
        else:
            print("Failed to fetch page content. Check the error message above (it was likely a 403).")

print("Final scraped data: ", course_to_prof)
with open("course_to_prof.json", "w") as f:
    import json
    json.dump(course_to_prof, f, indent=4)