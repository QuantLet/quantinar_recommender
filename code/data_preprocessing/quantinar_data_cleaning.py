import json
import csv
import re
import codecs
import os
from collections import Counter

def extract_description(content):
    description_match = re.search(r'### Description\n\n(.*?)(?=\n\n###|\Z)', content, re.DOTALL)
    if description_match:
        return description_match.group(1).strip()
    return ''

def clean_text(text):
    # Replace common HTML entities
    text = text.replace('&auml;', 'ä').replace('&uuml;', 'ü').replace('&ouml;', 'ö')
    text = text.replace('&Auml;', 'Ä').replace('&Uuml;', 'Ü').replace('&Ouml;', 'Ö')
    text = text.replace('&szlig;', 'ß')
    
    # Remove any remaining HTML tags
    text = re.sub('<[^<]+?>', '', text)
    
    return text

def extract_instructors(content):
    instructor_pattern = r'\[([^\]]+)\]\([^\)]+\s+"instructor"\)'
    matches = re.findall(instructor_pattern, content)
    return matches  # Return only instructor names, without links

def extract_last_updated(content):
    last_updated_match = re.search(r'Last Updated\s*:\s*([^#\n\]]+)', content)
    if last_updated_match:
        date_str = last_updated_match.group(1).strip()
        # Remove any trailing characters that are not part of the date
        date_str = re.sub(r'[^\w\s]$', '', date_str)
        return date_str
    return ''

def extract_course_data(json_data):
    courses = []
    total_students = 0
    total_rating = 0
    total_rated_courses = 0
    instructors_counter = Counter()
    
    for item in json_data:
        for doc in item['docs']:
            metadata = doc.get('metadata', {})
            content = clean_text(doc.get('content', ''))
            
            title = metadata.get('ogTitle', '').strip()
            url = metadata.get('ogUrl', '').strip()
            description = clean_text(extract_description(content)).strip()
            
            if not title or not url or not description:
                # Skip empty or incomplete entries
                continue
            
            last_updated = extract_last_updated(content)
            instructors = extract_instructors(content)
            instructors_str = '; '.join(instructors)
            for instructor in instructors:
                instructors_counter[instructor] += 1
            
            students_enrolled_match = re.search(r'(\d+)\s*Students Enrolled', content)
            students_enrolled = students_enrolled_match.group(1) if students_enrolled_match else '0'
            total_students += int(students_enrolled)
            
            rating_match = re.search(r'(\d+(\.\d+)?)\s+out of\s+5\s+stars', content)
            if rating_match:
                rating = float(rating_match.group(1))
                total_rating += rating
                total_rated_courses += 1
            
            courses.append({
                'title': title,
                'url': url,
                'description': description,
                'last_updated': last_updated if last_updated else 'N/A',
                'instructors': instructors_str if instructors_str else 'N/A',
                'students_enrolled': students_enrolled
            })
    
    total_courses = len(courses)
    avg_rating = total_rating / total_rated_courses if total_rated_courses > 0 else 0

    # Print summary information
    print(f"Total number of courses: {total_courses}")
    print(f"Total number of enrolled students: {total_students}")
    print(f"Average course rating: {avg_rating:.2f}")
    print("\nTop 5 instructors by number of courses:")
    for instructor, count in instructors_counter.most_common(5):
        print(f"{instructor}: {count} courses")
    
    return courses

def save_to_csv(courses, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['title', 'url', 'description', 'last_updated', 'instructors', 'students_enrolled']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for course in courses:
            writer.writerow(course)

# Load JSON data from the updated location
json_file_path = os.path.join('data', 'raw', 'quantinar_courses_raw.json')

with codecs.open(json_file_path, 'r', encoding='utf-8-sig') as file:
    json_data = json.load(file)

# Extract course data
courses = extract_course_data(json_data)

# Define the CSV file path
csv_file_path = os.path.join('data', 'raw', 'course_data.csv')

# Save to CSV
save_to_csv(courses, csv_file_path)

print(f"Data has been saved to {csv_file_path}")
