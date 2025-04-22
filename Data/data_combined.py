import json
import csv
import re
import requests

# Checking emails
def is_valid_email(email):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

# Extracting and checkinhg phone number
def is_valid_phone(phone):
    digits = re.sub(r"\D", "", phone)
    return len(digits) >= 10

# Sends a GET request to verify if the LinkedIn link is reachable
def is_valid_linkedin(link):
    try:
        response = requests.get(link, timeout=5)
        return response.status_code == 200
    except:
        return False


def process_lever(data):
    candidate = {}
    candidate["Name"] = data.get("name", "")
    
    emails = data.get("emails", [])
    candidate["Email"] = emails[0] if emails and is_valid_email(emails[0]) else "Invalid"

    candidate["Title"] = data.get("headline", "")

    linkedin_links = [link for link in data.get("links", []) if "linkedin.com" in link]
    candidate["LinkedIn Ad"] = linkedin_links[0] if linkedin_links else "Not provided"
    candidate["Is LinkedIn Valid?"] = "Unknown"  # We’ll update this later if needed

    phones = data.get("phones", [])
    raw_phone = phones[0]["value"] if phones and "value" in phones[0] else ""
    candidate["Phone"] = raw_phone if is_valid_phone(raw_phone) else "Invalid"

    # Placeholder values when info is missing
    candidate["Company"] = "Unknown"
    candidate["Job"] = "Unknown"
    candidate["Location"] = data.get("location", "")

    return candidate


def process_workable(data):
    candidate = {}
    candidate["Name"] = data.get("name", "")
    
    email = data.get("email", "")
    candidate["Email"] = email if is_valid_email(email) else "Invalid"

    candidate["Title"] = data.get("headline", "")

    linkedin_profiles = [
        profile.get("url", "")
        for profile in data.get("social_profiles", [])
        if "linkedin.com" in profile.get("url", "")
    ]
    candidate["LinkedIn Ad"] = linkedin_profiles[0] if linkedin_profiles else "Not provided"
    candidate["Is LinkedIn Valid?"] = "Unknown"

    phone = data.get("phone", "")
    candidate["Phone"] = phone if is_valid_phone(phone) else "Invalid"

    candidate["Company"] = data.get("account", {}).get("name", "")
    candidate["Job"] = data.get("job", {}).get("title", "")
    candidate["Location"] = data.get("location", {}).get("location_str", "")

    return candidate

# Load JSON files and extract structured candidate data
with open("lever.json", "r", encoding="utf-8") as f:
    lever_json = json.load(f)
    lever_candidate = process_lever(lever_json["data"])

with open("workable.json", "r", encoding="utf-8") as f:
    workable_json = json.load(f)
    workable_candidate = process_workable(workable_json["candidate"])

# Writing the combined candidate data into a CSV file
with open("candidates.csv", "w", newline='', encoding="utf-8") as f:
    fieldnames = ["Name", "Email", "Title", "LinkedIn Ad", "Is LinkedIn Valid?", "Phone", "Company", "Job", "Location"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(lever_candidate)
    writer.writerow(workable_candidate)
