# realestate_processor.py

import os
import csv
import json
import re
import pandas as pd
import pdfplumber
import openai
from rapidfuzz import process, fuzz
from typing import List, Dict, Tuple
import streamlit as st


# ---------------------------
# Configuration & API Key
# ---------------------------
openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------
# ADDRESS EXTRACTION FUNCTIONS
# ---------------------------
address_schema = {
    "name": "address_schema",
    "schema": {
        "type": "object",
        "properties": {
            "addresses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Street Address": {"type": "string"},
                        "Unit": {"type": "string"},
                        "City": {"type": "string"},
                        "State": {"type": "string"},
                        "Zip Code": {"type": "string"}
                    },
                    "required": ["Street Address", "City", "State", "Zip Code"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["addresses"],
        "additionalProperties": False
    }
}


def extract_addresses_with_ai(text: str) -> List[dict]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a highly accurate data extraction assistant. "
                "Your task is to extract all real estate addresses from unstructured text. "
                "Ensure the addresses are correctly formatted and structured according to the provided schema. "
                "If an address has a unit number, include it separately in the 'Unit' field. "
                "Return all addresses in a structured list format inside an 'addresses' object."
            )
        },
        {
            "role": "user",
            "content": (
                "Extract all property addresses from the following text. "
                "Ensure that the output strictly follows the JSON schema provided. "
                "Text:\n\n" + text + "\n\n" +
                "Return ONLY valid addresses in JSON format."
            )
        }
    ]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_schema", "json_schema": address_schema},
            temperature=0
        )
        structured_response = response.choices[0].message.content
        parsed_data = json.loads(structured_response)
        return parsed_data.get("addresses", [])
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        return []


def extract_addresses_with_ai_chunked(text: str, max_lines: int = 50) -> List[dict]:
    """Break large text into chunks before sending to the AI for extraction."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return extract_addresses_with_ai(text)
    else:
        addresses = []
        for i in range(0, len(lines), max_lines):
            chunk = "\n".join(lines[i:i+max_lines])
            result = extract_addresses_with_ai(chunk)
            addresses.extend(result)
        return addresses


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
    return " ".join(text.split())


def extract_text_from_csv(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                text += " ".join(row) + "\n"
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
    return text.strip()


def extract_addresses_from_excel(file_path: str, logger=None) -> List[dict]:
    extracted_addresses = []
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        # Process sheets in chunks of 3
        if len(sheet_names) > 3:
            sheet_chunks = [sheet_names[i:i+3] for i in range(0, len(sheet_names), 3)]
        else:
            sheet_chunks = [sheet_names]
        for chunk in sheet_chunks:
            group_text = ""
            for sheet_name in chunk:
                df = pd.read_excel(xls, sheet_name)
                for col in df.columns:
                    if "address" in col.lower() or "property" in col.lower() or "location" in col.lower():
                        group_text += "\n".join(df[col].dropna().astype(str).tolist()) + "\n"
            if logger:
                logger(f"Processing a group of sheets from Excel file {os.path.basename(file_path)}")
            chunk_addresses = extract_addresses_with_ai_chunked(group_text, max_lines=50)
            extracted_addresses.extend(chunk_addresses)
    except Exception as e:
        if logger:
            logger(f"Error processing Excel {os.path.basename(file_path)}: {e}")
        return []
    return extracted_addresses


def extract_and_save_addresses(file_paths: List[str], output_file: str, logger=None):
    """Extracts addresses from given files and saves them to a CSV output_file."""
    all_extracted_addresses = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[-1].lower()
        file_name = os.path.basename(file_path)
        if ext == ".pdf":
            if logger:
                logger(f"Processing PDF for addresses: {file_name}")
            text = extract_text_from_pdf(file_path)
            addresses = extract_addresses_with_ai_chunked(text, max_lines=50)
        elif ext == ".csv":
            if logger:
                logger(f"Processing CSV for addresses: {file_name}")
            text = extract_text_from_csv(file_path)
            addresses = extract_addresses_with_ai_chunked(text, max_lines=50)
        elif ext in [".xls", ".xlsx"]:
            if logger:
                logger(f"Processing Excel for addresses: {file_name}")
            addresses = extract_addresses_from_excel(file_path, logger=logger)
        else:
            if logger:
                logger(f"Unsupported file type for address extraction: {file_name}")
            continue
        all_extracted_addresses.extend(addresses)

    try:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Street Address", "Unit", "City", "State", "Zip Code"])
            writer.writeheader()
            for address in all_extracted_addresses:
                writer.writerow(address)
        if logger:
            logger(f"Extracted addresses saved to {output_file}")
    except Exception as e:
        if logger:
            logger(f"Error saving extracted addresses: {e}")

# ---------------------------
# MERGE & CLASSIFICATION FUNCTIONS
# ---------------------------

BROKERAGE_EMAIL_DOMAINS = {
    "compass.com", "coldwellbanker.com", "century21.com",
    "sothebysrealty.com", "corcoran.com", "betterhomesandgardens.com",
    "era.com", "berkshirehathawayhs.com", "edinarealty.com", "longandfoster.com",
    "interorealestate.com", "realtysouth.com", "harrynorman.com",
    "exp.com", "douglaselliman.com", "howardhanna.com",
    "allentate.com", "randrealty.com", "coachrealtors.com",
    "atproperties.com", "christiesrealestate.com", "homesmart.com",
    "unitedrealestate.com", "pearsonsmithrealty.com", "virtualpropertiesrealty.com",
    "benchmarkrealtytn.com", "remax.com", "remaxgold.com", "elliman.com",
    
    # Newly Added Domains
    "kleiers.com", "bouklisgroup.com", "triplemint.com", "bhsusa.com",
    "brownstonelistings.com", "kw.com", "dwellresidentialny.com", "evrealestate.com",
    "sothebys.realty", "blunyc.com", "lgfairmont.com", "firstbostonrealty.com",
    "elegran.com", "wernewyork.com", "corcoransunshine.com", "serhant.com",
    "warburgrealty.com", "nestseekers.com", "officialpartners.com", "livingny.com",
    "onesothebysrealty.com", "undividedre.com", "spiralny.com", "platinumpropertiesnyc.com",
    "theagencyre.com", "mrgnyc.com", "ostrovrealty.com", "bhhsnyp.com",
    "trumpintlrealty.com", "reuveni.com", "sir.com", "casa-blanca.com",
    "maxwelljacobs.com", "simplerealestatenyc.com", "levelgroup.com", "kwnyc.com",
    "corenyc.com", "raveis.com", "bohemiarealtygroup.com", "avenuesir.com",
    "rosenyc.com", "halstead.com", "foxresidential.com", "hellerorg.com",
    "rutenbergnyc.com", "cantorpecorella.com", "brgnyc.com", "citihabitats.com",
    "andrewsbc.com", "urbancompass.com", "mercedesberk.com", "realogy.com",
    "metropolitanpropertygroup.com", "brodskyorg.com", "stonehengenyc.com",
    "manhattanmiami.com", "newmarkkf.com", "bondnewyork.com", "primemanhattan.com",
    "sothebyshomes.com", "broadwayrealty.com", "lionsgroupnyc.com", "irnewyork.com",
    "tungstenproperty.com", "platinumluxuryauctions.com", "dollylenz.com",
    "christies.com", "continentalrealestate.com", "lesliegarfield.com",
    "rubinteam.com", "vestanewyork.com", "brooklynproperties.com", "aaronkirman.com",
    "triumphproperty.com", "coopercooper.com", "windermere.com", "spiregroupny.com",
    "stationcities.com", "onemanhattanre.com", "awmrealestate.com", "modernspacesnyc.com",
    "trumporg.com", "montskyrealestate.com", "yoreevo.com", "homedax.com",
    "peterashe.com", "miradorrealestate.com", "garfieldbrooklyn.com",
    "reserverealestate.com", "fillmore.com", "wfp.com", "blessoproperties.com",
    "digsrealtynyc.com", "real212.com", "oldendorpgroup.com", "vanguardchelsea.com",
    "newamsterdambrokerage.com", "engelvoelkers.com", "remaxedgeny.com", "cbrnyc.com",
    "mkrealtyny.com", "remax.net", "expansionteamnyc.com", "kwcommercial.com",
    "kwrealty.com", "thebestapartments.com", "moravillarealty.com",
    "remax-100-ny.com", "cushwake.com", "moiniangroup.com"
}
VENDOR_KEYWORDS = ["title", "mortgage", "lending", "escrow"]

def is_real_estate_agent(emails: List[str]) -> bool:
    for email in emails:
        match = re.search(r"@([\w.-]+)$", email)
        if match:
            domain = match.group(1).lower()
            if domain in BROKERAGE_EMAIL_DOMAINS:
                return True
    return False


def is_vendor(emails: List[str]) -> bool:
    for email in emails:
        match = re.search(r"@([\w.-]+)$", email)
        if match:
            domain = match.group(1).lower()
            if any(keyword in domain for keyword in VENDOR_KEYWORDS):
                return True
    return False

def normalize_phone(phone: str) -> str:
    """Normalize a phone number by removing non-digit characters and leading '1' if present."""
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]
    return digits

def categorize_columns(headers: List[str]) -> Dict[str, List[str]]:
    categories = {
        "first_name": [], "last_name": [], "email": [],
        "phone": [], "title": [], "company": [], "address": []
    }
    for header in headers:
        h_lower = header.lower()
        if ("first" in h_lower and "name" in h_lower) or "fname" in h_lower or "f_name" in h_lower:
            categories["first_name"].append(header)
        elif ("last" in h_lower and "name" in h_lower) or "lname" in h_lower or "l_name" in h_lower:
            categories["last_name"].append(header)
        elif "title" in h_lower:
            categories["title"].append(header)
        elif "company" in h_lower:
            categories["company"].append(header)
        elif "email" in h_lower:
            categories["email"].append(header)
        elif "phone" in h_lower:  # New condition to capture phone columns.
            categories["phone"].append(header)
        elif any(x in h_lower for x in ["address", "street", "city", "zip", "state", "country"]):
            categories["address"].append(header)
    return categories


def load_compass_csv(compass_file: str) -> List[Dict[str, str]]:
    with open(compass_file, mode="r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_phone_csv(phone_file: str) -> List[Dict[str, str]]:
    with open(phone_file, mode="r", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def extract_field(row: Dict[str, str], columns: List[str]) -> str:
    for col in columns:
        val = row.get(col, "").strip()
        if val:
            return val
    return ""


def build_name_key(row: Dict[str, str], cat_map: Dict[str, List[str]]) -> str:
    first_name = extract_field(row, cat_map["first_name"]).lower()
    last_name = extract_field(row, cat_map["last_name"]).lower()
    return f"{first_name} {last_name}".strip()


def fuzzy_name_match(name_key: str, all_keys: List[str], threshold: int = 97) -> Tuple[str, float]:
    if not name_key:
        return (None, 0)
    best = process.extractOne(name_key, all_keys, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return (best[0], best[1])
    return (None, 0)


def extract_phone_address_component(header: str) -> Tuple[str, str]:
    """
    Extracts the address component type and group identifier from a phone export header.
    For example, 'Street 1 (Home Address)' returns ('street', 'home address').
    If no group is found, 'default' is returned.
    """
    # Look for text inside parentheses to serve as the group identifier.
    m = re.search(r'\(([^)]+)\)', header)
    group = m.group(1).strip().lower() if m else "default"
    header_lower = header.lower()
    component = None
    if "street" in header_lower or ("line" in header_lower and "1" in header_lower):
        component = "street"
    elif "line 2" in header_lower:
        component = "street2"
    elif "city" in header_lower:
        component = "city"
    elif "state" in header_lower:
        component = "state"
    elif "zip" in header_lower or "postal" in header_lower:
        component = "zip"
    elif "country" in header_lower:
        component = "country"
    return component, group


def merge_address_into_compass(
    compass_row: Dict[str, str],
    phone_row: Dict[str, str],
    compass_cat_map: Dict[str, List[str]],
    phone_cat_map: Dict[str, List[str]]
) -> Tuple[Dict[str, str], List[str]]:
    changes = []
    
    # 1. Build phone address groups.
    phone_addr_groups = {}  # group identifier -> {component: value}
    for col in phone_cat_map["address"]:
        val = phone_row.get(col, "").strip()
        if not val:
            continue
        component, group = extract_phone_address_component(col)
        if not component:
            continue
        if group not in phone_addr_groups:
            phone_addr_groups[group] = {}
        phone_addr_groups[group][component] = val
    
    if not phone_addr_groups:
        return compass_row, changes
    
    # 2. Group Compass address columns by group identifier.
    #    For example, "Address Line 1" becomes group "1" (default), "Address 2 Line 1" becomes group "2", etc.
    compass_groups = {}
    for col in compass_cat_map["address"]:
        col_lower = col.lower()
        m = re.search(r'address\s*(\d*)', col_lower)
        group_id = m.group(1) if m and m.group(1) else "1"
        component = None
        if "line 1" in col_lower or ("street" in col_lower and "line" not in col_lower):
            component = "street"
        elif "line 2" in col_lower:
            component = "street2"
        elif "city" in col_lower:
            component = "city"
        elif "state" in col_lower:
            component = "state"
        elif "zip" in col_lower or "postal" in col_lower:
            component = "zip"
        elif "country" in col_lower:
            component = "country"
        if group_id not in compass_groups:
            compass_groups[group_id] = {}
        if component:
            compass_groups[group_id][component] = col

    # 3. For each phone address group, check if its address is already present.
    #    We compare the "street" component (case-insensitive).
    for phone_group, phone_addr in phone_addr_groups.items():
        already_present = False
        for group_id, comp_dict in compass_groups.items():
            comp_street = compass_row.get(comp_dict.get("street", ""), "").strip().lower()
            if "street" in phone_addr and comp_street and phone_addr["street"].lower() == comp_street:
                already_present = True
                break
        if already_present:
            continue  # Skip if this phone address is already merged.
        
        # 4. Find the first completely empty Compass group for a new address.
        target_group = None
        for group_id, comp_dict in compass_groups.items():
            group_empty = all(not compass_row.get(col, "").strip() for col in comp_dict.values())
            if group_empty:
                target_group = group_id
                break
        if target_group is None:
            # No empty group available; optionally, you could create new columns here.
            continue
        
        # 5. Fill the target Compass group with the phone address components.
        for comp_type, col in compass_groups[target_group].items():
            if comp_type in phone_addr and not compass_row.get(col, "").strip():
                compass_row[col] = phone_addr[comp_type]
                changes.append(f"Address Group {target_group} -> {col}: {phone_addr[comp_type]}")
    
    return compass_row, changes



def merge_phone_into_compass(
    compass_row: Dict[str, str],
    phone_row: Dict[str, str],
    compass_cat_map: Dict[str, List[str]],
    phone_cat_map: Dict[str, List[str]]
) -> Tuple[Dict[str, str], List[str]]:
    changes = []
    
    # Merge missing email addresses (unchanged)
    phone_emails = [phone_row.get(col, "").strip().lower() for col in phone_cat_map["email"] if phone_row.get(col, "").strip()]
    existing_emails = {compass_row.get(col, "").strip().lower() for col in compass_cat_map["email"] if compass_row.get(col, "").strip()}
    for email in phone_emails:
        if email not in existing_emails:
            for col in compass_cat_map["email"]:
                if not compass_row.get(col, "").strip():
                    compass_row[col] = email
                    changes.append(f"Email->{col}: {email}")
                    existing_emails.add(email)
                    break

    # Merge missing phone numbers using normalization
    phone_numbers = [phone_row.get(col, "").strip() for col in phone_cat_map["phone"] if phone_row.get(col, "").strip()]
    # Normalize phone numbers from phone export
    normalized_phone_numbers = [normalize_phone(p) for p in phone_numbers if p]
    # Normalize existing phone numbers in Compass
    existing_phones = {normalize_phone(compass_row.get(col, "").strip()) for col in compass_cat_map["phone"] if compass_row.get(col, "").strip()}
    for original_phone, norm_phone in zip(phone_numbers, normalized_phone_numbers):
        if norm_phone not in existing_phones:
            for col in compass_cat_map["phone"]:
                if not compass_row.get(col, "").strip():
                    compass_row[col] = original_phone  # Retain original formatting if desired.
                    changes.append(f"Phone->{col}: {original_phone}")
                    existing_phones.add(norm_phone)
                    break

    # Update category based on email domains
    if is_real_estate_agent(phone_emails + list(existing_emails)):
        compass_row["Category"] = "Agent"
        changes.append("Category=Agent")

    return compass_row, changes


def classify_agents(final_data: List[Dict[str, str]], email_columns: List[str]):
    for row in final_data:
        all_emails = set()
        for col in email_columns:
            email = row.get(col, "").strip().lower()
            if email:
                all_emails.add(email)
        row["Category"] = "Agent" if is_real_estate_agent(list(all_emails)) else "Non-Agent"


def classify_vendors(final_data: List[Dict[str, str]], email_columns: List[str]):
    for row in final_data:
        all_emails = set()
        for col in email_columns:
            email = row.get(col, "").strip().lower()
            if email:
                all_emails.add(email)
        if is_vendor(list(all_emails)):
            row["Vendor Classification"] = "Vendor"
            row["Changes Made"] = row.get("Changes Made", "") + " | Classified as Vendor"


def integrate_phone_into_compass(compass_file: str, phone_file: str, output_file: str, logger=None):
    compass_data = load_compass_csv(compass_file)
    phone_data = load_phone_csv(phone_file)
    if not compass_data:
        if logger: 
            logger("Compass CSV has no data.")
        return
    if not phone_data:
        if logger: 
            logger("Phone CSV has no data.")
        return

    compass_cat_map = categorize_columns(list(compass_data[0].keys()))
    phone_cat_map = categorize_columns(list(phone_data[0].keys()))

    compass_keys = [build_name_key(row, compass_cat_map) for row in compass_data]
    updated_compass = list(compass_data)

    # Iterate through phone contacts and only update matching Compass rows.
    for pcontact in phone_data:
        phone_name_key = build_name_key(pcontact, phone_cat_map)
        if not phone_name_key.strip():
            if logger:
                logger("Skipping phone contact with no name.")
            continue

        best_key, score = fuzzy_name_match(phone_name_key, compass_keys, threshold=97)
        if best_key:
            match_idx = compass_keys.index(best_key)
            matched_row = updated_compass[match_idx]
            merged, changes = merge_phone_into_compass(matched_row, pcontact, compass_cat_map, phone_cat_map)
            merged, addr_changes = merge_address_into_compass(merged, pcontact, compass_cat_map, phone_cat_map)
            all_changes = changes + addr_changes
            merged["Changes Made"] = f"Matched {score}% | {'; '.join(all_changes) if all_changes else 'No fields updated.'}"
            updated_compass[match_idx] = merged
        else:
            if logger:
                logger(f"Phone contact '{phone_name_key}' not found in Compass export; skipping.")

    # Only use updated Compass data, without appending new rows.
    final_data = updated_compass

    # Ensure default values for certain fields.
    for row in final_data:
        row.setdefault("Changes Made", "No changes made.")
        row.setdefault("Category", "")

    email_columns = compass_cat_map["email"]
    classify_agents(final_data, email_columns)

    fieldnames = list(compass_data[0].keys())
    extra_columns = ["Category", "Changes Made"]
    for col in extra_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    try:
        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_data)
        if logger:
            logger(f"Merged data written to {output_file}.")
    except Exception as e:
        if logger:
            logger(f"Error writing merged CSV: {e}")

def load_extracted_addresses(address_file: str) -> List[str]:
    """Load only the 'Street Address' from the extracted addresses CSV."""
    extracted_street_addresses = []
    with open(address_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            street = row.get("Street Address", "").strip()
            if street:
                extracted_street_addresses.append(street)
    return extracted_street_addresses


def extract_street_addresses_from_row(row: Dict[str, str]) -> List[str]:
    """Extract values from a row whose header contains 'street' (ignoring case)."""
    street_addresses = []
    for key, value in row.items():
        if value and "street" in key.lower():
            street_addresses.append(value.strip())
    return street_addresses


def fuzzy_address_match(address: str, extracted_addresses: List[str], threshold: int = 90) -> bool:
    """Use fuzzy matching to check if a given street address matches any extracted street addresses."""
    if not address:
        return False
    best_match = process.extractOne(address, extracted_addresses, scorer=fuzz.WRatio)
    return best_match and best_match[1] >= threshold


def extract_address_from_row(row: Dict[str, str], address_columns: List[str]) -> str:
    parts = [row.get(col, "").strip() for col in address_columns if row.get(col)]
    return " ".join(parts)


def classify_clients(final_data: List[Dict[str, str]], address_columns: List[str], extracted_addresses: List[str]):
    """Classify as 'Client' if row's address is fuzzily matched to addresses in the extracted list."""
    for row in final_data:
        address = extract_address_from_row(row, address_columns)
        if fuzzy_address_match(address, extracted_addresses, threshold=90):
            row["Client Classification"] = "Client"
            row["Changes Made"] = row.get("Changes Made", "") + " | Classified as Client"
def update_groups_with_classification(final_data: List[Dict[str, str]]) -> None:
    """
    Updates the 'Groups' field for each contact based on their classification.
    If the contact is classified as an Agent (Category == 'Agent') and the Groups field
    does not include 'Agents', it appends it. The same applies for Vendor.
    Any changes are appended to the 'Changes Made' field.
    """
    for row in final_data:
        groups = row.get("Groups", "").strip()
        changes_for_groups = []
        
        # Check Agent classification from Category field.
        if row.get("Category", "").strip().lower() == "agent":
            if "agents" not in groups.lower():
                if groups:
                    groups += ",Agents"
                else:
                    groups = "Agents"
                changes_for_groups.append("Added Agents to Groups")
        
        # Check Vendor classification from Vendor Classification field.
        if row.get("Vendor Classification", "").strip().lower() == "vendor":
            if "vendors" not in groups.lower():
                if groups:
                    groups += ",Vendors"
                else:
                    groups = "Vendors"
                changes_for_groups.append("Added Vendors to Groups")
        
        row["Groups"] = groups
        if changes_for_groups:
            current_changes = row.get("Changes Made", "").strip()
            if current_changes:
                row["Changes Made"] = current_changes + " | " + "; ".join(changes_for_groups)
            else:
                row["Changes Made"] = "; ".join(changes_for_groups)

def process_files(compass_file: str, phone_file: str, mls_files: List[str], output_dir: str, logger=None):
    """
    Orchestrates the entire flow:
      1) Extract addresses from provided MLS/sales files
      2) Merge phone data into the Compass CSV
      3) Classify agent, client, vendor, etc.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extracted_addresses_file = os.path.join(output_dir, "extracted_addresses.csv")
    merged_file = os.path.join(output_dir, "compass_merged.csv")

    if logger: logger("Starting address extraction...")
    extract_and_save_addresses(mls_files, extracted_addresses_file, logger=logger)
    if logger: logger("Address extraction completed.")

    if logger: logger("Integrating phone data into compass export...")
    integrate_phone_into_compass(compass_file, phone_file, merged_file, logger=logger)
    if logger: logger("Phone integration completed.")

    # Now load the final merged file and classify further
    extracted_addresses = load_extracted_addresses(extracted_addresses_file)
    final_data = load_compass_csv(merged_file)
    if not final_data:
        if logger: logger("No merged data found!")
        return

    compass_cat_map = categorize_columns(list(final_data[0].keys()))
    address_columns = compass_cat_map["address"]
    email_columns = compass_cat_map["email"]

    # Ensure these columns exist for classification
    for row in final_data:
        row.setdefault("Agent Classification", "")
        row.setdefault("Client Classification", "")
        row.setdefault("Vendor Classification", "")

    classify_agents(final_data, email_columns)
    classify_clients(final_data, address_columns, extracted_addresses)
    classify_vendors(final_data, email_columns)
    
    # Update the Groups field based on classification.
    update_groups_with_classification(final_data)

    # Add new columns if missing
    fieldnames = list(final_data[0].keys())
    extra_columns = ["Agent Classification", "Client Classification", "Vendor Classification", "Changes Made"]
    for col in extra_columns:
        if col not in fieldnames:
            fieldnames.append(col)

    try:
        with open(merged_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_data)
        if logger:
            logger(f"Final merged data with classifications written to {merged_file}")
    except Exception as e:
        if logger:
            logger(f"Error saving final merged CSV: {e}")


