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
# Ensure streamlit is imported if st.secrets is used, or handle its absence
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except (ImportError, AttributeError, KeyError):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_DEFAULT_KEY_IF_ANY")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------
# Configuration & API Key
# ---------------------------
# Handled above for flexibility

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
                        "Zip Code": {"type": "string"},
                        "Home Anniversary Date": {"type": "string"}
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
                "Your task is to extract all real estate addresses and their corresponding close dates from unstructured text. "
                "For each property, extract the property address and if available, the close date (which you will return as 'Home Anniversary Date'). "
                "Ensure that the output strictly follows the provided JSON schema."
            )
        },
        {
            "role": "user",
            "content": (
                "Extract all property addresses and their close dates from the following text. "
                "Return ONLY valid addresses in JSON format following the schema provided.\n\n"
                "Text:\n\n" + text
            )
        }
    ]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", # Ensure this model is available and appropriate
            messages=messages,
            response_format={"type": "json_object", "schema": address_schema["schema"]}, # Corrected: pass schema directly
            temperature=0
        )
        structured_response = response.choices[0].message.content
        parsed_data = json.loads(structured_response)
        return parsed_data.get("addresses", [])
    except Exception as e:
        # It's good to log or print the text that caused the error for debugging
        # print(f"Error parsing AI response for text: '{text[:200]}...' Error: {e}")
        # Consider logging to a logger if available
        print(f"Error parsing AI response: {e}")
        return []


def extract_addresses_with_ai_chunked(text: str, max_lines: int = 50, logger=None) -> List[dict]:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        if logger:
            logger("[DEBUG] Sending full text to OpenAI for address extraction")
        return extract_addresses_with_ai(text)
    else:
        if logger:
            logger(f"[DEBUG] Splitting text into {len(lines) // max_lines + 1} chunks for address extraction")
        addresses = []
        for i in range(0, len(lines), max_lines):
            chunk = "\n".join(lines[i:i+max_lines])
            if logger:
                logger(f"[DEBUG] Sending chunk {i//max_lines + 1} to OpenAI for address extraction")
            result = extract_addresses_with_ai(chunk)
            addresses.extend(result)
        return addresses

def extract_addresses_row_by_row(df: pd.DataFrame, logger=None) -> List[dict]:
    extracted_addresses = []
    for idx, row in df.iterrows():
        row_text_parts = [] # Changed variable name for clarity
        for col in df.columns:
            val = str(row[col]).strip()
            if val and val.lower() != "nan":
                row_text_parts.append(f"{col}: {val}") # Changed variable name
        if row_text_parts:
            prompt_text = "\n".join(row_text_parts)
            if logger:
                logger(f"[DEBUG] Extracting addresses from row {idx + 1}...")
            try:
                # Each row is relatively small, so direct call is fine
                result = extract_addresses_with_ai(prompt_text)
                extracted_addresses.extend(result)
            except Exception as e:
                if logger:
                    logger(f"[ERROR] Row {idx + 1} address extraction failed: {e}")
    return extracted_addresses


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}") # Consider using logger
    return " ".join(text.split()) # Normalize whitespace


def extract_text_from_csv(file_path: str) -> str:
    # This function constructs a single string from CSV, which might not be ideal for row-by-row AI.
    # The current implementation in extract_and_save_addresses for CSVs uses extract_addresses_row_by_row.
    # This function (extract_text_from_csv) seems unused in the main flow for address extraction.
    # If it were to be used with extract_addresses_with_ai_chunked, it needs to be effective.
    text = ""
    try:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            row_text_parts = []
            for col in df.columns:
                val = str(row[col]).strip()
                if val and val.lower() != 'nan':
                    row_text_parts.append(f"{col}: {val}")
            text += "\n".join(row_text_parts) + "\n---\n" # Use a clear separator
    except Exception as e:
        print(f"Error reading CSV {file_path} for text extraction: {e}") # Consider logger
    return text.strip()


def extract_addresses_from_excel(file_path: str, logger=None) -> List[dict]:
    extracted_addresses = []
    try:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            if logger:
                logger(f"[DEBUG] Processing sheet: {sheet_name} with {len(df)} rows for address extraction")
            # Using extract_addresses_row_by_row for Excel sheets
            sheet_addresses = extract_addresses_row_by_row(df, logger=logger)
            extracted_addresses.extend(sheet_addresses)
    except Exception as e:
        if logger:
            logger(f"Error processing Excel {os.path.basename(file_path)} for address extraction: {e}")
    return extracted_addresses


def extract_and_save_addresses(file_paths: List[str], output_file: str, logger=None):
    all_extracted_addresses = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[-1].lower()
        file_name = os.path.basename(file_path)
        addresses = [] # Initialize addresses for this file
        if ext == ".pdf":
            if logger:
                logger(f"Processing PDF for addresses: {file_name}")
            text = extract_text_from_pdf(file_path)
            if text: # Only call AI if text was extracted
                addresses = extract_addresses_with_ai_chunked(text, max_lines=50, logger=logger)
            else:
                if logger:
                    logger(f"No text extracted from PDF: {file_name}")
        elif ext == ".csv":
            if logger:
                logger(f"Processing CSV for addresses: {file_name}")
            try:
                # Read CSV into DataFrame for row-by-row processing
                df = pd.read_csv(file_path, on_bad_lines='skip') # Handle potential bad lines
                if not df.empty:
                    addresses = extract_addresses_row_by_row(df, logger=logger)
                else:
                    if logger:
                        logger(f"CSV file is empty or unreadable: {file_name}")
            except Exception as e:
                if logger:
                    logger(f"[ERROR] Failed to process CSV {file_name}: {e}")
        elif ext in [".xls", ".xlsx"]:
            if logger:
                logger(f"Processing Excel for addresses: {file_name}")
            addresses = extract_addresses_from_excel(file_path, logger=logger)
        else:
            if logger:
                logger(f"Unsupported file type for address extraction: {file_name}")
            continue # Skip to next file
        
        all_extracted_addresses.extend(addresses)
    
    # Define fieldnames for the CSV. These should match the keys in the dictionaries from AI.
    fieldnames = ["Street Address", "Unit", "City", "State", "Zip Code", "Home Anniversary Date"]

    try:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Ensure all address dicts have all keys to prevent errors with writerow
            for address in all_extracted_addresses:
                row_to_write = {key: address.get(key, "") for key in fieldnames}
                writer.writerow(row_to_write)
        if logger:
            logger(f"Extracted addresses saved to {output_file}")
    except Exception as e:
        if logger:
            logger(f"Error saving extracted addresses to {output_file}: {e}")

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
    "remax-100-ny.com", "cushwake.com", "moiniangroup.com",
    "exprealty.com", "bhhscalifornia.com", "rodeore.com", "c21allstars.com",
    "century21realtymasters.com", "c21peak.com", "century21king.com", "eplahomes.com",
    "remaxone.com", "remax-olson.com", "remaxchampions.com", "rogunited.com",
    "realtyonegroup.com", "nourmand.com", "johnhartrealestate.com", "dilbeck.com",
    "pinnacleestate.com", "firstteam.com", "weahomes.com", "hiltonhyland.com",
    "ogroup.com", "plgestates.com", "acme-re.com", "dppre.com",
    "beverlyandcompany.com", "excellencere.com", "c21abetterservice.com",
    "c21plaza.com", "realtymasters.com", "podley.com", "lotusproperties.com",
    "greenstreetla.com", "eliterealestatela.com", "c21beachside.com", "erayess.com",
    "c21union.com", "c21everest.com", "c21masters.com", "c21desertrock.com",
    "citrusrealty.com", "loislauer.com", "rnewyork.com", "opgny.com",
    "exitrealtygenesis.com", "exitrealtyminimax.com", "rutenbergny.com", "spireny.com",
    "winzonerealty.com", "erealtyinternational.com", "exitrealtysi.com",
    "century21mk.com", "hlresidential.com", "ccrny.com",
    "kianrealtynyc.com", "weichert.com", "bhhsnyproperties.com",
    "c21metropolitan.com", "century21prorealty.com", "rapidnyc.com", "exitrealtypro.com",
    "c21metrostar.com", "exitfirstchoice.com", "chasegr.com", "cityexpressrealty.com",
    "barealtygroup.com", "papprealty.com", "hgrealty.com", "c21futurehomes.com",
    "eratopservice.com", "exitrealty.com", "coldwellbankerhomes.com", "bairdwarner.com",
    "bhhschicago.com", "remaxsuburban.com", "remax10.com", "illinoisproperty.com",
    "dreamtown.com", "jamesonsir.com", "exitrealtyredefined.com", "c21affiliated.com",
    "c21circle.com", "kalerealty.com", "crrdraper.com", "redfin.com",
    "coldwellhomes.com", "propertiesmidwest.com", "chasebroker.com",
    "daprileproperties.com", "century21universal.com", "starhomes.com",
    "unitedrealestateaustin.com", "fathomrealty.com", "urbanrealestate.com",
    "starckre.com", "c21langos.com", "c211stclasshomes.com", "remax-ultimatepros.com",
    "lolliproperties.com", "realpeoplerealty.com", "erasunriserealty.com", "remax1st.com",
    "bhgre.com", "ggsir.com", "vanguardproperties.com", "zephyrre.com", "intero.com",
    "c21rea.com", "c21realtyalliance.com", "serenogroup.com", "legacyrea.com",
    "apr.com", "mcguire.com", "sequoia-re.com", "kinokorealestate.com",
    "cityrealestatesf.com", "ascendrealestategroup.com", "amirealestate.com",
    "bhgthrive.com", "cornerstonerealty.com", "parknorth.com", "polaris-realty.com",
    "jacksonfuller.com", "mosaikrealestate.com", "barbco.com", "kwpeninsulaestates.com",
    "c21mm.com", "c21baldini.com", "remaxaccord.com", "realtyaustin.com",
    "jbgoodwin.com", "allcityhomes.com", "moreland.com", "kuperrealty.com",
    "remax1.org", "austin.evrealestate.com", "rogprosper.com", "texasrealtypartners.com",
    "denpg.com", "austinportfolio.com", "magnoliarealty.com", "residentrealty.com",
    "bhhstxrealty.com", "skyrealtyaustin.com", "twelveriversrealty.com",
    "streamlineproperties.com", "realtytexas.com", "highlandlakesrealtors.com",
    "avalaraustin.com", "stanberry.com", "urbanspacerealtors.com", "austinrealestate.com",
    "bhghomecity.com", "purerealty.com", "austinoptions.com", "highlandsrealty.com",
    "dochenrealtors.com", "c21judgefite.com", "ebby.com", "rmdfw.com", "jpar.com",
    "winansbhg.com", "briggsfreeman.com", "alliebeth.com", "remaxtrinity.com",
    "rogershealy.com", "unitedrealestatedallas.com", "colleenfrost.com", "brikorealty.com",
    "kwurbantx.com", "halogrouprealty.com", "competitiveedgerealty.com",
    "century21allianceproperties.com", "ultimare.com", "txconnectrealty.com",
    "williamdavisrealty.com", "dallas.evrealestate.com", "bhhspenfed.com",
    "robertelliott.com", "c21bowman.com", "cbapex.com", "keyes.com", "ewm.com",
    "bhhsfloridarealty.com", "advancerealtyfl.com", "beachfrontonline.com", "urgfl.com",
    "lokationre.com", "partnershiprealty.com", "lptrealty.com",
    "c21worldconnection.com", "luxeproperties.com", "avantiway.com", "cervera.com",
    "bestbeach.net", "globalluxuryrealty.com", "realestateempire.com", "ipre.com",
    "miami.evrealestate.com", "fir.com", "relatedisg.com", "bhsmia.com",
    "opulenceintlrealty.com", "rutenbergftl.com", "c21yarlex.com",
    "miamirealestategroup.com", "floridacapitalrealty.com", "eliteoceanviewrealty.com",
    "legacyplusrealty.com", "onepathrealty.com", "xcellencerealty.com",
    "skyelouisrealty.com", "c21tenace.com", "cltrealestate.com", "dickensmitchener.com",
    "helenadamsrealty.com", "bhhscarolinas.com", "paraclarealty.com", "c21murphy.com",
    "markspain.com", "costellorei.com", "lakenormanrealty.com", "ivesterjackson.com",
    "savvyandcompany.com", "c21vanguard.com", "wilkinsonera.com", "eralivemoore.com",
    "southernhomesnc.com", "givingtreerealty.com", "rogrevolution.com", "prostead.com",
    "charlotte.evrealestate.com", "sycamoreproperties.net", "carolinarealtyadvisors.com",
    "wilsonrealtync.com", "northgroupre.com", "stephencooley.com", "bhhsgeorgia.com",
    "metrobrokers.com", "palmerhouseproperties.com", "c21results.com",
    "atlantacommunities.net", "maximumone.com", "chapmanhallrealtors.com",
    "crye-leike.com", "atlanta.evrealestate.com", "solidsourcehomes.com", "ansleyre.com",
    "beacham.com", "atlantafinehomes.com", "c21intown.com", "bhgjbarry.com",
    "rogedge.com", "therealtygroupga.com", "remax-around-atlanta-ga.com",
    "c21connectrealty.com", "c21novus.com"
}

VENDOR_KEYWORDS = ["title", "mortgage", "lending", "escrow"] 

def is_real_estate_agent(emails: List[str]) -> bool:
    if not emails: return False
    for email in emails:
        if not email or not isinstance(email, str): continue
        match = re.search(r"@([\w.-]+)$", email)
        if match:
            domain = match.group(1).lower()
            if domain in BROKERAGE_EMAIL_DOMAINS:
                return True
    return False


def is_vendor(emails: List[str]) -> bool:
    if not emails: return False
    for email in emails:
        if not email or not isinstance(email, str): continue
        match = re.search(r"@([\w.-]+)$", email)
        if match:
            domain = match.group(1).lower()
            # Check if any part of the domain is a vendor keyword (e.g., "mortgagecompany.com")
            # or if the keyword is part of the subdomain (e.g., "title.services.com")
            if any(keyword in domain for keyword in VENDOR_KEYWORDS):
                return True
    return False

def normalize_phone(phone: str) -> str:
    if not phone or not isinstance(phone, str): return ""
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 11 and digits.startswith('1'):
        digits = digits[1:]
    return digits

def categorize_columns(headers: List[str]) -> Dict[str, List[str]]:
    categories = {
        "first_name": [], "last_name": [], "email": [],
        "phone": [], "title": [], "company": [], "address": []
    }
    if not headers: return categories
    for header in headers:
        if not header or not isinstance(header, str): continue
        h_lower = header.lower()
        if ("first" in h_lower and "name" in h_lower) or "fname" in h_lower or "f_name" in h_lower:
            categories["first_name"].append(header)
        elif ("last" in h_lower and "name" in h_lower) or "lname" in h_lower or "l_name" in h_lower:
            categories["last_name"].append(header)
        elif "title" in h_lower: # Be careful this doesn't overlap too much with "Job Title" etc if not desired
            categories["title"].append(header)
        elif "company" in h_lower:
            categories["company"].append(header)
        elif "email" in h_lower: # Catches "Email", "Email Address", "Primary Email"
            categories["email"].append(header)
        elif "phone" in h_lower: # Catches "Phone", "Mobile Phone", "Work Phone"
            categories["phone"].append(header)
        elif any(x in h_lower for x in ["address", "street", "city", "zip", "state", "country", "line 1", "line 2"]):
            categories["address"].append(header)
    return categories


def load_compass_csv(compass_file: str) -> List[Dict[str, str]]:
    data = []
    try:
        with open(compass_file, mode="r", encoding="utf-8-sig") as f: # utf-8-sig handles BOM
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: Compass file not found at {compass_file}") # Use logger if available
    except Exception as e:
        print(f"Error reading Compass CSV {compass_file}: {e}") # Use logger
    return data


def load_phone_csv(phone_file: str) -> List[Dict[str, str]]:
    data = []
    try:
        with open(phone_file, mode="r", encoding="utf-8-sig", errors="replace") as f: # errors='replace' for bad chars
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: Phone file not found at {phone_file}") # Use logger
    except Exception as e:
        print(f"Error reading Phone CSV {phone_file}: {e}") # Use logger
    return data


def extract_field(row: Dict[str, str], columns: List[str]) -> str:
    if not row or not columns: return ""
    for col in columns:
        val = row.get(col, "").strip()
        if val:
            return val
    return ""


def build_name_key(row: Dict[str, str], cat_map: Dict[str, List[str]]) -> str:
    if not row or not cat_map: return ""
    first_name = extract_field(row, cat_map.get("first_name", [])).lower()
    last_name = extract_field(row, cat_map.get("last_name", [])).lower()
    return f"{first_name} {last_name}".strip()


def fuzzy_name_match(name_key: str, all_keys: List[str], threshold: int = 97) -> Tuple[str, float]:
    if not name_key or not all_keys: # Ensure all_keys is not empty
        return (None, 0)
    # process.extractOne can return None if choices are empty or no match above score_cutoff
    # We handle score threshold manually.
    best_match_tuple = process.extractOne(name_key, all_keys, scorer=fuzz.WRatio)
    if best_match_tuple and best_match_tuple[1] >= threshold:
        return (best_match_tuple[0], best_match_tuple[1])
    return (None, 0)


def extract_phone_address_component(header: str) -> Tuple[str, str]:
    if not header: return None, "default"
    m = re.search(r'\(([^)]+)\)', header) # Extracts content within parentheses
    group = m.group(1).strip().lower() if m else "default"
    
    header_lower = header.lower()
    component = None
    if "street" in header_lower or ("line" in header_lower and "1" in header_lower): # Prioritize "line 1" for street
        component = "street"
    elif "line 2" in header_lower:
        component = "street2"
    elif "city" in header_lower:
        component = "city"
    elif "state" in header_lower or "province" in header_lower: # Added province
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
    if not compass_row or not phone_row or not compass_cat_map or not phone_cat_map:
        return compass_row, changes

    phone_addr_groups = {} 
    for col in phone_cat_map.get("address", []):
        val = phone_row.get(col, "").strip()
        if not val:
            continue
        component, group_label = extract_phone_address_component(col) # group_label e.g. "home address"
        if not component:
            continue
        
        group_key = group_label # Use the extracted label as the key
        if group_key not in phone_addr_groups:
            phone_addr_groups[group_key] = {}
        phone_addr_groups[group_key][component] = val
    
    if not phone_addr_groups:
        return compass_row, changes
    
    # Map Compass address columns to groups (e.g., "Address 1", "Address 2")
    compass_addr_column_groups = {} # Key: group_id ('1', '2', ...), Value: {comp_type: col_name}
    for col_name in compass_cat_map.get("address", []):
        col_lower = col_name.lower()
        
        # Determine group (Address 1, Address 2, etc.)
        group_id_match = re.search(r'address\s*([1-6])', col_lower) # Look for "Address 1" through "Address 6"
        group_id = group_id_match.group(1) if group_id_match else "1" # Default to group "1"

        component_type = None
        if "line 1" in col_lower or ("street" in col_lower and "line 2" not in col_lower and "city" not in col_lower and "state" not in col_lower and "zip" not in col_lower and "country" not in col_lower) :
            component_type = "street"
        elif "line 2" in col_lower:
            component_type = "street2"
        elif "city" in col_lower:
            component_type = "city"
        elif "state" in col_lower:
            component_type = "state"
        elif "zip" in col_lower or "postal" in col_lower:
            component_type = "zip"
        elif "country" in col_lower:
            component_type = "country"

        if component_type:
            if group_id not in compass_addr_column_groups:
                compass_addr_column_groups[group_id] = {}
            # Avoid overwriting if a more specific type (like 'Address Line 1') was already mapped for 'street'
            if not (component_type == "street" and "street" in compass_addr_column_groups[group_id] and "line 1" in compass_addr_column_groups[group_id]["street"].lower()):
                 compass_addr_column_groups[group_id][component_type] = col_name


    # Try to merge phone addresses into Compass address slots
    for phone_group_label, phone_addr_components in phone_addr_groups.items():
        phone_street = phone_addr_components.get("street", "").strip().lower()
        if not phone_street: continue # Skip if phone address has no street

        # Check if this phone address (by street) is already in any Compass address group
        already_present_in_compass = False
        for group_id, compass_cols in compass_addr_column_groups.items():
            compass_street_col = compass_cols.get("street")
            if compass_street_col:
                compass_street_val = compass_row.get(compass_street_col, "").strip().lower()
                if phone_street == compass_street_val:
                    already_present_in_compass = True
                    break
        if already_present_in_compass:
            continue

        # Find an empty Compass address group to put this new address
        target_compass_group_id = None
        sorted_group_ids = sorted(compass_addr_column_groups.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))

        for group_id in sorted_group_ids:
            is_group_empty = True
            for component_col in compass_addr_column_groups[group_id].values():
                if compass_row.get(component_col, "").strip():
                    is_group_empty = False
                    break
            if is_group_empty:
                target_compass_group_id = group_id
                break
        
        if target_compass_group_id:
            compass_target_cols = compass_addr_column_groups[target_compass_group_id]
            for ph_comp_type, ph_val in phone_addr_components.items():
                compass_col_for_comp = compass_target_cols.get(ph_comp_type)
                if compass_col_for_comp and not compass_row.get(compass_col_for_comp, "").strip(): # If corresponding Compass field is empty
                    compass_row[compass_col_for_comp] = ph_val
                    changes.append(f"Addr Grp {target_compass_group_id} ({ph_comp_type} from {phone_group_label}) -> {compass_col_for_comp}: {ph_val}")
    
    return compass_row, changes


def merge_phone_into_compass(
    compass_row: Dict[str, str],
    phone_row: Dict[str, str],
    compass_cat_map: Dict[str, List[str]],
    phone_cat_map: Dict[str, List[str]]
) -> Tuple[Dict[str, str], List[str]]:
    changes = []
    if not compass_row or not phone_row or not compass_cat_map or not phone_cat_map:
        return compass_row, changes
    
    # Merge Emails
    phone_emails = {phone_row.get(col, "").strip().lower() for col in phone_cat_map.get("email", []) if phone_row.get(col, "").strip()}
    existing_compass_emails = {compass_row.get(col, "").strip().lower() for col in compass_cat_map.get("email", []) if compass_row.get(col, "").strip()}
    
    new_emails_added = False
    for email_to_add in phone_emails:
        if email_to_add not in existing_compass_emails:
            # Find an empty email slot in Compass row
            for compass_email_col in compass_cat_map.get("email", []):
                if not compass_row.get(compass_email_col, "").strip():
                    compass_row[compass_email_col] = phone_row.get([pcol for pcol in phone_cat_map.get("email",[]) if phone_row.get(pcol,"").strip().lower() == email_to_add][0], email_to_add) # Keep original casing from phone_row
                    changes.append(f"Email -> {compass_email_col}: {compass_row[compass_email_col]}")
                    existing_compass_emails.add(email_to_add) # Add to set to track it's now in compass
                    new_emails_added = True
                    break 
    
    # Merge Phones
    phone_numbers_original_case = {phone_row.get(col, "").strip() for col in phone_cat_map.get("phone", []) if phone_row.get(col, "").strip()}
    normalized_phone_numbers_from_phone_csv = {normalize_phone(p) for p in phone_numbers_original_case if normalize_phone(p)}
    
    existing_compass_phones_normalized = {normalize_phone(compass_row.get(col, "").strip()) for col in compass_cat_map.get("phone", []) if compass_row.get(col, "").strip() and normalize_phone(compass_row.get(col,"").strip())}

    new_phones_added = False
    for original_phone_val in phone_numbers_original_case:
        norm_phone_to_add = normalize_phone(original_phone_val)
        if norm_phone_to_add and norm_phone_to_add not in existing_compass_phones_normalized:
            for compass_phone_col in compass_cat_map.get("phone", []):
                if not compass_row.get(compass_phone_col, "").strip():
                    compass_row[compass_phone_col] = original_phone_val # Keep original casing
                    changes.append(f"Phone -> {compass_phone_col}: {original_phone_val}")
                    existing_compass_phones_normalized.add(norm_phone_to_add)
                    new_phones_added = True
                    break
    
    # Agent classification based on *all* emails now in the compass_row
    # This is done later in classify_agents based on the final state of emails.
    # if new_emails_added or new_phones_added: # Or any other change
    # pass, changes are returned

    return compass_row, changes


def classify_agents(final_data: List[Dict[str, str]], email_columns: List[str]):
    if not final_data or not email_columns: return
    for row in final_data:
        # Collect all emails from the specified email columns for the current row
        all_emails_for_row = set()
        for col in email_columns:
            email = row.get(col, "").strip().lower()
            if email:
                all_emails_for_row.add(email)
        
        # Preserve existing category if it's already "Agent", otherwise classify
        # Also, if it's already classified, don't change it from Agent to Non-Agent unless it's a specific requirement.
        # Current logic: if any email is agent email, mark as Agent. Otherwise, Non-Agent (unless already Agent).
        is_agent_by_email = is_real_estate_agent(list(all_emails_for_row))
        current_category = row.get("Category", "").strip().lower()

        if is_agent_by_email:
            if current_category != "agent":
                row["Category"] = "Agent"
                # Changes for this specific classification are handled by update_groups_and_log_changes
        elif not current_category: # Only set to Non-Agent if category is currently empty
             row["Category"] = "Non-Agent"


def classify_vendors(final_data: List[Dict[str, str]], email_columns: List[str]):
    if not final_data or not email_columns: return
    for row in final_data:
        all_emails_for_row = set()
        for col in email_columns:
            email = row.get(col, "").strip().lower()
            if email:
                all_emails_for_row.add(email)
        
        is_vendor_by_email = is_vendor(list(all_emails_for_row))
        
        if is_vendor_by_email:
            if row.get("Vendor Classification", "").strip().lower() != "vendor":
                row["Vendor Classification"] = "Vendor"
                # Changes for this specific classification are handled by update_groups_and_log_changes
        # No "Non-Vendor" status usually, it's either Vendor or empty.


def integrate_phone_into_compass(compass_file: str, phone_file: str, output_file: str, logger=None):
    compass_data = load_compass_csv(compass_file)
    phone_data = load_phone_csv(phone_file)

    if not compass_data:
        if logger: 
            logger("Compass CSV is empty or could not be loaded. Cannot proceed with phone integration.")
        # Write empty output file with headers if possible, or handle error
        # For now, just return if no base data.
        # Consider if an empty compass_file should result in an empty output_file or an error.
        # If `compass_file` path was valid but empty, `load_compass_csv` returns [].
        # To create an output file with just headers (if that's desired):
        if os.path.exists(compass_file):
            try:
                with open(compass_file, "r", encoding="utf-8-sig") as cf_headers_only, \
                     open(output_file, "w", newline="", encoding="utf-8") as of_empty:
                    original_order_empty = cf_headers_only.readline().strip().split(',')
                    if original_order_empty and original_order_empty[0]: # Check if headers were actually read
                        extra_columns = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification"]
                        final_fieldnames_empty = original_order_empty + [col for col in extra_columns if col not in original_order_empty]
                        writer_empty = csv.DictWriter(of_empty, fieldnames=final_fieldnames_empty)
                        writer_empty.writeheader()
            except Exception as e_header:
                if logger: logger(f"Could not write empty output with headers: {e_header}")
        return

    compass_headers = list(compass_data[0].keys()) if compass_data else []
    # Initialize necessary columns that might not be in the original compass_data
    # These columns will be populated by subsequent classification steps.
    extra_columns_to_ensure = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification", "Groups"]
    for row in compass_data:
        for col in extra_columns_to_ensure:
            row.setdefault(col, "") # Default to empty string
        if not row.get("Changes Made"): # If "Changes Made" is empty after setdefault
            row["Changes Made"] = "No changes made."


    if not phone_data:
        if logger: 
            logger("Phone CSV is empty or could not be loaded. Proceeding without phone data merge.")
        # compass_data (with initialized columns) becomes the final_data for this stage
        final_data_for_write = compass_data
    else:
        compass_cat_map = categorize_columns(compass_headers)
        phone_headers = list(phone_data[0].keys()) if phone_data else []
        phone_cat_map = categorize_columns(phone_headers)

        # Create a dictionary for faster lookups of compass_data by name_key
        # This assumes name_keys are unique enough for a first match.
        # If multiple compass rows match a phone_name_key, this merges into the first one found.
        compass_data_dict_by_name = {build_name_key(row, compass_cat_map): row for row in compass_data}
        # To handle multiple identical names in compass_data, could map name_key to list of indices
        # For simplicity, current code updates the first match if using dict, or iterates if using list.
        
        # Let's stick to iterating and updating `compass_data` (list of dicts) directly or copies
        # to avoid issues with dicts if keys are not perfectly unique or order matters.
        # Build a list of name keys from compass_data for fuzzy matching
        compass_name_keys_list = [build_name_key(row, compass_cat_map) for row in compass_data]

        for p_idx, pcontact in enumerate(phone_data):
            phone_name_key = build_name_key(pcontact, phone_cat_map)
            if not phone_name_key.strip():
                if logger:
                    logger(f"Skipping phone contact at index {p_idx} with no name.")
                continue

            best_match_key, score = fuzzy_name_match(phone_name_key, compass_name_keys_list, threshold=97)
            
            if best_match_key:
                # Find all indices in compass_data that match best_match_key
                # (in case of duplicate names in compass_data that were used for fuzzy matching)
                indices_to_update = [i for i, key in enumerate(compass_name_keys_list) if key == best_match_key]

                for match_idx in indices_to_update: # Iterate if multiple exact name matches in compass
                    compass_row_to_update = compass_data[match_idx] 
                    
                    # Perform merge for data fields (email, phone)
                    merged_row_data, data_changes = merge_phone_into_compass(
                        compass_row_to_update.copy(), # Pass a copy to avoid modifying original during merge logic
                        pcontact, compass_cat_map, phone_cat_map
                    )
                    # Perform merge for addresses
                    merged_row_final, addr_changes = merge_address_into_compass(
                        merged_row_data, # Result from previous merge
                        pcontact, compass_cat_map, phone_cat_map
                    )
                    
                    all_new_changes_for_row = data_changes + addr_changes
                    
                    if all_new_changes_for_row:
                        current_changes_log = compass_row_to_update.get("Changes Made", "")
                        # If current log is "No changes made.", overwrite it. Otherwise, append.
                        change_log_entry = f"Matched {score}% with '{phone_name_key}': {'; '.join(all_new_changes_for_row)}"
                        if current_changes_log.lower() == "no changes made." or not current_changes_log :
                            merged_row_final["Changes Made"] = change_log_entry
                        else:
                            merged_row_final["Changes Made"] = f"{current_changes_log} | {change_log_entry}"
                    else: # No new changes from this phone contact, keep existing "Changes Made"
                        merged_row_final["Changes Made"] = compass_row_to_update.get("Changes Made", "No changes made.")


                    compass_data[match_idx] = merged_row_final # Update the row in the main list
                    if len(indices_to_update) > 1 and logger:
                        logger(f"Updated Compass contact at index {match_idx} for phone contact '{phone_name_key}' (part of multiple matches).")
                    # If only one match, break (or remove break if all exact matches should be updated by same phone contact)
                    break # Assuming we update only the first best match index
            else:
                if logger:
                    logger(f"Phone contact '{phone_name_key}' (Index {p_idx}) not matched in Compass export; skipping merge for this contact.")
        
        final_data_for_write = compass_data

    # Determine final fieldnames for writing the merged file
    # Start with original compass headers, then add any new ones from extra_columns_to_ensure
    final_fieldnames = compass_headers + [col for col in extra_columns_to_ensure if col not in compass_headers]
    # Ensure no duplicates if a column was somehow in both
    final_fieldnames = list(dict.fromkeys(final_fieldnames))


    try:
        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction='ignore')
            writer.writeheader()
            # Ensure all rows have all fieldnames before writing
            for row in final_data_for_write:
                writer.writerow({k: row.get(k, "") for k in final_fieldnames})
        if logger:
            logger(f"Integrated phone data (or initialized file) written to {output_file}.")
    except Exception as e:
        if logger:
            logger(f"Error writing merged CSV after phone integration: {e}")


def load_extracted_addresses(address_file: str) -> List[dict]:
    extracted_records = []
    if not address_file or not os.path.exists(address_file):
        print(f"Info: Extracted addresses file not found or path is null: {address_file}.")
        return extracted_records
    try:
        with open(address_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                extracted_records.append(row)
    except Exception as e: # Catch more general errors like empty file, permissions etc.
        print(f"Error reading extracted addresses file {address_file}: {e}")
    return extracted_records


def classify_clients_simplified(final_data: List[Dict[str, str]], 
                                extracted_addresses: List[dict], 
                                logger=None):
    if logger: logger("[INFO] Starting client classification and HAD update.")

    if not extracted_addresses:
        if logger: logger("[DEBUG] Extracted addresses list (MLS/Sales) is empty. No HAD update possible from it.")
        return

    mls_address_to_had_lookup = {}
    for mls_record in extracted_addresses:
        street = str(mls_record.get("Street Address", "")).strip().lower()
        # Assuming date is already YYYY-MM-DD or similar, if not, needs parsing/validation
        had = str(mls_record.get("Home Anniversary Date", "")).strip() 
        if street and had:
            if street not in mls_address_to_had_lookup:
                 mls_address_to_had_lookup[street] = had
            # else: # Handle duplicates if necessary, e.g. log or use latest date
                # if logger: logger(f"[DEBUG] Duplicate street '{street}' in MLS data. Using first HAD found: {had}")
    
    if not mls_address_to_had_lookup:
        if logger: logger("[DEBUG] MLS address to HAD lookup dictionary is empty after processing. No HAD update possible.")
        return
    
    if logger: logger(f"[DEBUG] Created MLS lookup with {len(mls_address_to_had_lookup)} unique street-to-HAD entries.")

    successful_updates = 0
    mls_street_keys_list = list(mls_address_to_had_lookup.keys()) # For fuzzy matching

    for compass_row_idx, compass_row in enumerate(final_data):
        # Only process if Home Anniversary Date is currently null/empty in the Compass row
        if compass_row.get("Home Anniversary Date", "").strip(): 
            continue

        found_match_for_this_compass_row = False
        
        # Define which Compass address columns to check against MLS data
        # This should be comprehensive: Address Line 1 of primary, and Address 1 Line 1 of Address 2, Address 3 etc.
        potential_compass_street_cols = []
        if "Address Line 1" in compass_row: # Primary address line 1
             potential_compass_street_cols.append("Address Line 1")
        for i in range(1, 7): # Address 1 Line 1, Address 2 Line 1, ... Address 6 Line 1
            col_name = f"Address {i} Line 1"
            if col_name in compass_row: # Check if column actually exists in the data
                potential_compass_street_cols.append(col_name)
        if "Street Address" in compass_row: # Generic street address column, if present
            potential_compass_street_cols.append("Street Address")
        
        potential_compass_street_cols = list(dict.fromkeys(potential_compass_street_cols)) # Deduplicate

        for address_col_name in potential_compass_street_cols:
            compass_street_line = str(compass_row.get(address_col_name, "")).strip().lower()
            if not compass_street_line:
                continue

            # Attempt Direct Lookup
            if compass_street_line in mls_address_to_had_lookup:
                had_to_set = mls_address_to_had_lookup[compass_street_line]
                compass_row["Home Anniversary Date"] = had_to_set
                compass_row["Client Classification"] = "Past client" 
                
                current_changes = compass_row.get("Changes Made", "")
                change_msg = f"HAD set from direct match on '{address_col_name}'"
                if current_changes.lower() == "no changes made." or not current_changes:
                    compass_row["Changes Made"] = change_msg
                else:
                    compass_row["Changes Made"] = f"{current_changes} | {change_msg}"
                
                successful_updates += 1
                # if logger: logger(f"[DEBUG] Direct match for HAD update on Compass row {compass_row_idx} via '{address_col_name}'. Set HAD: {had_to_set}")
                found_match_for_this_compass_row = True
                break 

            # Attempt Fuzzy Lookup if direct failed for this compass_street_line
            if not found_match_for_this_compass_row and mls_street_keys_list:
                best_match_tuple = process.extractOne(compass_street_line, mls_street_keys_list, scorer=fuzz.WRatio, score_cutoff=93)

                if best_match_tuple: # extractOne with score_cutoff returns None if below cutoff
                    matched_mls_street_key = best_match_tuple[0]
                    score = best_match_tuple[1]
                    had_to_set = mls_address_to_had_lookup[matched_mls_street_key]
                    
                    compass_row["Home Anniversary Date"] = had_to_set
                    compass_row["Client Classification"] = "Past client" # Changed from "Client" to "Past client" for consistency

                    current_changes = compass_row.get("Changes Made", "")
                    change_msg = f"HAD set from fuzzy match (Score: {score:.0f}%) on '{address_col_name}' (C: '{compass_street_line}' vs MLS: '{matched_mls_street_key}')"
                    if current_changes.lower() == "no changes made." or not current_changes:
                        compass_row["Changes Made"] = change_msg
                    else:
                        compass_row["Changes Made"] = f"{current_changes} | {change_msg}"
                    
                    successful_updates += 1
                    # if logger: logger(f"[DEBUG] Fuzzy match for HAD update on Compass row {compass_row_idx} via '{address_col_name}'. Set HAD: {had_to_set}")
                    found_match_for_this_compass_row = True
                    break 
        
    if logger: 
        logger(f"[INFO] Client classification and HAD update finished. {successful_updates} records had HAD populated/updated.")


def update_groups_with_classification(final_data: List[Dict[str, str]], logger=None) -> None:
    if not final_data: return
    if logger: logger("[INFO] Updating 'Groups' field based on classifications.")
    updated_count = 0

    for row_idx, row in enumerate(final_data):
        groups_str = row.get("Groups", "").strip()
        # Split by comma, strip whitespace for each item, convert to lowercase for set operations
        current_groups_set = {g.strip().lower() for g in groups_str.split(",") if g.strip()}
        original_groups_set_for_comparison = set(current_groups_set) # For detecting actual change

        # Temporary list to store textual descriptions of group changes for this row
        group_changes_log_parts = []

        # Check Agent classification
        if row.get("Category", "").strip().lower() == "agent":
            if "agents" not in current_groups_set:
                current_groups_set.add("agents")
                group_changes_log_parts.append("Added Agents to Groups")

        # Check Vendor classification
        if row.get("Vendor Classification", "").strip().lower() == "vendor":
            if "vendors" not in current_groups_set:
                current_groups_set.add("vendors")
                group_changes_log_parts.append("Added Vendors to Groups")

        # Check Client classification for 'Past client'
        if row.get("Client Classification", "").strip().lower() == "past client":
            if "past clients" not in current_groups_set: # ensure "past clients" (plural)
                current_groups_set.add("past clients")
                group_changes_log_parts.append("Added Past Clients to Groups")

        # Only update row's "Groups" and "Changes Made" if actual group membership changed
        if current_groups_set != original_groups_set_for_comparison:
            updated_count +=1
            # Reconstruct the Groups string: Title Case, sorted alphabetically
            final_groups_list = sorted([g.title() for g in current_groups_set if g])
            row["Groups"] = ",".join(final_groups_list)

            if group_changes_log_parts: # If there are specific log messages for group changes
                group_change_summary = "; ".join(group_changes_log_parts)
                current_main_changes = row.get("Changes Made", "").strip()
                if current_main_changes.lower() == "no changes made." or not current_main_changes:
                    row["Changes Made"] = group_change_summary
                else:
                    row["Changes Made"] = f"{current_main_changes} | {group_change_summary}"
    if logger: logger(f"[INFO] Groups field updated for {updated_count} records.")


def export_updated_records(merged_file_path: str, import_output_dir: str, logger=None):
    exclude_cols = {
        "Created At", "Last Contacted", "Changes Made", "Category", 
        "Agent Classification", "Client Classification", "Vendor Classification" 
        # "Home Anniversary Date" IS KEPT for import
    }

    rows_to_export = []
    original_fieldnames_from_merged_file = []

    if not os.path.exists(merged_file_path):
        if logger: logger(f"[ERROR export] Merged file {merged_file_path} not found. Cannot export.")
        return
        
    try:
        with open(merged_file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            original_fieldnames_from_merged_file = reader.fieldnames or []
            for row in reader:
                changes_made_val = row.get("Changes Made", "").strip().lower()
                if changes_made_val and changes_made_val != "no changes made.":
                    rows_to_export.append(row)
    except Exception as e:
        if logger:
            logger(f"[ERROR export] Could not read {merged_file_path} for export: {e}")
        return

    if not original_fieldnames_from_merged_file:
        if logger: logger("[WARNING export] Merged file seems to have no headers. Cannot determine export columns.")
        return

    if not rows_to_export:
        if logger:
            logger("[INFO export] No records found with actual changes. Nothing to export for Compass import.")
        return

    # Determine fieldnames for the import file: original order, excluding specified columns
    import_fieldnames = [col for col in original_fieldnames_from_merged_file if col not in exclude_cols]

    if not os.path.exists(import_output_dir):
        try:
            os.makedirs(import_output_dir)
        except OSError as e:
            if logger: logger(f"[ERROR export] Could not create output directory {import_output_dir}: {e}")
            return
    
    # For now, assuming one output file is sufficient as per original logic (compass_import_part1)
    # If chunking is needed later, the logic can be re-introduced. Max 2000 was mentioned.
    # Current code doesn't implement chunking but writes all to part1.
    output_path = os.path.join(import_output_dir, "compass_import_part1.csv")
    
    # Prepare rows for writing: only include designated import_fieldnames
    final_rows_for_csv = []
    for original_row in rows_to_export:
        row_for_csv = {key: original_row.get(key, "") for key in import_fieldnames}
        final_rows_for_csv.append(row_for_csv)

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=import_fieldnames, extrasaction='drop') # drop if any stray keys
            writer.writeheader()
            writer.writerows(final_rows_for_csv)
        if logger:
            logger(f"[INFO export] Exported {len(final_rows_for_csv)} updated record(s) to {output_path}")
    except Exception as e:
        if logger:
            logger(f"[ERROR export] Could not write import file {output_path}: {e}")


def process_files(compass_file: str, phone_file: str, mls_files: List[str], output_dir: str, logger=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extracted_addresses_file = os.path.join(output_dir, "extracted_addresses.csv")
    merged_file = os.path.join(output_dir, "compass_merged.csv")
    import_output_dir = os.path.join(output_dir, "compass_import")

    extracted_addresses = []
    if mls_files:
        if logger: logger("Starting address extraction from MLS files...")
        extract_and_save_addresses(mls_files, extracted_addresses_file, logger=logger)
        extracted_addresses = load_extracted_addresses(extracted_addresses_file) 
        if logger: logger(f"Address extraction from MLS complete. Found {len(extracted_addresses)} addresses.")
    else:
        if logger: logger("No MLS files provided. Skipping MLS address extraction.")
        # Ensure extracted_addresses.csv is empty or non-existent if no MLS files
        if os.path.exists(extracted_addresses_file):
            try: # Create an empty file with headers
                 with open(extracted_addresses_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["Street Address", "Unit", "City", "State", "Zip Code", "Home Anniversary Date"])
                    writer.writeheader()
            except Exception as e_empty_mls:
                if logger: logger(f"Could not create empty extracted_addresses.csv: {e_empty_mls}")


    # Step 1: Initial creation of merged_file (from compass_file, potentially with phone_file merged)
    if phone_file:
        if logger: logger("Integrating phone data into Compass export...")
        integrate_phone_into_compass(compass_file, phone_file, merged_file, logger=logger)
    else: # No phone file, copy compass_file to merged_file and initialize columns
        if logger: logger("No Phone file provided; using Compass file as base for merged output and initializing columns.")
        try:
            # Load original compass data
            temp_data_from_compass = load_compass_csv(compass_file)
            if not temp_data_from_compass:
                if logger: logger("[ERROR] Compass file is empty or unreadable. Cannot proceed.")
                 # Create an empty merged file with minimal headers if compass was truly empty
                if not os.path.exists(merged_file) or (os.path.exists(merged_file) and os.path.getsize(merged_file) == 0):
                    with open(merged_file, "w", newline="", encoding="utf-8") as mf_empty:
                        # Attempt to get headers from original compass file if it exists, even if load_compass_csv failed due to content
                        headers = ["First Name", "Last Name", "Email"] # Basic default
                        try:
                            with open(compass_file, "r", encoding="utf-8-sig") as cf_h:
                                r = csv.reader(cf_h)
                                h_temp = next(r, None)
                                if h_temp : headers = h_temp
                        except: pass # Ignore if can't read headers
                        
                        extra_cols_init = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification", "Groups"]
                        all_headers = list(dict.fromkeys(headers + extra_cols_init))
                        writer_empty = csv.DictWriter(mf_empty, fieldnames=all_headers)
                        writer_empty.writeheader()
                return extracted_addresses_file if mls_files and os.path.exists(extracted_addresses_file) else None, merged_file, []


            original_headers = list(temp_data_from_compass[0].keys()) if temp_data_from_compass else []
            extra_cols_to_ensure_init = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification", "Groups"]
            
            for row in temp_data_from_compass:
                for col in extra_cols_to_ensure_init:
                    row.setdefault(col, "")
                if not row.get("Changes Made"): # Ensure "Changes Made" is initialized
                    row["Changes Made"] = "No changes made."
            
            final_fieldnames_for_initial_merged = original_headers + [col for col in extra_cols_to_ensure_init if col not in original_headers]
            final_fieldnames_for_initial_merged = list(dict.fromkeys(final_fieldnames_for_initial_merged))

            with open(merged_file, mode="w", newline="", encoding="utf-8") as f_init_merge:
                writer = csv.DictWriter(f_init_merge, fieldnames=final_fieldnames_for_initial_merged, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(temp_data_from_compass)
            if logger: logger("Initialized merged file from Compass data.")

        except Exception as e_copy:
            if logger: logger(f"[CRITICAL ERROR] Error preparing initial merged file from Compass: {e_copy}")
            return extracted_addresses_file if mls_files and os.path.exists(extracted_addresses_file) else None, merged_file, []

    # Step 2: Load the merged data for subsequent classifications
    final_data_in_memory = load_compass_csv(merged_file)
    if not final_data_in_memory:
        if logger: logger("[ERROR] Merged data is empty after initial processing step. Aborting further classifications.")
        return extracted_addresses_file if mls_files and os.path.exists(extracted_addresses_file) else None, merged_file, []

    current_headers_of_final_data = list(final_data_in_memory[0].keys()) if final_data_in_memory else []
    final_data_cat_map = categorize_columns(current_headers_of_final_data)
    email_columns_for_classification = final_data_cat_map.get("email", [])

    # Step 3: Classifications (modifying final_data_in_memory)
    if email_columns_for_classification:
        classify_agents(final_data_in_memory, email_columns_for_classification)
        classify_vendors(final_data_in_memory, email_columns_for_classification) # classify_vendors also uses email_columns
    else:
        if logger: logger("[WARNING] No email columns found in merged data. Agent and Vendor classification might be incomplete.")
    
    if extracted_addresses: # Only run if MLS data was processed
        classify_clients_simplified(final_data_in_memory, extracted_addresses, logger=logger) 
    else:
        if logger: logger("Skipping client classification (HAD update) as no MLS addresses were available.")

    # Update groups based on all classifications done so far
    update_groups_with_classification(final_data_in_memory, logger=logger)
    
    # Step 4: Final write of the fully processed final_data_in_memory to merged_file
    # Preserve original Compass export column order + new columns
    original_compass_headers = []
    try:
        with open(compass_file, "r", encoding="utf-8-sig") as f_orig_headers:
            reader = csv.reader(f_orig_headers)
            original_compass_headers = next(reader, [])
            if not original_compass_headers and final_data_in_memory : 
                original_compass_headers = list(final_data_in_memory[0].keys()) # Fallback
    except Exception as e_orig_h:
        if logger: logger(f"Notice: Could not read original Compass file headers for column ordering: {e_orig_h}. Using current headers.")
        if final_data_in_memory: original_compass_headers = list(final_data_in_memory[0].keys())
        else: original_compass_headers = ["First Name", "Last Name", "Email", "Changes Made", "Category", "Home Anniversary Date", "Vendor Classification", "Client Classification", "Groups"] # Absolute fallback

    all_known_extra_columns = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification", "Groups"]
    
    # Determine final fieldnames for the output merged_file
    # Start with original compass order, then add any *new* columns that are in all_known_extra_columns
    # and actually present in the data (final_data_in_memory[0].keys())
    
    final_ordered_fieldnames = list(original_compass_headers) # Make a mutable copy
    if final_data_in_memory:
        current_data_keys = final_data_in_memory[0].keys()
        for extra_col in all_known_extra_columns:
            if extra_col in current_data_keys and extra_col not in final_ordered_fieldnames:
                final_ordered_fieldnames.append(extra_col)
    else: # If final_data_in_memory is empty for some reason, ensure these columns are in headers
        for extra_col in all_known_extra_columns:
            if extra_col not in final_ordered_fieldnames:
                 final_ordered_fieldnames.append(extra_col)

    final_ordered_fieldnames = list(dict.fromkeys(final_ordered_fieldnames)) # Deduplicate


    merged_write_successful = False
    try:
        with open(merged_file, mode="w", newline="", encoding="utf-8") as f_final_merge:
            writer = csv.DictWriter(f_final_merge, fieldnames=final_ordered_fieldnames, extrasaction="ignore")
            writer.writeheader()
            # Ensure all rows conform to the final_ordered_fieldnames
            for row_to_write in final_data_in_memory:
                writer.writerow({k: row_to_write.get(k, "") for k in final_ordered_fieldnames})
        if logger:
            logger(f"Final classified and merged data written to {merged_file}")
        merged_write_successful = True
    except Exception as e_final_save:
        if logger:
            logger(f"[CRITICAL ERROR] Error saving final fully processed merged CSV to {merged_file}: {e_final_save}")
        # merged_file on disk might be corrupted/empty/old.

    # Step 5: Export records that had changes
    import_files_generated = []
    if merged_write_successful:
        if logger: logger(f"Proceeding to export records with changes from {merged_file}.")
        export_updated_records(merged_file, import_output_dir, logger=logger)
        if os.path.exists(import_output_dir):
            import_files_generated = [
                os.path.join(import_output_dir, f) for f in os.listdir(import_output_dir)
                if os.path.isfile(os.path.join(import_output_dir, f))
            ]
    else:
        if logger:
            logger(f"Skipping export of updated records because the final merge to {merged_file} failed or was incomplete.")
    
    # Determine which files to return based on whether they were processed/exist
    final_extracted_output_file = extracted_addresses_file if mls_files and os.path.exists(extracted_addresses_file) else None
    final_merged_output_file = merged_file if os.path.exists(merged_file) else None # Path exists, content depends on success

    return final_extracted_output_file, final_merged_output_file, import_files_generated
