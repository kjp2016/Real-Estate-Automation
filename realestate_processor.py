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
# Ensure API key is loaded only once and handled securely if needed outside Streamlit
try:
    openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (AttributeError, KeyError): # Handle cases where st.secrets might not be available
    openai_api_key_env = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key_env:
        # This is a fallback for local testing if st.secrets and env var are missing.
        # In a deployed Streamlit app, st.secrets should ideally be used.
        print("Warning: OPENAI_API_KEY not found in st.secrets or environment variables.")
        openai_client = None # Or raise an error, or use a placeholder client if applicable
    else:
        openai_client = openai.OpenAI(api_key=openai_api_key_env)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_client = None


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

def extract_addresses_with_ai(text: str, logger=None) -> List[dict]:
    if not openai_client:
        if logger: logger("[ERROR] OpenAI client not initialized. Cannot extract addresses with AI.")
        return []
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
            model="gpt-4o", # Consider making model configurable if needed
            messages=messages,
            response_format={"type": "json_schema", "json_schema": address_schema},
            temperature=0
        )
        structured_response = response.choices[0].message.content
        parsed_data = json.loads(structured_response)
        return parsed_data.get("addresses", [])
    except Exception as e:
        if logger: logger(f"[ERROR] Error parsing AI response or with OpenAI API: {e}")
        else: print(f"Error parsing AI response or with OpenAI API: {e}")
        return []


def extract_addresses_with_ai_chunked(text: str, max_lines: int = 50, logger=None) -> List[dict]:
    lines = text.splitlines()
    all_addresses = []
    if not text.strip():
        if logger: logger("[DEBUG] extract_addresses_with_ai_chunked: Received empty text.")
        return all_addresses

    if len(lines) <= max_lines:
        if logger: logger("[DEBUG] Sending full text to OpenAI (non-chunked)")
        all_addresses.extend(extract_addresses_with_ai(text, logger=logger))
    else:
        if logger: logger(f"[DEBUG] Splitting text into {len(lines) // max_lines + 1} chunks")
        for i in range(0, len(lines), max_lines):
            chunk = "\n".join(lines[i:i+max_lines])
            if logger: logger(f"[DEBUG] Sending chunk {i//max_lines + 1} to OpenAI")
            result = extract_addresses_with_ai(chunk, logger=logger)
            all_addresses.extend(result)
    return all_addresses

def extract_addresses_row_by_row(df: pd.DataFrame, logger=None) -> List[dict]:
    extracted_addresses = []
    if df.empty:
        if logger: logger("[DEBUG] extract_addresses_row_by_row: DataFrame is empty.")
        return extracted_addresses

    for idx, row in df.iterrows():
        row_text_parts = []
        for col in df.columns:
            val = str(row[col]).strip()
            if val and val.lower() not in ["nan", "none", ""]: # More robust check for empty-like values
                row_text_parts.append(f"{col}: {val}")
        
        if row_text_parts:
            prompt_text = "\n".join(row_text_parts)
            if logger: logger(f"[DEBUG] Extracting from row {idx + 1} text: {prompt_text[:200]}...") # Log snippet
            try:
                result = extract_addresses_with_ai(prompt_text, logger=logger)
                extracted_addresses.extend(result)
            except Exception as e:
                if logger: logger(f"[ERROR] Row {idx + 1} AI extraction failed: {e}")
        elif logger:
            logger(f"[DEBUG] Row {idx+1} was empty or all 'nan'/'none'. Skipping AI extraction for this row.")
    return extracted_addresses


def extract_text_from_pdf(pdf_path: str, logger=None) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                # else:
                #     if logger: logger(f"[DEBUG] No text extracted from PDF '{os.path.basename(pdf_path)}' page {i+1}")
    except Exception as e:
        if logger: logger(f"[ERROR] Error processing PDF {os.path.basename(pdf_path)}: {e}")
        else: print(f"Error processing PDF {os.path.basename(pdf_path)}: {e}")
    return " ".join(text.split()) # Normalize whitespace


def extract_text_from_csv(file_path: str, logger=None) -> str: # Not directly used for AI address extraction in current flow but kept
    text_parts = []
    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False) # Read all as string, keep NA as is
        for _, row in df.iterrows():
            row_text_parts = []
            for col in df.columns:
                val = str(row[col]).strip() # Ensure it's a string before stripping
                if val and val.lower() != 'nan': # Check for 'nan' string
                    row_text_parts.append(f"{col}: {val}")
            if row_text_parts:
                text_parts.append("\n".join(row_text_parts))
    except Exception as e:
        if logger: logger(f"[ERROR] Error reading CSV for text extraction {os.path.basename(file_path)}: {e}")
        else: print(f"Error reading CSV for text extraction {os.path.basename(file_path)}: {e}")
    return "\n---\n".join(text_parts).strip()



def extract_addresses_from_excel(file_path: str, logger=None) -> List[dict]:
    extracted_addresses = []
    try:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            if logger: logger(f"[DEBUG] Reading Excel sheet: {sheet_name}")
            df = pd.read_excel(xls, sheet_name, dtype=str, keep_default_na=False) # Read all as string
            df.fillna("", inplace=True) # Replace actual NaN/NaT with empty strings for consistency
            if logger: logger(f"[DEBUG] Processing sheet: {sheet_name} with {len(df)} rows for address extraction.")
            sheet_addresses = extract_addresses_row_by_row(df, logger=logger)
            extracted_addresses.extend(sheet_addresses)
    except Exception as e:
        if logger: logger(f"[ERROR] Error processing Excel {os.path.basename(file_path)}: {e}")
    return extracted_addresses


def extract_and_save_addresses(file_paths: List[str], output_file: str, logger=None):
    all_extracted_addresses = []
    if not file_paths:
        if logger: logger("[INFO] No MLS/Sales Activity files provided for address extraction.")
        # Ensure an empty CSV with headers is created if the file is expected later
        # This prevents FileNotFoundError if load_extracted_addresses is called.
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["Street Address", "Unit", "City", "State", "Zip Code", "Home Anniversary Date"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        if logger: logger(f"Created empty extracted addresses file: {output_file}")
        return

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[-1].lower()
        file_name = os.path.basename(file_path)
        addresses_from_file = []
        if ext == ".pdf":
            if logger: logger(f"Processing PDF for addresses: {file_name}")
            text = extract_text_from_pdf(file_path, logger=logger)
            if text:
                addresses_from_file = extract_addresses_with_ai_chunked(text, max_lines=50, logger=logger)
            else:
                if logger: logger(f"[INFO] No text extracted from PDF: {file_name}")
        elif ext == ".csv":
            if logger: logger(f"Processing CSV for addresses: {file_name}")
            try:
                # Read all as string, keep NA as is, then replace with empty string
                df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
                df.fillna("", inplace=True)
                addresses_from_file = extract_addresses_row_by_row(df, logger=logger)
            except Exception as e:
                if logger: logger(f"[ERROR] Failed to process CSV {file_name}: {e}")
        elif ext in [".xls", ".xlsx"]:
            if logger: logger(f"Processing Excel for addresses: {file_name}")
            addresses_from_file = extract_addresses_from_excel(file_path, logger=logger)
        else:
            if logger: logger(f"Unsupported file type for address extraction: {file_name}")
            continue
        
        if addresses_from_file:
            all_extracted_addresses.extend(addresses_from_file)
            if logger: logger(f"Found {len(addresses_from_file)} addresses in {file_name}.")
        elif logger:
             logger(f"No addresses found/extracted from {file_name}.")
    
    fieldnames = ["Street Address", "Unit", "City", "State", "Zip Code", "Home Anniversary Date"]
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if all_extracted_addresses:
                for address in all_extracted_addresses:
                    # Ensure all keys are present, defaulting to empty string
                    row_to_write = {key: address.get(key, "") for key in fieldnames}
                    writer.writerow(row_to_write)
        if logger: logger(f"Extracted {len(all_extracted_addresses)} total addresses saved to {output_file}")
    except Exception as e:
        if logger: logger(f"[ERROR] Error saving extracted addresses to {output_file}: {e}")

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
        if not isinstance(email, str): continue # Skip if email is not a string
        match = re.search(r"@([\w.-]+)$", email)
        if match:
            domain = match.group(1).lower()
            if domain in BROKERAGE_EMAIL_DOMAINS:
                return True
    return False


def is_vendor(emails: List[str]) -> bool:
    if not emails: return False
    for email in emails:
        if not isinstance(email, str): continue
        match = re.search(r"@([\w.-]+)$", email)
        if match:
            domain = match.group(1).lower()
            if any(keyword in domain for keyword in VENDOR_KEYWORDS):
                return True
    return False

def normalize_phone(phone: str) -> str:
    if not isinstance(phone, str): phone = str(phone) # Ensure it's a string
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
        h_lower = str(header).lower() # Ensure header is string
        if ("first" in h_lower and "name" in h_lower) or "fname" in h_lower or "f_name" in h_lower:
            categories["first_name"].append(header)
        elif ("last" in h_lower and "name" in h_lower) or "lname" in h_lower or "l_name" in h_lower:
            categories["last_name"].append(header)
        elif "title" in h_lower:
            categories["title"].append(header)
        elif "company" in h_lower: # Could be "company name"
            categories["company"].append(header)
        elif "email" in h_lower:
            categories["email"].append(header)
        elif "phone" in h_lower or "mobile" in h_lower:
            categories["phone"].append(header)
        elif any(x in h_lower for x in ["address", "street", "city", "zip", "state", "country", "line 1", "line 2"]):
            categories["address"].append(header)
    return categories


def load_compass_csv(compass_file: str, logger=None) -> List[Dict[str, str]]:
    try:
        with open(compass_file, mode="r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        if logger: logger(f"[ERROR] Compass file not found: {compass_file}")
    except Exception as e:
        if logger: logger(f"[ERROR] Error loading Compass CSV {compass_file}: {e}")
    return []


def load_phone_csv(phone_file: str, logger=None) -> List[Dict[str, str]]:
    try:
        with open(phone_file, mode="r", encoding="utf-8-sig", errors="replace") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        if logger: logger(f"[ERROR] Phone file not found: {phone_file}")
    except Exception as e:
        if logger: logger(f"[ERROR] Error loading Phone CSV {phone_file}: {e}")
    return []


def extract_field(row: Dict[str, str], columns: List[str]) -> str:
    for col in columns:
        val = row.get(col, "").strip()
        if val:
            return val
    return ""


def build_name_key(row: Dict[str, str], cat_map: Dict[str, List[str]]) -> str:
    first_name = extract_field(row, cat_map.get("first_name", [])).lower()
    last_name = extract_field(row, cat_map.get("last_name", [])).lower()
    return f"{first_name} {last_name}".strip()


def fuzzy_name_match(name_key: str, all_keys: List[str], threshold: int = 97) -> Tuple[str, float]:
    if not name_key or not all_keys:
        return (None, 0)
    best = process.extractOne(name_key, all_keys, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return (best[0], best[1])
    return (None, 0)


def extract_phone_address_component(header: str) -> Tuple[str, str]:
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
    phone_cat_map: Dict[str, List[str]],
    logger=None
) -> Tuple[Dict[str, str], List[str]]:
    changes = []
    
    phone_addr_groups = {} 
    for col in phone_cat_map.get("address", []):
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
    
    compass_groups = {}
    for col in compass_cat_map.get("address", []):
        col_lower = col.lower()
        m = re.search(r'address\s*(\d*)', col_lower) # Looks for "address" possibly followed by a number
        group_id_match = m.group(1) if m and m.group(1) else None
        
        # More robust group_id determination
        if group_id_match: # "Address 2 Line 1" -> group "2"
            group_id = group_id_match
        elif "address" in col_lower and not any(char.isdigit() for char in col_lower.split("address")[1].split(" ")[0]):
            # Catches "Address Line 1" (no number directly after "Address") -> group "1"
            group_id = "1" 
        else: # Fallback or other patterns not fitting the numbered groups
            group_id = "default_group" # Or skip, or handle differently

        component = None
        if "line 1" in col_lower or ("street" in col_lower and "line" not in col_lower and "city" not in col_lower and "state" not in col_lower and "zip" not in col_lower and "country" not in col_lower):
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
            compass_groups[group_id][component] = col # Store the original column name

    for phone_group_label, phone_addr_details in phone_addr_groups.items():
        phone_street_to_match = phone_addr_details.get("street", "").strip().lower()
        if not phone_street_to_match:
            continue

        already_present = False
        for compass_group_id, compass_component_map in compass_groups.items():
            compass_street_col_name = compass_component_map.get("street")
            if compass_street_col_name:
                compass_street_val = compass_row.get(compass_street_col_name, "").strip().lower()
                if phone_street_to_match == compass_street_val:
                    already_present = True
                    break
        if already_present:
            if logger: logger(f"[DEBUG merge_address] Phone street '{phone_street_to_match}' from group '{phone_group_label}' already present in Compass row.")
            continue
        
        target_compass_group_id = None
        # Prefer matching numbered groups first if possible, then find any empty group
        sorted_compass_group_ids = sorted(compass_groups.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))

        for group_id in sorted_compass_group_ids:
            comp_dict = compass_groups[group_id]
            group_empty = all(not compass_row.get(col_name, "").strip() for col_name in comp_dict.values())
            if group_empty:
                target_compass_group_id = group_id
                break
        
        if target_compass_group_id is None:
            if logger: logger(f"[DEBUG merge_address] No empty address group in Compass row to merge phone address: {phone_addr_details}")
            continue
            
        if logger: logger(f"[DEBUG merge_address] Merging phone address {phone_addr_details} into Compass group {target_compass_group_id}")
        for phone_comp_type, phone_comp_val in phone_addr_details.items():
            compass_col_for_comp = compass_groups[target_compass_group_id].get(phone_comp_type)
            if compass_col_for_comp: # If this component type exists in the target compass group
                 # Only fill if compass field is empty
                if not compass_row.get(compass_col_for_comp, "").strip():
                    compass_row[compass_col_for_comp] = phone_comp_val
                    changes.append(f"Address Group {target_compass_group_id} -> {compass_col_for_comp}: {phone_comp_val}")
    
    return compass_row, changes


def merge_phone_into_compass(
    compass_row: Dict[str, str],
    phone_row: Dict[str, str],
    compass_cat_map: Dict[str, List[str]],
    phone_cat_map: Dict[str, List[str]],
    logger=None
) -> Tuple[Dict[str, str], List[str]]:
    changes = []
    
    # Merge missing email addresses
    phone_emails = [phone_row.get(col, "").strip().lower() for col in phone_cat_map.get("email",[]) if phone_row.get(col, "").strip()]
    existing_emails_in_compass = {compass_row.get(col, "").strip().lower() for col in compass_cat_map.get("email",[]) if compass_row.get(col, "").strip()}
    
    available_compass_email_cols = [col for col in compass_cat_map.get("email",[]) if not compass_row.get(col, "").strip()]
    
    for email_to_add in phone_emails:
        if email_to_add not in existing_emails_in_compass:
            if available_compass_email_cols:
                target_col = available_compass_email_cols.pop(0) # Use the first available empty slot
                compass_row[target_col] = email_to_add # Original casing from phone_row might be better here if desired
                changes.append(f"Email->{target_col}: {email_to_add}")
                existing_emails_in_compass.add(email_to_add) # Add to set of existing for next check
            # else:
                # if logger: logger(f"[DEBUG merge_phone] No empty email slot in Compass to add: {email_to_add}")


    # Merge missing phone numbers
    phone_numbers_original = [phone_row.get(col, "").strip() for col in phone_cat_map.get("phone",[]) if phone_row.get(col, "").strip()]
    normalized_phone_numbers_from_phone_row = {normalize_phone(p) for p in phone_numbers_original if p}
    
    existing_normalized_phones_in_compass = {normalize_phone(compass_row.get(col, "").strip()) for col in compass_cat_map.get("phone",[]) if compass_row.get(col, "").strip()}
    available_compass_phone_cols = [col for col in compass_cat_map.get("phone",[]) if not compass_row.get(col, "").strip()]

    for original_phone_val in phone_numbers_original:
        norm_phone_to_add = normalize_phone(original_phone_val)
        if norm_phone_to_add not in existing_normalized_phones_in_compass:
            if available_compass_phone_cols:
                target_col = available_compass_phone_cols.pop(0)
                compass_row[target_col] = original_phone_val 
                changes.append(f"Phone->{target_col}: {original_phone_val}")
                existing_normalized_phones_in_compass.add(norm_phone_to_add)
            # else:
                # if logger: logger(f"[DEBUG merge_phone] No empty phone slot in Compass to add: {original_phone_val}")

    # Update category based on combined email domains
    # Collect all emails after potential merge for classification
    all_current_emails_in_compass_row = [compass_row.get(col, "").strip().lower() for col in compass_cat_map.get("email",[]) if compass_row.get(col, "").strip()]
    if is_real_estate_agent(all_current_emails_in_compass_row):
        if compass_row.get("Category","").lower() != "agent": # Only log change if it's new
            compass_row["Category"] = "Agent"
            changes.append("Category set to Agent based on phone data email")

    return compass_row, changes


def classify_agents(final_data: List[Dict[str, str]], email_columns: List[str], logger=None):
    if not email_columns and logger:
        logger("[WARNING classify_agents] No email columns identified for agent classification.")
        return

    for row in final_data:
        all_emails_in_row = set()
        for col in email_columns: # email_columns are the names of columns that contain emails
            email_val = row.get(col, "").strip().lower()
            if email_val:
                all_emails_in_row.add(email_val)
        
        # Preserve existing category if it's already "Agent" from phone merge, otherwise classify
        # Also, ensure "Non-Agent" is explicitly set if not an agent and category is blank
        current_category = row.get("Category", "").strip().lower()
        is_agent_by_email = is_real_estate_agent(list(all_emails_in_row))

        if is_agent_by_email:
            if current_category != "agent":
                row["Category"] = "Agent"
                # Change logging for this specific update is usually handled by integrate_phone_into_compass or later stages
        elif not current_category: # If category is blank and not an agent
             row["Category"] = "Non-Agent"


def classify_vendors(final_data: List[Dict[str, str]], email_columns: List[str], logger=None):
    if not email_columns and logger:
        logger("[WARNING classify_vendors] No email columns identified for vendor classification.")
        return

    for row in final_data:
        all_emails_in_row = set()
        for col in email_columns:
            email_val = row.get(col, "").strip().lower()
            if email_val:
                all_emails_in_row.add(email_val)
        
        if is_vendor(list(all_emails_in_row)):
            if row.get("Vendor Classification", "") != "Vendor":
                row["Vendor Classification"] = "Vendor"
                current_changes = row.get("Changes Made", "")
                change_msg = "Classified as Vendor"
                if current_changes and current_changes.lower() != "no changes made.":
                    row["Changes Made"] = f"{current_changes} | {change_msg}"
                else:
                    row["Changes Made"] = change_msg


def integrate_phone_into_compass(compass_file: str, phone_file: str, output_file: str, logger=None):
    compass_data = load_compass_csv(compass_file, logger=logger)
    phone_data = load_phone_csv(phone_file, logger=logger)
    
    if not compass_data:
        if logger: logger("Compass CSV has no data. Cannot integrate phone data.")
        # If compass_file itself was empty, output_file will be empty or not created by load_compass_csv.
        # We might want to create an empty output_file with Compass headers if that's the expectation.
        # For now, just returning as there's nothing to merge into.
        return

    if not phone_data:
        if logger: logger("Phone CSV has no data. Writing original Compass data to output.")
        # Write original compass_data to output_file, ensuring necessary columns exist
        original_headers = list(compass_data[0].keys()) if compass_data else []
        extra_cols_to_ensure = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification", "Groups"]
        for row_to_prep in compass_data:
            for col_to_ensure in extra_cols_to_ensure:
                row_to_prep.setdefault(col_to_ensure, "")
            row_to_prep.setdefault("Changes Made", "No changes made.")
        
        final_fieldnames_no_phone = original_headers + [col for col in extra_cols_to_ensure if col not in original_headers]
        try:
            with open(output_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=final_fieldnames_no_phone, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(compass_data)
            if logger: logger(f"Copied original Compass data to {output_file} as no phone data was provided.")
        except Exception as e:
            if logger: logger(f"[ERROR] Error writing original Compass data (no phone data): {e}")
        return

    compass_cat_map = categorize_columns(list(compass_data[0].keys()))
    phone_cat_map = categorize_columns(list(phone_data[0].keys()))

    compass_name_keys = [build_name_key(row, compass_cat_map) for row in compass_data]
    # Create a deep copy for modification
    updated_compass_data = [row.copy() for row in compass_data]


    for phone_contact_row in phone_data:
        phone_name_key = build_name_key(phone_contact_row, phone_cat_map)
        if not phone_name_key.strip():
            if logger: logger("[DEBUG] Skipping phone contact with no name.")
            continue

        best_compass_key, match_score = fuzzy_name_match(phone_name_key, compass_name_keys, threshold=97)
        
        if best_compass_key:
            try:
                # Find all indices that match best_compass_key, though typically it should be one unless names are identical
                indices_to_update = [i for i, key in enumerate(compass_name_keys) if key == best_compass_key]
                for match_idx in indices_to_update: # Usually only one index
                    compass_row_to_update = updated_compass_data[match_idx] # Get the row from the copied list
                    
                    # Perform merge operations
                    # Pass a copy of compass_row_to_update to avoid direct modification by sub-functions if they don't return the modified dict
                    merged_row_after_phone, phone_field_changes = merge_phone_into_compass(
                        compass_row_to_update.copy(), phone_contact_row, compass_cat_map, phone_cat_map, logger=logger
                    )
                    merged_row_after_address, address_field_changes = merge_address_into_compass(
                        merged_row_after_phone, phone_contact_row, compass_cat_map, phone_cat_map, logger=logger
                    )
                    
                    all_new_changes_this_pass = phone_field_changes + address_field_changes
                    
                    # Consolidate "Changes Made"
                    existing_changes_made = compass_row_to_update.get("Changes Made", "") # Get from the version we are updating
                    if not existing_changes_made or existing_changes_made.lower() == "no changes made.":
                        existing_changes_made = ""

                    if all_new_changes_this_pass:
                        change_log_message = f"Matched Phone Contact (Name: '{phone_name_key}', Score: {match_score}%) | {'; '.join(all_new_changes_this_pass)}"
                        merged_row_after_address["Changes Made"] = f"{existing_changes_made} | {change_log_message}" if existing_changes_made else change_log_message
                    elif not existing_changes_made: # No new changes and no prior changes
                         merged_row_after_address["Changes Made"] = "No changes made."
                    else: # No new changes, but prior changes existed
                        merged_row_after_address["Changes Made"] = existing_changes_made


                    updated_compass_data[match_idx] = merged_row_after_address # Update the list with the fully merged row
            except ValueError: # Should not happen if best_compass_key is from compass_name_keys
                if logger: logger(f"[ERROR] Matched key '{best_compass_key}' not found in compass_name_keys list during phone integration.")
        # else:
            # if logger: logger(f"[DEBUG] Phone contact '{phone_name_key}' not found in Compass export; skipping merge for this contact.")

    final_data_for_output = updated_compass_data
    
    # Ensure all necessary columns exist in all rows before writing
    original_headers = list(compass_data[0].keys()) if compass_data else []
    extra_columns_to_ensure = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification", "Groups"]
    for row_to_prep in final_data_for_output:
        for col_to_ensure in extra_columns_to_ensure:
            row_to_prep.setdefault(col_to_ensure, "")
        row_to_prep.setdefault("Changes Made", "No changes made.") # Default if no changes logged

    # Determine final fieldnames for the output CSV
    final_fieldnames = original_headers + [col for col in extra_columns_to_ensure if col not in original_headers]
    
    try:
        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction='ignore') # ignore extra fields not in headers
            writer.writeheader()
            writer.writerows(final_data_for_output)
        if logger: logger(f"Phone data integration complete. Merged data written to {output_file}")
    except Exception as e:
        if logger: logger(f"[ERROR] Error writing merged CSV after phone integration: {e}")


def load_extracted_addresses(address_file: str, logger=None) -> List[dict]:
    extracted_records = []
    try:
        with open(address_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                extracted_records.append(row)
        if logger: logger(f"[INFO] Loaded {len(extracted_records)} addresses from {address_file}")
    except FileNotFoundError:
        if logger: logger(f"[INFO] Extracted addresses file not found: {address_file}. Returning empty list.")
    except Exception as e:
        if logger: logger(f"[ERROR] Error loading extracted addresses from {address_file}: {e}")
    return extracted_records


def classify_clients(final_data: List[Dict[str, str]], address_columns: List[str], extracted_addresses: List[dict], logger=None):
    if logger: logger("[INFO classify_clients] Starting client classification.")

    def normalize_extracted_address_key(addr: dict) -> str:
        street = addr.get("Street Address", "").strip().lower()
        return street

    def get_compass_address_keys(row_data: Dict[str, str], compass_addr_cols: List[str]) -> List[str]:
        street_keys = []
        for col_name in compass_addr_cols:
            col_l = col_name.lower()
            # More specific conditions to target primary street lines
            is_street_line_1 = ("address" in col_l and ("line 1" in col_l or "line1" in col_l))
            is_generic_street = ("street" in col_l and "address" in col_l) # e.g. "Street Address"
            is_unadorned_street = col_l == "street" # if column is just "Street"

            if is_street_line_1 or is_generic_street or is_unadorned_street:
                street_val = row_data.get(col_name, "").strip()
                if street_val:
                    street_keys.append(street_val.lower())
        return list(dict.fromkeys(street_keys)) # Return unique street keys

    if not extracted_addresses:
        if logger: logger("[DEBUG classify_clients] Extracted_addresses list is empty. No client classification possible.")
        return

    extracted_lookup = {}
    for record in extracted_addresses:
        key = normalize_extracted_address_key(record)
        if key: # Only add if key is not empty
            if key not in extracted_lookup: # Prioritize first encountered MLS record for a given street
                extracted_lookup[key] = record
            # else:
                # if logger: logger(f"[DEBUG classify_clients] Duplicate MLS street key '{key}'. Keeping first encountered: {extracted_lookup[key]}")
    
    extracted_street_keys = list(extracted_lookup.keys())
    if not extracted_street_keys:
        if logger: logger("[DEBUG classify_clients] No valid unique street keys from MLS data. Cannot classify clients.")
        return
    
    if logger:
        logger(f"[DEBUG classify_clients] Number of unique MLS street keys for matching: {len(extracted_street_keys)}")
        # logger(f"[DEBUG classify_clients] Sample MLS street keys: {extracted_street_keys[:3]}")

    for row_idx, row_data in enumerate(final_data):
        # If already client and has HAD, skip (unless we want to allow updates to HAD)
        if row_data.get("Client Classification", "") == "Client" and row_data.get("Home Anniversary Date", ""):
            continue

        # `address_columns` here are all columns identified by `categorize_columns` as address-related for this row
        compass_street_keys_for_row = get_compass_address_keys(row_data, address_columns) 
        
        # contact_name_for_log = f"{row_data.get('First Name', '')} {row_data.get('Last Name', '')}".strip() # For debug

        best_match_for_this_row = None # To store the (score, mls_record) of the best date-providing match

        for compass_street_key in compass_street_keys_for_row:
            if not compass_street_key:
                continue
            
            # Find the best fuzzy match for this compass_street_key among all MLS street keys
            match_result = process.extractOne(compass_street_key, extracted_street_keys, scorer=fuzz.WRatio)
            
            if match_result and match_result[1] >= 93: # If score is good
                matched_mls_street_key = match_result[0]
                score = match_result[1]
                matched_mls_record = extracted_lookup[matched_mls_street_key]
                
                # Mark as client if not already
                if row_data.get("Client Classification", "") != "Client":
                    row_data["Client Classification"] = "Client"
                    current_changes = row_data.get("Changes Made", "")
                    change_msg = "Classified as Client"
                    row_data["Changes Made"] = f"{current_changes} | {change_msg}" if current_changes and current_changes.lower() != "no changes made." else change_msg
                
                # Check for Home Anniversary Date
                home_anniv_date_from_mls = matched_mls_record.get("Home Anniversary Date")
                if home_anniv_date_from_mls:
                    # If this match provides a date, and it's better than any previous match for THIS ROW
                    if best_match_for_this_row is None or score > best_match_for_this_row[0]:
                        best_match_for_this_row = (score, matched_mls_record)
                    # If a date is found, we can break if we only want the first date.
                    # If we want the date from the *best score* match, we continue the inner loop.
                    # Current logic: takes date from first good match providing a date.
                    if row_data.get("Home Anniversary Date","") != home_anniv_date_from_mls:
                        row_data["Home Anniversary Date"] = home_anniv_date_from_mls
                    break # Found a good enough match with a date, process next Compass row

        # After checking all street keys for this Compass row:
        # If no match provided a date directly but we found a date from the "best_match_for_this_row" logic (if implemented)
        # This part is slightly redundant if the break above is hit with a date.
        # The break above ensures we take the date from the first good match.
        # If we wanted to iterate all compass_street_keys and then pick the HAD from the highest scoring match:
        # if best_match_for_this_row:
        #     had_to_set = best_match_for_this_row[1].get("Home Anniversary Date")
        #     if had_to_set and row_data.get("Home Anniversary Date", "") != had_to_set:
        #         row_data["Home Anniversary Date"] = had_to_set
                # Optionally log "HAD updated from best match"

    if logger: logger("[INFO classify_clients] Client classification finished.")
            
def update_groups_with_classification(final_data: List[Dict[str, str]], logger=None) -> None:
    if logger: logger("[INFO] Updating groups based on classification...")
    for row in final_data:
        groups_str = row.get("Groups", "").strip()
        current_groups_set = {g.strip().lower() for g in groups_str.split(',') if g.strip()}
        
        changes_for_groups = []
        
        if row.get("Category", "").strip().lower() == "agent":
            if "agents" not in current_groups_set:
                current_groups_set.add("agents")
                changes_for_groups.append("Added Agents to Groups")
        
        if row.get("Vendor Classification", "").strip().lower() == "vendor":
            if "vendors" not in current_groups_set:
                current_groups_set.add("vendors")
                changes_for_groups.append("Added Vendors to Groups")
        
        final_groups_list = sorted([g.title() for g in current_groups_set if g])
        new_groups_str = ",".join(final_groups_list)

        if row.get("Groups","") != new_groups_str : # Only update if there's an actual change to groups string
            row["Groups"] = new_groups_str
            if changes_for_groups: # Only add to "Changes Made" if we actually added "Agents" or "Vendors"
                current_changes_made = row.get("Changes Made", "").strip()
                group_change_log = "; ".join(changes_for_groups)
                if current_changes_made and current_changes_made.lower() != "no changes made.":
                    row["Changes Made"] = f"{current_changes_made} | {group_change_log}"
                else:
                    row["Changes Made"] = group_change_log
    if logger: logger("[INFO] Group update finished.")


def export_updated_records(merged_file: str, import_output_dir: str, logger=None):
    exclude_cols = {
        "Created At", "Last Contacted", "Changes Made", "Category", 
        "Agent Classification", "Client Classification", "Vendor Classification"
    }
    try:
        with open(merged_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            original_order = reader.fieldnames or []
    except FileNotFoundError:
        if logger: logger(f"[ERROR] Merged file {merged_file} not found for export.")
        return
    except Exception as e:
        if logger: logger(f"[ERROR] Error reading merged file {merged_file} for export: {e}")
        return

    if not rows:
        if logger: logger("[INFO] No data in merged file to export.")
        return

    updated_rows = [
        row for row in rows 
        if row.get("Changes Made", "").strip().lower() not in {"", "no changes made."}
    ]

    if not updated_rows:
        if logger: logger("[INFO] No records with 'Changes Made' found. Nothing to export for Compass import.")
        return
    if logger: logger(f"[INFO] Found {len(updated_rows)} updated records for Compass import.")

    import_fieldnames = [col for col in original_order if col not in exclude_cols]

    chunk_size = 2000
    total_chunks = (len(updated_rows) + chunk_size - 1) // chunk_size

    if not os.path.exists(import_output_dir):
        try:
            os.makedirs(import_output_dir)
        except OSError as e:
            if logger: logger(f"[ERROR] Could not create import output directory {import_output_dir}: {e}")
            return


    for i in range(total_chunks):
        chunk_data = updated_rows[i*chunk_size : (i+1)*chunk_size]
        # Filter each row in the chunk to only include import_fieldnames
        chunk_to_write = []
        for original_row_in_chunk in chunk_data:
            filtered_row = {key: original_row_in_chunk.get(key,"") for key in import_fieldnames}
            chunk_to_write.append(filtered_row)

        output_path = os.path.join(import_output_dir, f"compass_import_part{i+1}.csv")
        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=import_fieldnames, extrasaction="drop")
                writer.writeheader()
                writer.writerows(chunk_to_write)
            if logger: logger(f"Exported {len(chunk_to_write)} records to {output_path}")
        except Exception as e:
            if logger: logger(f"[ERROR] Error writing import file {output_path}: {e}")

def process_files(compass_file: str, phone_file: str, mls_files: List[str], output_dir: str, logger=None):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            if logger: logger(f"[ERROR] Could not create output directory {output_dir}: {e}")
            return None, None, []


    extracted_addresses_file = os.path.join(output_dir, "extracted_addresses.csv")
    merged_file = os.path.join(output_dir, "compass_merged.csv")
    import_output_dir = os.path.join(output_dir, "compass_import")

    extracted_addresses = []
    if mls_files:
        if logger: logger("[INFO] Starting address extraction from MLS files...")
        extract_and_save_addresses(mls_files, extracted_addresses_file, logger=logger)
        extracted_addresses = load_extracted_addresses(extracted_addresses_file, logger=logger)
    else:
        if logger: logger("[INFO] No MLS files provided. Skipping address extraction.")
        # Create empty extracted_addresses.csv if it doesn't exist, so subsequent steps don't fail
        if not os.path.exists(extracted_addresses_file):
             extract_and_save_addresses([], extracted_addresses_file, logger=logger)


    if phone_file:
        if logger: logger("[INFO] Starting phone data integration...")
        integrate_phone_into_compass(compass_file, phone_file, merged_file, logger=logger)
    else:
        if logger: logger("[INFO] No Phone file provided. Copying Compass file to merged output.")
        try:
            shutil.copy(compass_file, merged_file)
            # Ensure necessary columns are present in the copied file
            temp_data = load_compass_csv(merged_file, logger=logger)
            if temp_data:
                original_headers = list(temp_data[0].keys())
                extra_cols_to_ensure = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification", "Groups"]
                for row_to_prep in temp_data:
                    for col_to_ensure in extra_cols_to_ensure:
                        row_to_prep.setdefault(col_to_ensure, "")
                    row_to_prep.setdefault("Changes Made", "No changes made.")
                
                final_fieldnames_no_phone = original_headers + [col for col in extra_cols_to_ensure if col not in original_headers]
                with open(merged_file, mode="w", newline="", encoding="utf-8") as f_prep:
                    writer_prep = csv.DictWriter(f_prep, fieldnames=final_fieldnames_no_phone, extrasaction="ignore")
                    writer_prep.writeheader()
                    writer_prep.writerows(temp_data)
            elif logger:
                logger("[WARNING] Copied Compass file seems empty after loading for column prep.")

        except Exception as e:
            if logger: logger(f"[ERROR] Error copying or preparing Compass file as merged output: {e}")
            return None, None, []

    final_data = load_compass_csv(merged_file, logger=logger)
    if not final_data:
        if logger: logger("[ERROR] No data loaded from merged file. Cannot proceed with classifications.")
        # Try to provide paths to whatever was created, even if empty
        fe_path = extracted_addresses_file if os.path.exists(extracted_addresses_file) else None
        mf_path = merged_file if os.path.exists(merged_file) else None
        return fe_path, mf_path, []

    # Get column categories from the final_data which might have been modified
    # This should happen *after* integrate_phone_into_compass or the copy operation
    current_headers = list(final_data[0].keys()) if final_data else []
    if not current_headers and logger:
        logger("[WARNING] final_data is empty or has no headers for categorization.")
    
    compass_cat_map = categorize_columns(current_headers)
    address_columns_for_classification = compass_cat_map.get("address", [])
    email_columns_for_classification = compass_cat_map.get("email", [])

    if logger:
        logger(f"[DEBUG process_files] Address columns for classification: {address_columns_for_classification}")
        logger(f"[DEBUG process_files] Email columns for classification: {email_columns_for_classification}")

    # Classifications
    if email_columns_for_classification:
        classify_agents(final_data, email_columns_for_classification, logger=logger)
        classify_vendors(final_data, email_columns_for_classification, logger=logger)
    elif logger:
        logger("[WARNING] No email columns identified; skipping agent and vendor classification.")
    
    if extracted_addresses:
        classify_clients(final_data, address_columns_for_classification, extracted_addresses, logger=logger)
    elif logger:
        logger("[INFO] No extracted MLS addresses; skipping client classification.")

    update_groups_with_classification(final_data, logger=logger)
    
    # Final save of merged_file with all classifications
    # Get original column order from the source Compass file to maintain it as much as possible
    original_compass_headers = []
    try:
        with open(compass_file, "r", encoding="utf-8-sig") as f_orig_headers:
            reader_orig_headers = csv.reader(f_orig_headers)
            original_compass_headers = next(reader_orig_headers, [])
    except Exception as e_header:
        if logger: logger(f"[WARNING] Could not read original Compass file headers for ordering: {e_header}. Using headers from merged data.")
        if final_data: original_compass_headers = list(final_data[0].keys())


    all_potential_extra_columns = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification", "Groups"]
    
    # Determine the final set of fieldnames: original + any new ones actually present in the data
    final_fieldnames_set = set(original_compass_headers)
    if final_data: # Ensure final_data is not empty
        for row_in_final_data in final_data:
            final_fieldnames_set.update(row_in_final_data.keys()) # Add any keys found in data
    
    # Ensure all_potential_extra_columns are in the set if they might have been added
    final_fieldnames_set.update(all_potential_extra_columns)

    # Create the final ordered list of fieldnames
    # Start with original headers, then add any new ones from the set
    final_ordered_fieldnames = list(original_compass_headers)
    for col in sorted(list(final_fieldnames_set)): # Sort for consistent order of new columns
        if col not in final_ordered_fieldnames:
            final_ordered_fieldnames.append(col)
            
    # Re-map data to ensure all rows have all columns in the final_fieldnames list
    reordered_final_data = []
    if final_data:
        for row_item in final_data:
            new_row_dict = {key: row_item.get(key, "") for key in final_ordered_fieldnames}
            reordered_final_data.append(new_row_dict)
    
    try:
        with open(merged_file, mode="w", newline="", encoding="utf-8") as f_final_write:
            writer_final = csv.DictWriter(f_final_write, fieldnames=final_ordered_fieldnames, extrasaction="drop")
            writer_final.writeheader()
            writer_final.writerows(reordered_final_data)
        if logger: logger(f"Final classified and merged data written to {merged_file}")
    except Exception as e_final_save:
        if logger: logger(f"[ERROR] Error saving final fully merged CSV: {e_final_save}")
        # Return whatever paths we have, even if final save failed
        fe_path = extracted_addresses_file if os.path.exists(extracted_addresses_file) else None
        mf_path = merged_file # it might be partially correct or from a previous step
        return fe_path, mf_path, []

    export_updated_records(merged_file, import_output_dir, logger=logger)

    import_files_paths = []
    if os.path.exists(import_output_dir):
        import_files_paths = [
            os.path.join(import_output_dir, f_name) for f_name in os.listdir(import_output_dir)
            if os.path.isfile(os.path.join(import_output_dir, f_name))
        ]
    
    final_extracted_file_path = extracted_addresses_file if os.path.exists(extracted_addresses_file) and os.path.getsize(extracted_addresses_file) > 0 else None
    final_merged_file_path = merged_file if os.path.exists(merged_file) and os.path.getsize(merged_file) > 0 else None

    return final_extracted_file_path, final_merged_file_path, import_files_paths

