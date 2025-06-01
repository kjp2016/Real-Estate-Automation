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
except (AttributeError, KeyError):  # Handle local testing without Streamlit secrets
    openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_DEFAULT_KEY_IF_ANY"))


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


def extract_addresses_with_ai_chunked(text: str, max_lines: int = 50, logger=None) -> List[dict]:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        if logger:
            logger("[DEBUG] Sending full text to OpenAI")
        return extract_addresses_with_ai(text)
    else:
        if logger:
            logger(f"[DEBUG] Splitting text into {len(lines) // max_lines + 1} chunks")
        addresses = []
        for i in range(0, len(lines), max_lines):
            chunk = "\n".join(lines[i:i+max_lines])
            if logger:
                logger(f"[DEBUG] Sending chunk {i//max_lines + 1} to OpenAI")
            result = extract_addresses_with_ai(chunk)
            addresses.extend(result)
        return addresses


def extract_addresses_row_by_row(df: pd.DataFrame, logger=None) -> List[dict]:
    extracted_addresses = []
    for idx, row in df.iterrows():
        row_text = []
        for col in df.columns:
            val = str(row[col]).strip()
            if val and val.lower() != "nan":
                row_text.append(f"{col}: {val}")
        if row_text:
            prompt_text = "\n".join(row_text)
            if logger:
                logger(f"[DEBUG] Extracting row {idx + 1}...")
            try:
                result = extract_addresses_with_ai(prompt_text)
                extracted_addresses.extend(result)
            except Exception as e:
                if logger:
                    logger(f"[ERROR] Row {idx + 1} extraction failed: {e}")
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
        print(f"Error processing PDF {pdf_path}: {e}")
    return " ".join(text.split())


def extract_text_from_csv(file_path: str) -> str:
    text = ""
    try:
        df = pd.read_csv(file_path)
        # We'll format each row as a structured record
        for _, row in df.iterrows():
            row_text = []
            for col in df.columns:
                val = str(row[col]).strip()
                if val and val.lower() != 'nan':
                    row_text.append(f"{col}: {val}")
            text += "\n".join(row_text) + "\n---\n"
    except Exception as e:
        print(f"Error reading CSV {file_path}: {e}")
    return text.strip()


def extract_addresses_from_excel(file_path: str, logger=None) -> List[dict]:
    extracted_addresses = []
    try:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            if logger:
                logger(f"[DEBUG] Processing sheet: {sheet_name} with {len(df)} rows")
            sheet_addresses = extract_addresses_row_by_row(df, logger=logger)
            extracted_addresses.extend(sheet_addresses)
    except Exception as e:
        if logger:
            logger(f"Error processing Excel {os.path.basename(file_path)}: {e}")
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
            addresses = extract_addresses_with_ai_chunked(text, max_lines=50, logger=logger)
        elif ext == ".csv":
            if logger:
                logger(f"Processing CSV for addresses: {file_name}")
            try:
                df = pd.read_csv(file_path)
                addresses = extract_addresses_row_by_row(df, logger=logger)
            except Exception as e:
                if logger:
                    logger(f"[ERROR] Failed to process CSV: {e}")
                addresses = []
        elif ext in [".xls", ".xlsx"]:
            if logger:
                logger(f"Processing Excel for addresses: {file_name}")
            addresses = extract_addresses_from_excel(file_path, logger=logger)
        else:
            if logger:
                logger(f"Unsupported file type for address extraction: {file_name}")
            continue
        all_extracted_addresses.extend(addresses)

    fieldnames = ["Street Address", "Unit", "City", "State", "Zip Code", "Home Anniversary Date"]

    try:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for address in all_extracted_addresses:
                writer.writerow({
                    "Street Address": address.get("Street Address", ""),
                    "Unit": address.get("Unit", ""),
                    "City": address.get("City", ""),
                    "State": address.get("State", ""),
                    "Zip Code": address.get("Zip Code", ""),
                    "Home Anniversary Date": address.get("Home Anniversary Date", "")
                })
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

VENDOR_KEYWORDS = ["title", "mortgage", "lending", "escrow"]  # This remains unchanged

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
        elif "phone" in h_lower:
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

    phone_addr_groups = {}
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

    for phone_group, phone_addr in phone_addr_groups.items():
        already_present = False
        for group_id, comp_dict in compass_groups.items():
            comp_street = compass_row.get(comp_dict.get("street", ""), "").strip().lower()
            if "street" in phone_addr and comp_street and phone_addr["street"].lower() == comp_street:
                already_present = True
                break
        if already_present:
            continue

        target_group = None
        for group_id, comp_dict in compass_groups.items():
            group_empty = all(not compass_row.get(col, "").strip() for col in comp_dict.values())
            if group_empty:
                target_group = group_id
                break
        if target_group is None:
            continue

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

    phone_emails = [
        phone_row.get(col, "").strip().lower()
        for col in phone_cat_map["email"]
        if phone_row.get(col, "").strip()
    ]
    existing_emails = {
        compass_row.get(col, "").strip().lower()
        for col in compass_cat_map["email"]
        if compass_row.get(col, "").strip()
    }
    for email in phone_emails:
        if email not in existing_emails:
            for col in compass_cat_map["email"]:
                if not compass_row.get(col, "").strip():
                    compass_row[col] = email
                    changes.append(f"Email->{col}: {email}")
                    existing_emails.add(email)
                    break

    phone_numbers = [
        phone_row.get(col, "").strip()
        for col in phone_cat_map["phone"]
        if phone_row.get(col, "").strip()
    ]
    normalized_phone_numbers = [normalize_phone(p) for p in phone_numbers if p]
    existing_phones = {
        normalize_phone(compass_row.get(col, "").strip())
        for col in compass_cat_map["phone"]
        if compass_row.get(col, "").strip()
    }
    for original_phone, norm_phone in zip(phone_numbers, normalized_phone_numbers):
        if norm_phone not in existing_phones:
            for col in compass_cat_map["phone"]:
                if not compass_row.get(col, "").strip():
                    compass_row[col] = original_phone
                    changes.append(f"Phone->{col}: {original_phone}")
                    existing_phones.add(norm_phone)
                    break

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
        # Preserve existing category if it's already "Agent", otherwise classify
        if row.get("Category", "").strip().lower() != "agent":
            row["Category"] = "Agent" if is_real_estate_agent(list(all_emails)) else "Non-Agent"


def classify_vendors(final_data: List[Dict[str, str]], email_columns: List[str]):
    for row in final_data:
        all_emails = set()
        for col in email_columns:
            email = row.get(col, "").strip().lower()
            if email:
                all_emails.add(email)
        if is_vendor(list(all_emails)):
            # Only update if not already classified as Vendor to avoid duplicate entries
            if row.get("Vendor Classification", "") != "Vendor":
                row["Vendor Classification"] = "Vendor"
                current_changes = row.get("Changes Made", "")
                change_msg = "Classified as Vendor"
                row["Changes Made"] = (
                    f"{current_changes} | {change_msg}"
                    if current_changes and current_changes.strip().lower() != "no changes made."
                    else change_msg
                )


def integrate_phone_into_compass(compass_file: str, phone_file: str, output_file: str, logger=None):
    compass_data = load_compass_csv(compass_file)
    phone_data = load_phone_csv(phone_file)

    if not compass_data:
        if logger:
            logger("Compass CSV has no data.")
        # Create an empty output file with headers if compass_file was empty but phone_file was not
        if phone_data and os.path.exists(compass_file):
            with open(compass_file, "r", encoding="utf-8-sig") as cf, open(output_file, "w", newline="", encoding="utf-8") as of:
                reader = csv.reader(cf)
                original_order = next(reader, [])
                extra_columns = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification"]
                final_fieldnames = original_order + [col for col in extra_columns if col not in original_order]
                writer = csv.DictWriter(of, fieldnames=final_fieldnames)
                writer.writeheader()
        return

    if not phone_data:
        if logger:
            logger("Phone CSV has no data. Proceeding without phone integration.")
        final_data = compass_data
        compass_headers = list(compass_data[0].keys())
        extra_columns_to_add = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification"]
        for row in final_data:
            for col in extra_columns_to_add:
                row.setdefault(col, "")
            row.setdefault("Changes Made", "No changes made.")

        final_fieldnames = compass_headers + [col for col in extra_columns_to_add if col not in compass_headers]
        try:
            with open(output_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=final_fieldnames)
                writer.writeheader()
                writer.writerows(final_data)
            if logger:
                logger(f"Copied Compass data to {output_file} as no phone data was provided.")
        except Exception as e:
            if logger:
                logger(f"Error writing initial merged CSV (no phone data): {e}")
        return  # End early as no phone data to merge

    compass_cat_map = categorize_columns(list(compass_data[0].keys()))
    phone_cat_map = categorize_columns(list(phone_data[0].keys()))

    compass_keys = [build_name_key(row, compass_cat_map) for row in compass_data]
    updated_compass = [row.copy() for row in compass_data]

    for pcontact in phone_data:
        phone_name_key = build_name_key(pcontact, phone_cat_map)
        if not phone_name_key.strip():
            if logger:
                logger("Skipping phone contact with no name.")
            continue

        best_key, score = fuzzy_name_match(phone_name_key, compass_keys, threshold=97)
        if best_key:
            try:
                match_idx = compass_keys.index(best_key)
                matched_row_to_update = updated_compass[match_idx]

                merged_row, changes = merge_phone_into_compass(
                    matched_row_to_update.copy(), pcontact, compass_cat_map, phone_cat_map
                )
                merged_row, addr_changes = merge_address_into_compass(
                    merged_row, pcontact, compass_cat_map, phone_cat_map
                )

                all_changes = changes + addr_changes
                current_changes_made = merged_row.get("Changes Made", "")
                if not current_changes_made or current_changes_made.strip() == "No changes made.":
                    current_changes_made = ""

                if all_changes:
                    change_log_message = f"Matched {score}% | {'; '.join(all_changes)}"
                    merged_row["Changes Made"] = (
                        f"{current_changes_made} | {change_log_message}"
                        if current_changes_made
                        else change_log_message
                    )
                elif not current_changes_made:
                    merged_row["Changes Made"] = "No changes made."

                updated_compass[match_idx] = merged_row

            except ValueError:
                if logger:
                    logger(f"Error: Matched key '{best_key}' not found in compass_keys list.")
        else:
            if logger:
                logger(f"Phone contact '{phone_name_key}' not found in Compass export; skipping.")

    final_data = updated_compass
    original_headers = list(compass_data[0].keys())

    # Define all columns that might be added or used.
    extra_columns = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification"]

    # Ensure all rows have these columns, defaulting to empty string or "No changes made."
    for row in final_data:
        for col in extra_columns:
            if col == "Changes Made":
                row.setdefault(col, "No changes made.")
            else:
                row.setdefault(col, "")

    # Re-categorize emails after phone integration
    if final_data:
        final_headers = list(final_data[0].keys())
        final_cat_map = categorize_columns(final_headers)
        email_columns = final_cat_map["email"]
        if email_columns:
            classify_agents(final_data, email_columns)
        elif logger:
            logger("No email columns found in final_data for agent classification.")

    fieldnames = original_headers + [col for col in extra_columns if col not in original_headers]

    try:
        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final_data)
        if logger:
            logger(f"Merged data written to {output_file}.")
    except Exception as e:
        if logger:
            logger(f"Error writing merged CSV: {e}")


def load_extracted_addresses(address_file: str) -> List[dict]:
    extracted_records = []
    try:
        with open(address_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                extracted_records.append(row)
    except FileNotFoundError:
        # If the file doesn't exist, return empty list
        print(f"Info: Extracted addresses file not found at {address_file}. Returning empty list.")
    return extracted_records


def extract_street_addresses_from_row(row: Dict[str, str]) -> List[str]:
    """Extract values from a row whose header contains 'street' (ignoring case)."""
    street_addresses = []
    for key, value in row.items():
        if value and "street" in key.lower():
            street_addresses.append(value.strip())
    return street_addresses


def fuzzy_address_match(address: str, extracted_addresses: List[str], threshold: int = 90) -> bool:
    """Use fuzzy matching to check if a given street address matches any extracted street addresses."""
    if not address or not extracted_addresses:
        return False
    best_match = process.extractOne(address, extracted_addresses, scorer=fuzz.WRatio)
    return best_match and best_match[1] >= threshold


def extract_address_from_row(row: Dict[str, str], address_columns: List[str]) -> str:
    parts = [row.get(col, "").strip() for col in address_columns if row.get(col)]
    return " ".join(parts)


def classify_clients_simplified(final_data: List[Dict[str, str]],
                                extracted_addresses: List[dict],
                                logger=None):
    if logger:
        logger("[INFO classify_clients_simplified] Starting simplified client HAD update.")

    if not extracted_addresses:
        if logger:
            logger("[DEBUG classify_clients_simplified] Extracted_addresses list is empty. No HAD update possible.")
        return

    # Build a lookup from (lowercase street) → HAD
    mls_address_to_had_lookup = {}
    for mls_record in extracted_addresses:
        street = str(mls_record.get("Street Address", "")).strip().lower()
        had = mls_record.get("Home Anniversary Date", "")
        if street and had:
            if street not in mls_address_to_had_lookup:
                mls_address_to_had_lookup[street] = had

    if not mls_address_to_had_lookup:
        if logger:
            logger("[DEBUG classify_clients_simplified] MLS lookup empty. No HAD update possible.")
        return

    if logger:
        logger(f"[DEBUG classify_clients_simplified] Created MLS lookup with {len(mls_address_to_had_lookup)} entries.")

    successful_updates = 0

    for compass_row_idx, compass_row in enumerate(final_data):
        # Skip if HAD already present
        if compass_row.get("Home Anniversary Date", "").strip():
            continue

        found_match = False

        # Potential street columns in Compass
        potential_compass_street_cols = []
        if "Address Line 1" in compass_row:
            potential_compass_street_cols.append("Address Line 1")
        for i in range(1, 7):
            potential_compass_street_cols.append(f"Address {i} Line 1")
        if "Street Address" in compass_row:
            potential_compass_street_cols.append("Street Address")
        potential_compass_street_cols = list(dict.fromkeys(potential_compass_street_cols))

        for address_col_name in potential_compass_street_cols:
            compass_street_line = str(compass_row.get(address_col_name, "")).strip().lower()
            if not compass_street_line:
                continue

            # Direct lookup
            if compass_street_line in mls_address_to_had_lookup:
                had_to_set = mls_address_to_had_lookup[compass_street_line]
                compass_row["Home Anniversary Date"] = had_to_set
                compass_row["Client Classification"] = "Past client"

                current_changes = compass_row.get("Changes Made", "")
                change_msg = f"HAD set from direct match on '{address_col_name}'"
                compass_row["Changes Made"] = (
                    f"{current_changes} | {change_msg}"
                    if current_changes and current_changes.strip().lower() != "no changes made."
                    else change_msg
                )

                successful_updates += 1
                if logger:
                    logger(f"[DEBUG] Direct match for {compass_row.get('First Name')} {compass_row.get('Last Name')} on '{compass_street_line}'. Set HAD: {had_to_set}")
                found_match = True
                break

            # Fuzzy lookup
            if not found_match:
                mls_street_keys_list = list(mls_address_to_had_lookup.keys())
                if not mls_street_keys_list:
                    continue

                best_match_tuple = process.extractOne(compass_street_line, mls_street_keys_list, scorer=fuzz.WRatio)
                if best_match_tuple and best_match_tuple[1] >= 93:
                    matched_mls_street_key = best_match_tuple[0]
                    had_to_set = mls_address_to_had_lookup[matched_mls_street_key]

                    compass_row["Home Anniversary Date"] = had_to_set
                    compass_row["Client Classification"] = "Client"

                    current_changes = compass_row.get("Changes Made", "")
                    change_msg = (
                        f"HAD set from fuzzy match (Score: {best_match_tuple[1]}%) on "
                        f"'{address_col_name}' (Compass: '{compass_street_line}' vs MLS: '{matched_mls_street_key}')"
                    )
                    compass_row["Changes Made"] = (
                        f"{current_changes} | {change_msg}"
                        if current_changes and current_changes.strip().lower() != "no changes made."
                        else change_msg
                    )

                    successful_updates += 1
                    if logger:
                        logger(f"[DEBUG] Fuzzy match for {compass_row.get('First Name')} "
                               f"{compass_row.get('Last Name')} on Compass '{compass_street_line}' "
                               f"to MLS '{matched_mls_street_key}' (Score: {best_match_tuple[1]}). Set HAD: {had_to_set}")
                    found_match = True
                    break

        # No match → skip
        # if logger and not found_match:
        #     logger(f"[DEBUG] No HAD match found for {compass_row.get('First Name')} {compass_row.get('Last Name')}")

    if logger:
        logger(f"[INFO classify_clients_simplified] Finished. {successful_updates} records updated.")


def update_groups_with_classification(final_data: List[Dict[str, str]]) -> None:
    """
    Updates the 'Groups' field based on classifications:
      • If Category=='Agent', append 'Agents'.
      • If Vendor Classification=='Vendor', append 'Vendors'.
      • If Client Classification=='Past client', append 'Past Clients'.
    Any such additions get tacked onto 'Changes Made' as well.
    """
    for row in final_data:
        groups_str = row.get("Groups", "").strip()
        current_groups_set = {g.strip().lower() for g in groups_str.split(",") if g.strip()}

        changes_for_groups = []

        if row.get("Category", "").strip().lower() == "agent":
            if "agents" not in current_groups_set:
                current_groups_set.add("agents")
                changes_for_groups.append("Added Agents to Groups")

        if row.get("Vendor Classification", "").strip().lower() == "vendor":
            if "vendors" not in current_groups_set:
                current_groups_set.add("vendors")
                changes_for_groups.append("Added Vendors to Groups")

        if row.get("Client Classification", "").strip().lower() == "past client":
            if "past clients" not in current_groups_set:
                current_groups_set.add("past clients")
                changes_for_groups.append("Added Past Clients to Groups")

        final_groups_list = sorted([g.title() for g in current_groups_set if g])
        row["Groups"] = ",".join(final_groups_list)

        if changes_for_groups:
            current_changes = row.get("Changes Made", "").strip()
            group_change_log = "; ".join(changes_for_groups)
            if current_changes and current_changes.strip().lower() != "no changes made.":
                row["Changes Made"] = f"{current_changes} | {group_change_log}"
            else:
                row["Changes Made"] = group_change_log


def export_updated_records(merged_file: str, import_output_dir: str, logger=None):
    """
    Reads the merged CSV, finds any row whose 'Changes Made' is not empty and not 'No changes made.',
    then writes those rows—minus the excluded columns—into one or more csvs of at most 2000 rows each.
    """
    exclude_cols = {
        "Created At", "Last Contacted", "Changes Made", "Category",
        "Agent Classification", "Client Classification", "Vendor Classification"
        # Note: "Home Anniversary Date" stays in the import file
    }

    try:
        with open(merged_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            original_order = reader.fieldnames or []
    except FileNotFoundError:
        if logger:
            logger(f"[ERROR export] Merged file {merged_file} not found.")
        return
    except Exception as e:
        if logger:
            logger(f"[ERROR export] Could not read {merged_file}: {e}")
        return

    if not rows:
        if logger:
            logger("[INFO export] No data in merged file to export.")
        return

    # Collect all rows whose 'Changes Made' is neither blank nor exactly 'No changes made.'
    updated_rows = []
    for row in rows:
        raw = row.get("Changes Made", "")
        if not raw:
            continue
        stripped = raw.strip()
        if stripped and stripped.lower() != "no changes made.":
            updated_rows.append(row)

    if logger:
        logger(f"[DEBUG export] total rows = {len(rows)}, updated rows = {len(updated_rows)}")

    if not updated_rows:
        if logger:
            logger("[INFO export] No updated records found. Nothing to export for Compass import.")
        return

    import_fieldnames = [col for col in original_order if col not in exclude_cols]

    # Split into chunks of up to 2000 rows each
    chunk_size = 2000
    total_chunks = (len(updated_rows) + chunk_size - 1) // chunk_size

    if not os.path.exists(import_output_dir):
        os.makedirs(import_output_dir)

    for i in range(total_chunks):
        chunk = updated_rows[i * chunk_size : (i + 1) * chunk_size]
        chunk_to_write = [
            {key: row_original.get(key, "") for key in import_fieldnames}
            for row_original in chunk
        ]

        output_path = os.path.join(import_output_dir, f"compass_import_part{i+1}.csv")
        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=import_fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(chunk_to_write)
            if logger:
                logger(f"[INFO export] Exported {len(chunk)} record(s) to {output_path}")
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
        if logger:
            logger("Starting address extraction from MLS files...")
        extract_and_save_addresses(mls_files, extracted_addresses_file, logger=logger)
        extracted_addresses = load_extracted_addresses(extracted_addresses_file)
        if logger:
            logger(f"Extracted {len(extracted_addresses)} addresses.")
    else:
        if logger:
            logger("No MLS files provided. Skipping address extraction.")
        if os.path.exists(extracted_addresses_file):
            os.remove(extracted_addresses_file)

    if phone_file:
        if logger:
            logger("Integrating phone data into Compass export...")
        integrate_phone_into_compass(compass_file, phone_file, merged_file, logger=logger)
        if logger:
            logger("Phone integration completed.")
    else:
        import shutil
        try:
            shutil.copy(compass_file, merged_file)
            if logger:
                logger("No Phone file provided; using Compass file as initial merged output.")
            temp_data = load_compass_csv(merged_file)
            if temp_data:
                original_headers = list(temp_data[0].keys())
                extra_cols_ensure = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification"]
                for row in temp_data:
                    for col in extra_cols_ensure:
                        row.setdefault(col, "")
                    row.setdefault("Changes Made", "No changes made.")

                final_fieldnames_ensure = original_headers + [
                    col for col in extra_cols_ensure if col not in original_headers
                ]
                with open(merged_file, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=final_fieldnames_ensure, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(temp_data)
        except Exception as e:
            if logger:
                logger(f"Error copying or preparing compass file as merged output: {e}")
            return None, None, []

    final_data = load_compass_csv(merged_file)
    if not final_data:
        if logger:
            logger("No merged data found after phone integration or copy step!")
        if not os.path.exists(merged_file) or os.path.getsize(merged_file) == 0:
            try:
                with open(compass_file, "r", encoding="utf-8-sig") as cf_headers, \
                     open(merged_file, "w", newline="", encoding="utf-8") as mf_empty:
                    reader = csv.reader(cf_headers)
                    original_order_empty = next(reader, [])
                    extra_columns_empty = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification"]
                    final_fieldnames_empty = original_order_empty + [
                        col for col in extra_columns_empty if col not in original_order_empty
                    ]
                    writer_empty = csv.DictWriter(mf_empty, fieldnames=final_fieldnames_empty)
                    writer_empty.writeheader()
                if logger:
                    logger("Created an empty merged file with headers as final_data was empty.")
            except Exception as e_empty:
                if logger:
                    logger(f"Could not create empty merged file: {e_empty}")

        return extracted_addresses_file if mls_files else None, merged_file, []

    current_headers = list(final_data[0].keys())
    compass_cat_map = categorize_columns(current_headers)
    email_columns = compass_cat_map["email"]

    if email_columns:
        classify_agents(final_data, email_columns)
    elif logger:
        logger("No email columns for agent classification.")

    if extracted_addresses:
        classify_clients_simplified(final_data, extracted_addresses, logger=logger)
    else:
        if logger:
            logger("Skipping client classification as no MLS addresses were extracted/loaded.")

    if email_columns:
        classify_vendors(final_data, email_columns)
    elif logger:
        logger("No email columns for vendor classification.")

    update_groups_with_classification(final_data)

    try:
        with open(compass_file, "r", encoding="utf-8-sig") as f_orig:
            reader = csv.reader(f_orig)
            original_order = next(reader, [])
            if not original_order and final_data:
                original_order = list(final_data[0].keys())
            elif not original_order:
                original_order = ["First Name", "Last Name", "Email 1"]
    except Exception as e:
        if logger:
            logger(f"Error reading original Compass file header for column ordering: {e}")
        original_order = list(final_data[0].keys()) if final_data else ["First Name", "Last Name", "Email 1"]

    all_potential_extra_columns = ["Category", "Changes Made", "Home Anniversary Date", "Vendor Classification", "Client Classification", "Groups"]
    actual_extra_columns_in_data = [
        col for col in all_potential_extra_columns
        if col in final_data[0].keys() and col not in original_order
    ]

    final_fieldnames = list(dict.fromkeys(original_order + actual_extra_columns_in_data))

    reordered_data = []
    for row in final_data:
        new_row = {key: row.get(key, "") for key in final_fieldnames}
        reordered_data.append(new_row)

    try:
        with open(merged_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(reordered_data)
        if logger:
            logger(f"Final merged data with classifications written to {merged_file}")
    except Exception as e:
        if logger:
            logger(f"Error saving final merged CSV: {e}")
        return (extracted_addresses_file if mls_files else None), merged_file, []

    # Now export any updated rows
    export_updated_records(merged_file, import_output_dir, logger=logger)

    import_files = []
    if os.path.exists(import_output_dir):
        for fname in os.listdir(import_output_dir):
            full_path = os.path.join(import_output_dir, fname)
            if os.path.isfile(full_path):
                import_files.append(full_path)

    final_extracted_file = extracted_addresses_file if (mls_files and os.path.exists(extracted_addresses_file)) else None
    final_merged_file = merged_file if os.path.exists(merged_file) else None

    return final_extracted_file, final_merged_file, import_files
