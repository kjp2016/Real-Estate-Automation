import streamlit as st
import os
import csv
import pdfplumber
import pandas as pd
import json
import openai
import re
from typing import List, Dict, Tuple
from io import StringIO, BytesIO
from rapidfuzz import process, fuzz

###############################################################################
# 0) SETUP: OPENAI KEY (from Streamlit secrets) + PAGE TITLE
###############################################################################
st.set_page_config(page_title="Real Estate Automation")
openai.api_key = st.secrets["OPENAI_API_KEY"]

###############################################################################
# 1) ADDRESS EXTRACTION LOGIC (merged from AddressExtract.py, but adapted)
###############################################################################
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


def extract_text_from_pdf(file_bytes: BytesIO) -> str:
    """Extract text using pdfplumber from a PDF in memory."""
    text = ""
    with pdfplumber.open(file_bytes) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return " ".join(text.split())  # normalize spaces


def extract_text_from_csv(file_bytes: BytesIO) -> str:
    """Extract raw text from a CSV in memory."""
    string_data = StringIO(file_bytes.getvalue().decode("utf-8", errors="replace"))
    reader = csv.reader(string_data)
    lines = []
    for row in reader:
        lines.append(" ".join(row))
    return "\n".join(lines).strip()


def extract_text_from_excel(file_bytes: BytesIO) -> str:
    """Extract text from all sheets/columns in an Excel file in memory."""
    xls = pd.ExcelFile(file_bytes)
    collected = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Heuristic: any column that might hold addresses
        for col in df.columns:
            if any(k in col.lower() for k in ["address", "property", "location"]):
                collected.extend(df[col].dropna().astype(str).tolist())
    return "\n".join(collected)


def extract_addresses_with_ai(text: str) -> List[dict]:
    """
    Uses the standard openai.ChatCompletion (>=1.0.0) to parse addresses
    from unstructured text. We instruct GPT to return valid JSON with an
    'addresses' list that follows address_schema.
    """
    if not text.strip():
        return []

    # Incorporate the JSON schema into the system prompt
    system_prompt = f"""You are a data extraction assistant. 
Return valid JSON containing an 'addresses' list that follows this schema:
{json.dumps(address_schema)}

Do NOT include extraneous keys. If an address has a unit, it should be in the
'Unit' field. Output must be valid JSON, with a top-level object containing an
'addresses' array."""
    
    user_prompt = f"""Extract all property addresses from the text below:
{text}

Ensure your output strictly follows the 'addresses' schema above, in valid JSON.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
        )
        # The raw text from GPT
        raw_content = response.choices[0].message["content"]
        # DEBUG: print or st.write the raw output to see what's coming back
        print("RAW GPT OUTPUT:\n", raw_content)

        # Attempt to parse as JSON
        data = json.loads(raw_content)
        return data.get("addresses", [])
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        return []


def extract_addresses_from_mls_files(mls_files) -> pd.DataFrame:
    """
    For each user-uploaded MLS file, parse all text, run GPT-based extraction,
    and aggregate addresses into a DataFrame.
    """
    all_addresses = []

    for file in mls_files:
        filename = file.name
        ext = os.path.splitext(filename)[1].lower()
        st.info(f"Processing MLS file: {filename}")

        if ext == ".pdf":
            text = extract_text_from_pdf(file)
        elif ext == ".csv":
            text = extract_text_from_csv(file)
        elif ext in [".xls", ".xlsx"]:
            text = extract_text_from_excel(file)
        else:
            st.warning(f"Skipping unsupported file format: {filename}")
            continue

        # Send to GPT
        addresses = extract_addresses_with_ai(text)
        all_addresses.extend(addresses)

    if not all_addresses:
        df_empty = pd.DataFrame(columns=["Street Address", "Unit", "City", "State", "Zip Code"])
        return df_empty

    return pd.DataFrame(all_addresses)


###############################################################################
# 2) BROKERAGE + MERGE LOGIC (from stephenrun5.py, but adapted to in-memory)
###############################################################################
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


def is_real_estate_agent(email_list: List[str]) -> bool:
    """Checks if any email belongs to a known brokerage domain."""
    for email in email_list:
        match = re.search(r"@([\w.-]+)$", email)
        if match:
            domain = match.group(1).lower()
            if domain in BROKERAGE_EMAIL_DOMAINS:
                return True
    return False


def is_vendor(email_list: List[str]) -> bool:
    """Checks if any email domain has known vendor keywords."""
    for email in email_list:
        match = re.search(r"@([\w.-]+)$", email)
        if match:
            domain = match.group(1).lower()
            if any(k in domain for k in VENDOR_KEYWORDS):
                return True
    return False


def load_csv_in_memory(file: BytesIO) -> List[Dict[str, str]]:
    """Reads a CSV (BytesIO) into a list of dict rows, just like csv.DictReader."""
    text_data = file.getvalue().decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(StringIO(text_data))
    return list(reader)


def save_csv_in_memory(rows: List[dict], fieldnames: List[str]) -> BytesIO:
    """Takes a list of dictionaries + fieldnames -> CSV (in-memory)."""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return BytesIO(output.getvalue().encode("utf-8"))


###############################################################################
# 3) Fuzzy Merge + Classification Logic
###############################################################################
def fuzzy_name_merge_compass_phone(compass_data: List[dict], phone_data: List[dict]) -> List[dict]:
    """
    Example implementing some of the fuzzy name matching logic from stephenrun5.
    Merges phone data into the compass data (by the best fuzzy match).
    """
    if not compass_data:
        return []

    # We'll detect columns that look like 'First Name' or 'Last Name'
    # For simplicity, you can reuse your "categorize_columns" approach or do a simpler heuristic:
    compass_headers = list(compass_data[0].keys())
    phone_headers = list(phone_data[0].keys()) if phone_data else []

    # Minimal approach: look for columns containing "first" or "last"
    def guess_columns(headers: List[str], name_type: str) -> List[str]:
        result = []
        for h in headers:
            if name_type in ("first", "fname"):
                if "first" in h.lower() or "fname" in h.lower():
                    result.append(h)
            elif name_type in ("last", "lname"):
                if "last" in h.lower() or "lname" in h.lower():
                    result.append(h)
            elif name_type == "email":
                if "email" in h.lower():
                    result.append(h)
        return result

    c_first = guess_columns(compass_headers, "first")
    c_last = guess_columns(compass_headers, "last")
    c_email = guess_columns(compass_headers, "email")

    p_first = guess_columns(phone_headers, "first")
    p_last = guess_columns(phone_headers, "last")
    p_email = guess_columns(phone_headers, "email")

    def get_name(row: dict, first_cols: List[str], last_cols: List[str]) -> str:
        f = ""
        l = ""
        for c in first_cols:
            if row.get(c, "").strip():
                f = row[c].strip().lower()
                break
        for c in last_cols:
            if row.get(c, "").strip():
                l = row[c].strip().lower()
                break
        return f"{f} {l}".strip()

    # Build compass name dictionary
    compass_names = []
    for i, row in enumerate(compass_data):
        nm = get_name(row, c_first, c_last)
        compass_names.append((i, nm))

    # We'll use an array for final output
    updated_compass = list(compass_data)
    appended = []

    # Prepare a list of just the name strings for fuzzy matching
    all_names = [x[1] for x in compass_names]

    for phone_row in phone_data:
        phone_name = get_name(phone_row, p_first, p_last)
        if not phone_name:  # no name => just append as new row
            new_row = dict.fromkeys(compass_headers, "")
            new_row["Changes Made"] = "Created new contact (no name in phone)."
            appended.append(new_row)
            continue

        # Fuzzy match
        best = process.extractOne(phone_name, all_names, scorer=fuzz.WRatio)
        if best and best[1] >= 90:  # match threshold
            best_name = best[0]
            # Find the index in compass_names
            match_idx = all_names.index(best_name)
            comp_idx = compass_names[match_idx][0]
            # Merge emails from phone row into compass row
            changes = []
            # get phone emails
            phone_emails = []
            for col in p_email:
                val = phone_row.get(col, "").strip().lower()
                if val:
                    phone_emails.append(val)

            # get existing compass emails
            existing_emails = []
            for col in c_email:
                val = updated_compass[comp_idx].get(col, "").strip().lower()
                if val:
                    existing_emails.append(val)

            for em in phone_emails:
                if em not in existing_emails:
                    # fill first empty email col
                    for col in c_email:
                        if not updated_compass[comp_idx].get(col, "").strip():
                            updated_compass[comp_idx][col] = em
                            changes.append(f"Email->{col}: {em}")
                            existing_emails.append(em)
                            break

            # If any email is a real estate domain => Category=Agent
            if is_real_estate_agent(phone_emails + existing_emails):
                updated_compass[comp_idx]["Category"] = "Agent"
                changes.append("Category=Agent")

            if changes:
                updated_compass[comp_idx]["Changes Made"] = (
                    updated_compass[comp_idx].get("Changes Made", "") + 
                    f" FuzzyMatch {best[1]}%: " + "; ".join(changes)
                )
        else:
            # No fuzzy match => new row
            new_row = dict.fromkeys(compass_headers, "")
            # copy first/last from phone
            for col in p_first:
                new_row["First Name"] = phone_row[col].strip()
                break
            for col in p_last:
                new_row["Last Name"] = phone_row[col].strip()
                break
            # check if agent
            phone_emails = []
            for col in p_email:
                val = phone_row.get(col, "").strip().lower()
                if val:
                    phone_emails.append(val)

            if is_real_estate_agent(phone_emails):
                new_row["Category"] = "Agent"

            new_row["Changes Made"] = "No match in compass. Created new contact."
            appended.append(new_row)

    return updated_compass + appended


def classify_vendors_inplace(final_data: List[dict]):
    """Mark row["Vendor Classification"] = "Vendor" if domain has vendor keywords."""
    for row in final_data:
        # gather all emails
        row_emails = []
        for k, v in row.items():
            if "email" in k.lower() and v.strip():
                row_emails.append(v.strip().lower())
        if is_vendor(row_emails):
            row["Vendor Classification"] = "Vendor"
            row["Changes Made"] = row.get("Changes Made", "") + " | Classified as Vendor"


def classify_clients_inplace(final_data: List[dict], addresses_df: pd.DataFrame):
    """
    For each contact, if we can build an address string that fuzzy-matches
    anything in addresses_df => row["Client Classification"] = "Client".
    We'll guess the columns for address from the row's keys.
    """
    # Make a single list of all extracted addresses (concatenated as strings)
    extracted_list = []
    for idx, row_addr in addresses_df.iterrows():
        combo = f"{row_addr['Street Address']} {row_addr.get('Unit','')} {row_addr['City']} {row_addr['State']} {row_addr['Zip Code']}"
        extracted_list.append(" ".join(combo.split()))

    # Identify columns that might contain address data
    possible_address_cols = []
    if final_data:
        for col in final_data[0].keys():
            c_lower = col.lower()
            if any(k in c_lower for k in ["address", "street", "city", "zip", "state"]):
                possible_address_cols.append(col)

    for row in final_data:
        # build a single string
        parts = []
        for col in possible_address_cols:
            val = row.get(col, "").strip()
            if val:
                parts.append(val)
        my_address = " ".join(parts).lower()
        if not my_address:
            continue
        best = process.extractOne(my_address, extracted_list, scorer=fuzz.WRatio)
        if best and best[1] >= 90:
            row["Client Classification"] = "Client"
            row["Changes Made"] = row.get("Changes Made", "") + f" | Address matched {best[1]}% => Client"


def classify_agents_inplace(final_data: List[dict]):
    """
    If any email belongs to brokerage => Category=Agent.
    (You can also keep 'Non-Agent' if no match, as in your original code.)
    """
    for row in final_data:
        # gather all emails
        row_emails = []
        for k, v in row.items():
            if "email" in k.lower() and v.strip():
                row_emails.append(v.strip().lower())
        if is_real_estate_agent(row_emails):
            row["Category"] = "Agent"
        else:
            row.setdefault("Category", "Non-Agent")


###############################################################################
# 4) STREAMLIT APP
###############################################################################
def main():
    st.title("Real Estate Address & Contact Automation")

    st.header("1) Upload Files")
    st.subheader("Compass Export (exactly 1 CSV)")
    compass_file = st.file_uploader("Compass CSV", type=["csv"], accept_multiple_files=False)

    st.subheader("Phone Export(s) (one or more CSVs)")
    phone_files = st.file_uploader("Phone CSV Files", type=["csv"], accept_multiple_files=True)

    st.subheader("MLS Data (one or more) for Address Extraction (PDF, CSV, XLS, XLSX)")
    mls_files = st.file_uploader("MLS Data", 
                                 type=["pdf","csv","xls","xlsx"],
                                 accept_multiple_files=True)

    if st.button("Run Automation"):
        logs = []
        if not compass_file:
            st.error("Please upload at least one Compass export CSV.")
            return

        # -----------------------------
        # (A) Extract addresses from MLS
        # -----------------------------
        if mls_files:
            st.write("### Extracting addresses from MLS data...")
            addresses_df = extract_addresses_from_mls_files(mls_files)
            if addresses_df.empty:
                st.warning("No addresses were extracted.")
            else:
                st.success(f"Extracted {len(addresses_df)} addresses from MLS files.")
        else:
            addresses_df = pd.DataFrame()
            st.warning("No MLS files provided; skipping address extraction.")
        
        # Let user see or download extracted MLS addresses
        if not addresses_df.empty:
            st.subheader("Extracted MLS Addresses")
            st.dataframe(addresses_df)
            # Offer CSV download
            extracted_csv = save_csv_in_memory(
                addresses_df.to_dict("records"),
                ["Street Address","Unit","City","State","Zip Code"]
            )
            st.download_button(
                "Download extracted_addresses.csv",
                data=extracted_csv,
                file_name="extracted_addresses.csv",
                mime="text/csv"
            )

        # -----------------------------
        # (B) Merge Compass + Phone(s)
        # -----------------------------
        st.write("### Merging Compass + Phone data...")
        compass_data = load_csv_in_memory(compass_file)
        all_phones_merged = compass_data  # start with compass

        if phone_files:
            for ph in phone_files:
                phone_data = load_csv_in_memory(ph)
                # fuzzy merge
                all_phones_merged = fuzzy_name_merge_compass_phone(all_phones_merged, phone_data)
            st.success(f"Compass data size after phone merges: {len(all_phones_merged)}")
        else:
            st.info("No phone files provided; skipping phone merges.")

        # Ensure "Changes Made" and other columns exist
        for row in all_phones_merged:
            row.setdefault("Changes Made", "")

        # -----------------------------
        # (C) Classify Agents, Vendors, Clients
        # -----------------------------
        st.write("### Classifying Agents, Vendors, and Clients...")

        # 1) Agent classification by brokerage domain
        classify_agents_inplace(all_phones_merged)
        # 2) Vendor classification by vendor keywords
        classify_vendors_inplace(all_phones_merged)
        # 3) Client classification by address match
        if not addresses_df.empty:
            classify_clients_inplace(all_phones_merged, addresses_df)

        # Convert to DataFrame for display
        if all_phones_merged:
            final_df = pd.DataFrame(all_phones_merged)
            st.subheader("Final Merged + Classified Data")
            st.dataframe(final_df)
        else:
            st.error("No final data to display!")
            return

        # Prepare CSV for download
        final_fieldnames = list(final_df.columns)
        final_csv_bytes = save_csv_in_memory(all_phones_merged, final_fieldnames)
        st.download_button(
            "Download Final compass_merged.csv",
            data=final_csv_bytes,
            file_name="compass_merged.csv",
            mime="text/csv"
        )
        st.success("All done!")


if __name__ == "__main__":
    main()
