import streamlit as st
import os
import tempfile
import io
import csv

from realestate_processor import process_files

def main():
    st.title("Real Estate Data Processor (Streamlit)")

    logs = []
    def logger(message: str):
        logs.append(message)

    st.markdown("""
    This Streamlit app lets you:
    1. Select a Compass Export CSV (required).
    2. Optionally, select a Phone Export CSV.
    3. Optionally, select one or more MLS/Sales Activity files (PDF, CSV, Excel).
    4. Run the data extraction, merging, and classification flow.
    5. Download your processed files.
    """)

    # Ensure session state variables are initialized
    if "import_files" not in st.session_state:
        st.session_state["import_files"] = []

    # FILE UPLOADS
    compass_file = st.file_uploader("Upload Compass Export CSV", type=["csv"])
    phone_file = st.file_uploader("Upload Phone Export CSV (optional)", type=["csv"])
    mls_files = st.file_uploader(
        "Upload MLS / Sales Activity Files (optional; PDF, CSV, XLS, XLSX)",
        type=["pdf", "csv", "xls", "xlsx"],
        accept_multiple_files=True
    )

    # Require at least Compass plus one of Phone or MLS
    if st.button("Run Automation"):
        if not compass_file:
            st.error("Please provide a Compass CSV.")
        elif not phone_file and not mls_files:
            st.error("Please provide at least one of Phone CSV or MLS files.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                compass_path = os.path.join(tmpdir, "compass.csv")
                with open(compass_path, "wb") as f:
                    f.write(compass_file.getbuffer())

                phone_path = None
                if phone_file:
                    phone_path = os.path.join(tmpdir, "phone.csv")
                    with open(phone_path, "wb") as f:
                        f.write(phone_file.getbuffer())

                mls_paths = []
                if mls_files:
                    for i, mls_file in enumerate(mls_files, start=1):
                        ext = os.path.splitext(mls_file.name)[1]
                        mls_path = os.path.join(tmpdir, f"mls_{i}{ext}")
                        with open(mls_path, "wb") as f:
                            f.write(mls_file.getbuffer())
                        mls_paths.append(mls_path)

                output_dir = os.path.join(tmpdir, "output")

                try:
                    # Run the main processing function.
                    # Note: phone_path may be None.
                    extracted_file, merged_file, import_files = process_files(
                        compass_file=compass_path,
                        phone_file=phone_path,
                        mls_files=mls_paths,
                        output_dir=output_dir,
                        logger=logger
                    )

                    if extracted_file and os.path.exists(extracted_file):
                        with open(extracted_file, "rb") as f:
                            st.session_state["extracted_csv_data"] = f.read()

                    if merged_file and os.path.exists(merged_file):
                        with open(merged_file, "rb") as f:
                            st.session_state["merged_csv_data"] = f.read()

                    # Open import files correctly and store them in session state
                    st.session_state["import_files"] = []
                    for f in import_files:
                        with open(f, "rb") as file:
                            st.session_state["import_files"].append((os.path.basename(f), file.read()))

                    st.success("Processing completed! Check logs below.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Display download buttons
    if "extracted_csv_data" in st.session_state:
        st.download_button(
            label="Download Extracted Addresses CSV",
            data=st.session_state["extracted_csv_data"],
            file_name="extracted_addresses.csv",
            mime="text/csv"
        )

    if "merged_csv_data" in st.session_state:
        st.download_button(
            label="Download Merged Compass CSV",
            data=st.session_state["merged_csv_data"],
            file_name="compass_merged.csv",
            mime="text/csv"
        )

    if "import_files" in st.session_state and st.session_state["import_files"]:
        st.subheader("Compass Import File(s)")
        for filename, file_data in st.session_state["import_files"]:
            st.download_button(
                label=f"Download {filename}",
                data=file_data,
                file_name=filename,
                mime="text/csv"
            )

    # Show logs
    if logs:
        st.subheader("Logs")
        for log_line in logs:
            st.text(log_line)

if __name__ == "__main__":
    main()
