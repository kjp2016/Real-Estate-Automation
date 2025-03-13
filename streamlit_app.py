# streamlit_app.py

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
    1. Select a Compass Export CSV.
    2. Select a Phone Export CSV.
    3. Select multiple MLS/Sales Activity files (PDF, CSV, Excel).
    4. Run the data extraction, merging, and classification flow.
    5. Download your processed files.
    """)

    # FILE UPLOADS
    compass_file = st.file_uploader("Upload Compass Export CSV", type=["csv"])
    phone_file = st.file_uploader("Upload Phone Export CSV", type=["csv"])
    mls_files = st.file_uploader(
        "Upload MLS / Sales Activity Files (PDF, CSV, XLS, XLSX)", 
        type=["pdf", "csv", "xls", "xlsx"], 
        accept_multiple_files=True
    )

    # Ensure the button is inside the `main()` function
    if st.button("Run Automation"):
        if not compass_file or not phone_file:
            st.error("Please provide Compass CSV and Phone CSV.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save Compass file
                compass_path = os.path.join(tmpdir, "compass.csv")
                with open(compass_path, "wb") as f:
                    f.write(compass_file.getbuffer())

                # Save Phone file
                phone_path = os.path.join(tmpdir, "phone.csv")
                with open(phone_path, "wb") as f:
                    f.write(phone_file.getbuffer())

                # Save MLS files if they exist
                mls_paths = []
                if mls_files:
                    for i, mls_file in enumerate(mls_files, start=1):
                        ext = os.path.splitext(mls_file.name)[1]
                        mls_path = os.path.join(tmpdir, f"mls_{i}{ext}")
                        with open(mls_path, "wb") as f:
                            f.write(mls_file.getbuffer())
                        mls_paths.append(mls_path)

                # Create output directory in temp
                output_dir = os.path.join(tmpdir, "output")

                try:
                    # Run the main process (handles missing MLS files)
                    process_files(
                        compass_file=compass_path,
                        phone_file=phone_path,
                        mls_files=mls_paths,  # Pass an empty list if no MLS files
                        output_dir=output_dir,
                        logger=logger
                    )
                    st.success("Processing completed! Check logs below.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Download buttons outside the if-statement
    if "extracted_csv_data" in st.session_state and st.session_state["extracted_csv_data"]:
        st.download_button(
            label="Download Extracted Addresses CSV",
            data=st.session_state["extracted_csv_data"],
            file_name="extracted_addresses.csv",
            mime="text/csv",
            key="extracted_download"
        )

    if "merged_csv_data" in st.session_state and st.session_state["merged_csv_data"]:
        st.download_button(
            label="Download Merged Compass CSV",
            data=st.session_state["merged_csv_data"],
            file_name="compass_merged.csv",
            mime="text/csv",
            key="merged_download"
        )

    # Provide download links for compass_import_part*.csv (if any)
    if "import_files" in st.session_state and st.session_state["import_files"]:
        st.subheader("Compass Import File(s)")
        for i, (filename, file_data) in enumerate(st.session_state["import_files"], start=1):
            st.download_button(
                label=f"Download {filename}",
                data=file_data,
                file_name=filename,
                mime="text/csv",
                key=f"import_download_{i}"
            )

    # Show logs
    if logs:
        st.subheader("Logs")
        for log_line in logs:
            st.text(log_line)

if __name__ == "__main__":
    main()
