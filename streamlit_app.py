# streamlit_app.py

import streamlit as st
import os
import tempfile
import io
import csv

from realestate_processor import process_files

def main():
    st.title("Real Estate Data Processor (Streamlit)")

    # We'll keep logs in a list
    logs = []

    def logger(message: str):
        """Logger function to capture messages from the processor."""
        logs.append(message)
        # Optionally, print them (helps in debugging locally):
        # print(message)

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
    mls_files = st.file_uploader("Upload MLS / Sales Activity Files (PDF, CSV, XLS, XLSX)", 
                                 type=["pdf", "csv", "xls", "xlsx"], accept_multiple_files=True)

    if st.button("Run Automation"):
        if not compass_file or not phone_file or not mls_files:
            st.error("Please provide all required files: Compass CSV, Phone CSV, and at least one MLS file.")
        else:
            # Use a temporary directory to store the uploaded files
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save Compass file
                compass_path = os.path.join(tmpdir, "compass.csv")
                with open(compass_path, "wb") as f:
                    f.write(compass_file.getbuffer())

                # Save Phone file
                phone_path = os.path.join(tmpdir, "phone.csv")
                with open(phone_path, "wb") as f:
                    f.write(phone_file.getbuffer())

                # Save each MLS file
                mls_paths = []
                for i, mls_file in enumerate(mls_files, start=1):
                    ext = os.path.splitext(mls_file.name)[1]
                    mls_path = os.path.join(tmpdir, f"mls_{i}{ext}")
                    with open(mls_path, "wb") as f:
                        f.write(mls_file.getbuffer())
                    mls_paths.append(mls_path)

                # Create an output directory in temp as well
                output_dir = os.path.join(tmpdir, "output")
                
                try:
                    process_files(
                        compass_file=compass_path,
                        phone_file=phone_path,
                        mls_files=mls_paths,
                        output_dir=output_dir,
                        logger=logger
                    )

                    st.success("Processing completed! Check logs below.")
                    
                    # Provide download links for the output CSV files if they exist
                    extracted_csv_path = os.path.join(output_dir, "extracted_addresses.csv")
                    merged_csv_path = os.path.join(output_dir, "compass_merged.csv")

                    # Download button: extracted_addresses.csv
                    if os.path.exists(extracted_csv_path):
                        with open(extracted_csv_path, "rb") as f:
                            st.download_button(
                                label="Download Extracted Addresses CSV",
                                data=f.read(),
                                file_name="extracted_addresses.csv",
                                mime="text/csv"
                            )

                    # Download button: compass_merged.csv
                    if os.path.exists(merged_csv_path):
                        with open(merged_csv_path, "rb") as f:
                            st.download_button(
                                label="Download Merged Compass CSV",
                                data=f.read(),
                                file_name="compass_merged.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

    # Show the logs after the run
    if logs:
        st.subheader("Logs")
        for log_line in logs:
            st.text(log_line)


if __name__ == "__main__":
    main()
