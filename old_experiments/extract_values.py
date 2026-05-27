import pdfplumber
import re
import pandas as pd
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

PDF_FOLDER = "papers"
OUTPUT_CSV = "cleaned_extraction_results.csv"

# =====================================================
# KEYWORD-BASED CONTEXTUAL REGEX
# =====================================================

patterns = {
    "current_A":
        r'(?:current|welding current|arc current).*?(\d+\.?\d*)\s*A',

    "voltage_V":
        r'(?:voltage|arc voltage).*?(\d+\.?\d*)\s*V',

    "travel_speed":
        r'(?:travel speed|welding speed|speed).*?(\d+\.?\d*)\s*(?:mm/min|cm/min)',

    "oxygen_ppm":
        r'(?:oxygen|oxygen content).*?(\d+\.?\d*)\s*ppm',

    "penetration_mm":
        r'(?:penetration|depth of penetration|depth).*?(\d+\.?\d*)\s*mm',

    "hardness_HV":
        r'(?:hardness).*?(\d+\.?\d*)\s*HV',

    "ferrite_FN":
        r'(?:ferrite number|ferrite|FN).*?(\d+\.?\d*)',

    "heat_input":
        r'(?:heat input).*?(\d+\.?\d*)\s*(?:kJ/mm|kj/mm)'
}

# =====================================================
# STORAGE
# =====================================================

results = []

# =====================================================
# GET PDF FILES
# =====================================================

pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))

print(f"\nFound {len(pdf_files)} PDF files.\n")

# =====================================================
# PROCESS EACH PDF
# =====================================================

for pdf_file in pdf_files:

    print("=" * 70)
    print(f"Processing: {pdf_file.name}")
    print("=" * 70)

    full_text = ""

    try:

        # ---------------------------------------------
        # EXTRACT TEXT FROM PDF
        # ---------------------------------------------

        with pdfplumber.open(pdf_file) as pdf:

            for page in pdf.pages:

                text = page.extract_text()

                if text:
                    full_text += text + "\n"

        # Lowercase version
        text_lower = full_text.lower()

        # ---------------------------------------------
        # STORE RESULT ROW
        # ---------------------------------------------

        row = {
            "paper_name": pdf_file.name
        }

        # ---------------------------------------------
        # EXTRACT PARAMETERS
        # ---------------------------------------------

        for key, pattern in patterns.items():

            matches = re.findall(pattern, text_lower, re.IGNORECASE)

            # Remove duplicates
            matches = list(set(matches))

            # Keep first few values
            if matches:
                row[key] = ", ".join(matches[:5])
            else:
                row[key] = ""

        # =====================================================
        # CONTEXT SENTENCE EXTRACTION
        # =====================================================

        context_keywords = [
            "penetration",
            "oxygen",
            "ferrite",
            "hardness",
            "marangoni",
            "heat input",
            "activated flux"
        ]

        sentences = re.split(r'(?<=[.!?])\s+', full_text)

        selected_sentences = []

        for sentence in sentences:

            sentence_lower = sentence.lower()

            for keyword in context_keywords:

                if keyword in sentence_lower:

                    selected_sentences.append(sentence.strip())
                    break

        # Store first few important sentences
        row["context_sentences"] = " | ".join(selected_sentences[:10])

        # =====================================================
        # SIMPLE FLUX DETECTION
        # =====================================================

        fluxes = []

        known_fluxes = [
            "tio2",
            "sio2",
            "cr2o3",
            "moo3",
            "zro2",
            "al2o3",
            "mno2"
        ]

        for flux in known_fluxes:

            if flux in text_lower:
                fluxes.append(flux)

        row["detected_fluxes"] = ", ".join(fluxes)

        # =====================================================
        # SAVE ROW
        # =====================================================

        results.append(row)

        print("Extraction successful.\n")

    except Exception as e:

        print(f"Error processing {pdf_file.name}")
        print(e)

# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(results)

# =====================================================
# SAVE CSV
# =====================================================

df.to_csv(OUTPUT_CSV, index=False)

# =====================================================
# DONE
# =====================================================

print("\n")
print("=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(f"\nCSV saved as: {OUTPUT_CSV}\n")

print(df.head())