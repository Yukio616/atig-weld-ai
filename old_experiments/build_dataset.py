import pdfplumber
import re
import pandas as pd
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================

PDF_FOLDER = "papers"
OUTPUT_CSV = "final_atig_dataset.csv"

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def clean_values(values, min_val=None, max_val=None):

    cleaned = []

    for v in values:

        try:
            val = float(v)

            if min_val is not None and val < min_val:
                continue

            if max_val is not None and val > max_val:
                continue

            cleaned.append(val)

        except:
            continue

    return cleaned


def extract_first(values):

    if len(values) > 0:
        return values[0]

    return None

# ==========================================================
# STORAGE
# ==========================================================

dataset = []

# ==========================================================
# KNOWN FLUXES
# ==========================================================

known_fluxes = [
    "tio2",
    "sio2",
    "cr2o3",
    "moo3",
    "zro2",
    "al2o3",
    "mno2"
]

# ==========================================================
# GET PDF FILES
# ==========================================================

pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))

print(f"\nFound {len(pdf_files)} PDF files\n")

# ==========================================================
# PROCESS EACH PDF
# ==========================================================

for pdf_file in pdf_files:

    print("=" * 60)
    print(f"Processing: {pdf_file.name}")
    print("=" * 60)

    full_text = ""

    try:

        # --------------------------------------------------
        # READ PDF
        # --------------------------------------------------

        with pdfplumber.open(pdf_file) as pdf:

            for page in pdf.pages:

                text = page.extract_text()

                if text:
                    full_text += text + "\n"

        text_lower = full_text.lower()

        # --------------------------------------------------
        # CURRENT
        # --------------------------------------------------

        current_matches = re.findall(
            r'(?:current|welding current|arc current).*?(\d+\.?\d*)\s*a',
            text_lower
        )

        current_values = clean_values(current_matches, 40, 400)

        # --------------------------------------------------
        # VOLTAGE
        # --------------------------------------------------

        voltage_matches = re.findall(
            r'(?:voltage|arc voltage).*?(\d+\.?\d*)\s*v',
            text_lower
        )

        voltage_values = clean_values(voltage_matches, 5, 30)

        # --------------------------------------------------
        # SPEED
        # --------------------------------------------------

        speed_matches = re.findall(
            r'(?:travel speed|welding speed|speed).*?(\d+\.?\d*)\s*(?:mm/min|cm/min)',
            text_lower
        )

        speed_values = clean_values(speed_matches, 10, 500)

        # --------------------------------------------------
        # PENETRATION
        # --------------------------------------------------

        penetration_matches = re.findall(
            r'(?:penetration|depth of penetration|weld depth|depth).*?(\d+\.?\d*)\s*mm',
            text_lower
        )

        penetration_values = clean_values(penetration_matches, 0.5, 20)

        # --------------------------------------------------
        # OXYGEN
        # --------------------------------------------------

        oxygen_matches = re.findall(
            r'(?:oxygen|oxygen content).*?(\d+\.?\d*)\s*ppm',
            text_lower
        )

        oxygen_values = clean_values(oxygen_matches, 10, 1000)

        # --------------------------------------------------
        # HARDNESS
        # --------------------------------------------------

        hardness_matches = re.findall(
            r'(?:hardness).*?(\d+\.?\d*)\s*hv',
            text_lower
        )

        hardness_values = clean_values(hardness_matches, 50, 600)

        # --------------------------------------------------
        # FERRITE
        # --------------------------------------------------

        ferrite_matches = re.findall(
            r'(?:ferrite number|delta ferrite|ferrite|fn).*?(\d+\.?\d*)',
            text_lower
        )

        ferrite_values = clean_values(ferrite_matches, 0, 30)

        # --------------------------------------------------
        # FLUX DETECTION
        # --------------------------------------------------

        detected_fluxes = []

        for flux in known_fluxes:

            if flux in text_lower:
                detected_fluxes.append(flux)

        # --------------------------------------------------
        # CONTEXT EXTRACTION
        # --------------------------------------------------

        context_keywords = [
            "penetration",
            "oxygen",
            "ferrite",
            "marangoni",
            "activated flux",
            "hardness"
        ]

        sentences = re.split(r'(?<=[.!?])\s+', full_text)

        context_sentences = []

        for sentence in sentences:

            s = sentence.lower()

            for keyword in context_keywords:

                if keyword in s:
                    context_sentences.append(sentence.strip())
                    break

        # --------------------------------------------------
        # BUILD ROW
        # --------------------------------------------------

        row = {
            "paper_name": pdf_file.name,
            "fluxes": ", ".join(detected_fluxes),
            "current_A": extract_first(current_values),
            "voltage_V": extract_first(voltage_values),
            "travel_speed": extract_first(speed_values),
            "penetration_mm": extract_first(penetration_values),
            "oxygen_ppm": extract_first(oxygen_values),
            "hardness_HV": extract_first(hardness_values),
            "ferrite_FN": extract_first(ferrite_values),
            "context": " | ".join(context_sentences[:5])
        }

        # --------------------------------------------------
        # HEAT INPUT
        # --------------------------------------------------

        if (
            row["current_A"] is not None and
            row["voltage_V"] is not None and
            row["travel_speed"] is not None
        ):

            try:

                heat_input = (
                    row["voltage_V"] *
                    row["current_A"]
                ) / row["travel_speed"]

                row["heat_input"] = round(heat_input, 3)

            except:

                row["heat_input"] = None

        else:
            row["heat_input"] = None

        # --------------------------------------------------
        # SAVE ROW
        # --------------------------------------------------

        dataset.append(row)

        print("Extraction successful")
        print(row)
        print()

    except Exception as e:

        print(f"Error processing {pdf_file.name}")
        print(e)

# ==========================================================
# CREATE DATAFRAME
# ==========================================================

final_df = pd.DataFrame(dataset)

# ==========================================================
# REMOVE EMPTY ROWS
# ==========================================================

final_df = final_df.dropna(
    subset=[
        "current_A",
        "penetration_mm"
    ],
    how="all"
)

# ==========================================================
# SAVE CSV
# ==========================================================

final_df.to_csv(OUTPUT_CSV, index=False)

print("\n")
print("=" * 60)
print("FINAL DATASET CREATED")
print("=" * 60)

print(final_df.head())

print(f"\nSaved as: {OUTPUT_CSV}")