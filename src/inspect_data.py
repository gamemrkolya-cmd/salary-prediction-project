from pathlib import Path
import pandas as pd
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
EXPECTED_NAME = "salary_data_cleaned.csv"

# Try the canonical path first
data_dir = BASE_DIR / "data"
if not data_dir.exists():
	# Handle cases where the directory name includes a trailing colon (e.g. "data:")
	candidates = [p for p in BASE_DIR.iterdir() if p.is_dir() and p.name.lower().startswith("data")]
	if candidates:
		data_dir = candidates[0]

data_path = data_dir / EXPECTED_NAME

# If expected file not found, try a recursive search for a similarly named CSV
if not data_path.exists():
	matches = list(BASE_DIR.rglob("*salary*cleaned*.csv"))
	if matches:
		data_path = matches[0]

print("Reading from:", data_path)

if not data_path.exists():
	# Provide a helpful error message showing nearby directories/files
	nearby = []
	try:
		if data_dir.exists():
			nearby = [p.name for p in sorted(data_dir.iterdir())]
		else:
			nearby = [p.name for p in sorted(BASE_DIR.iterdir())]
	except Exception:
		nearby = []
	msg = f"Data file not found at expected path '{data_path}'.\n"
	if nearby:
		msg += f"Nearby files/dirs: {nearby}\n"
	msg += "You can place the CSV at 'data/salary_data_cleaned.csv' or update the script."
	print(msg, file=sys.stderr)
	raise FileNotFoundError(msg)

df = pd.read_csv(data_path)

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())
