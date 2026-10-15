from pathlib import Path
import re

# Percorso del progetto
PROJECT_PATH = Path(__file__).resolve().parent.parent

# Un singolo documento per il test
TEST_FILE = (
    PROJECT_PATH
    / "data"
    / "raw"
    / "enronRandom-XML"
    / "train"
    / "enron0"
    / "arnold-j_all_documents_1027"
)

print(f"File analizzato:\n{TEST_FILE}")

# Controlliamo che esista
if not TEST_FILE.exists():
    print("\nERRORE: il file non esiste.")
    exit()

# Leggiamo il documento
text = TEST_FILE.read_text(
    encoding="latin-1",
    errors="replace"
)

print(f"\nLunghezza del documento: {len(text)} caratteri")

# Cerchiamo le annotazioni PERSON
pattern = r"<true_name>(.*?)</true_name>"

matches = re.findall(pattern, text, flags=re.DOTALL)

print(f"\nNumero di PERSON trovate: {len(matches)}")

print("\nPERSON:")

for i, match in enumerate(matches, start=1):
    print(f"{i}. {match.strip()}")