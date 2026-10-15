
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "reid_candidates.jsonl"
OUTPUT_DIR = ROOT / "data" / "processed"

CONTEXT_FILE = OUTPUT_DIR / "reid_context_candidates.jsonl"
REVIEW_FILE = OUTPUT_DIR / "reid_manual_review.jsonl"


def normalize(value):
    return re.sub(r"\s+", " ", value.casefold()).strip()


def main():
    context_candidates = []
    manual_review = []
    seen = set()

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            doc_id = item.get("document_id", "")
            name = item.get("original_name", "").strip()
            placeholder = item.get("placeholder", "").strip()
            text = item.get("text_anonymized", "")

            if not all([doc_id, name, placeholder, text]):
                continue

            # Una sola valutazione per coppia documento-placeholder.
            key = (doc_id, placeholder)
            if key in seen:
                continue
            seen.add(key)

            normalized_name = normalize(name)
            normalized_text = normalize(text)

            reasons = []

            # Nome completo ancora presente nel documento.
            if normalized_name in normalized_text:
                reasons.append("nome_completo_presente")

            # Email che contengono il nome o parti significative del nome.
            emails = re.findall(
                r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
                text,
                flags=re.IGNORECASE,
            )

            tokens = [
                token.casefold()
                for token in re.findall(r"[^\W\d_]+", name, flags=re.UNICODE)
                if len(token) >= 3
            ]

            for email in emails:
                email_local = email.split("@", 1)[0].casefold()
                email_parts = re.findall(r"[a-z]+", email_local)

                if any(
                    token in email_parts or token in email_local
                    for token in tokens
                ):
                    reasons.append("possibile_nome_in_email")
                    break

            # Il filtro è prudenziale: gli altri identificatori residui
            # richiedono una revisione manuale.
            result = {
                **item,
                "filter_reasons": sorted(set(reasons)),
            }

            if reasons:
                manual_review.append(result)
            else:
                context_candidates.append(result)

    with CONTEXT_FILE.open("w", encoding="utf-8") as f:
        for item in context_candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with REVIEW_FILE.open("w", encoding="utf-8") as f:
        for item in manual_review:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Coppie documento-placeholder esaminate: {len(seen)}")
    print(f"Candidati contestuali preliminari: {len(context_candidates)}")
    print(f"Casi da revisionare: {len(manual_review)}")
    print(f"File candidati: {CONTEXT_FILE}")
    print(f"File revisione: {REVIEW_FILE}")


if __name__ == "__main__":
    main()
