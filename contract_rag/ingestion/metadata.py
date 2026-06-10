import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Clause Detection Patterns
# --------------------------------------------------

CLAUSE_PATTERNS = {
    "liability": [
        r"liabilit",
        r"limitation of liability",
        r"liable"
    ],
    "indemnification": [
        r"indemnif",
        r"hold harmless",
        r"indemnitor"
    ],
    "termination": [
        r"terminat",
        r"cancellation",
        r"expiration"
    ],
    "confidentiality": [
        r"confidential",
        r"non-disclosure",
        r"proprietary"
    ],
    "payment": [
        r"payment",
        r"invoice",
        r"fee",
        r"compensation"
    ],
    "governing_law": [
        r"governing law",
        r"jurisdiction",
        r"venue"
    ],
    "renewal": [
        r"renew",
        r"auto-renew",
        r"evergreen"
    ],
    "dispute": [
        r"dispute",
        r"arbitration",
        r"mediation"
    ]
}

# --------------------------------------------------
# Contract Type Detection
# --------------------------------------------------

CONTRACT_TYPES = {
    "nda": [
        r"non-disclosure",
        r"\bnda\b",
        r"confidentiality agreement"
    ],
    "msa": [
        r"master service",
        r"\bmsa\b",
        r"master agreement"
    ],
    "employment": [
        r"employment agreement",
        r"offer letter"
    ],
    "lease": [
        r"lease agreement",
        r"rental",
        r"landlord"
    ],
    "purchase": [
        r"purchase agreement",
        r"sale of goods",
        r"vendor"
    ]
}

# --------------------------------------------------
# Date Extraction
# --------------------------------------------------

def extract_dates(text: str) -> Dict[str, str | None]:
    """
    Extract common contract dates.
    """

    dates = {}

    patterns = {
        "effective_date": (
            r"(?:effective|commencement|start)\s+date[:\s]+"
            r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4})"
        ),
        "expiration_date": (
            r"(?:expir|terminat|end)\s+date[:\s]+"
            r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4})"
        ),
        "renewal_date": (
            r"(?:renew)\s+date[:\s]+"
            r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4})"
        ),
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        dates[key] = match.group(1) if match else None

    return dates


# --------------------------------------------------
# Clause Classification
# --------------------------------------------------

def detect_clause_type(text: str) -> str:
    """
    Detect clause category using keyword matching.
    """

    text_lower = text.lower()

    for clause_type, patterns in CLAUSE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return clause_type

    return "general"


# --------------------------------------------------
# Contract Classification
# --------------------------------------------------

def detect_contract_type(text: str) -> str:
    """
    Detect contract type from document text.
    """

    text_lower = text.lower()

    for contract_type, patterns in CONTRACT_TYPES.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return contract_type

    return "general"


# --------------------------------------------------
# Party Extraction
# --------------------------------------------------

def extract_parties(text: str) -> List[str]:
    """
    Extract company/person names from contract text.
    """

    patterns = [
        r"between\s+([A-Z][A-Za-z\s,\.]+?)\s+(?:and|&)",
        r"(?:Client|Vendor|Party|Employer|Employee)[:\s]+([A-Z][A-Za-z\s,\.]{3,40})"
    ]

    parties = []

    for pattern in patterns:
        matches = re.findall(pattern, text[:2000])

        for match in matches:
            party = match.strip()

            if len(party) > 3:
                parties.append(party)

    return list(set(parties[:4]))


# --------------------------------------------------
# Metadata Extraction
# --------------------------------------------------

def extract_metadata(
    chunk: Dict,
    full_text: str
) -> Dict:
    """
    Add metadata fields to chunk before indexing.
    """

    try:
        chunk["clause_type"] = detect_clause_type(
            chunk.get("text", "")
        )

        chunk["contract_type"] = detect_contract_type(
            full_text[:3000]
        )

        chunk["parties"] = extract_parties(
            full_text[:3000]
        )

        dates = extract_dates(
            full_text[:5000]
        )

        chunk.update(dates)

    except Exception as e:
        logger.exception(
            "Metadata extraction failed: %s",
            str(e)
        )

    return chunk