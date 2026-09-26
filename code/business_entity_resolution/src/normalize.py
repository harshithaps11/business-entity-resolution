"""Text normalisation for business names and addresses.

Everything here is rule-based and country-agnostic: it works on any Latin or
non-Latin input (unseen countries such as France included). Non-Latin scripts
(Devanagari, Telugu, Gujarati, Bengali, ...) are transliterated with unidecode
so they share a token space with the Latin records.
"""
import re
from functools import lru_cache

from unidecode import unidecode

# ---------------------------------------------------------------- vocabularies
# Legal-form / entity-type tokens. Removed from the "core" name, kept separately
# as a feature (a legal-form mismatch is a weak signal, not a hard veto).
LEGAL = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "pvt", "private", "pc", "pllc", "llp", "lp", "plc", "l", "c",
    "pte", "opc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "ei", "snc", "ets",
    "gmbh", "ag", "bv", "nv", "the",
}
LEGAL_CANON = {
    "incorporated": "inc", "corporation": "corp", "company": "co",
    "limited": "ltd", "private": "pvt",
}
# Trade-name markers: "X dba Y" -> Y is the name that other sources carry.
DBA_RE = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|t\s*/\s*a|aka|doing business as)\b", re.I)
DOMAIN_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|co|in|io|biz|us|fr|info|co\.in)\b")
NONALNUM_RE = re.compile(r"[^a-z0-9]+")
OCR_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th|er|e|eme)$")
HAS_ALPHA = re.compile(r"[a-z]")
HAS_DIGIT = re.compile(r"\d")
NON_ASCII = re.compile(r"[^\x00-\x7f]")

ADDR_ABBR = {
    # English (US / India)
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "drive": "dr", "boulevard": "blvd", "court": "ct", "lane": "ln", "highway": "hwy",
    "place": "pl", "terrace": "ter", "circle": "cir", "parkway": "pkwy", "square": "sq",
    "trail": "trl", "way": "wy", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "suite": "ste", "apartment": "apt", "floor": "fl", "building": "bldg", "bldng": "bldg",
    "unit": "unit", "room": "rm", "roomno": "rm", "sector": "sec", "near": "nr",
    "opposite": "opp", "opp": "opp", "behind": "bh", "nagar": "ngr", "marg": "mg",
    "township": "twp", "mount": "mt", "saint": "st", "fort": "ft", "center": "ctr",
    "centre": "ctr", "county": "cnty", "first": "1st", "second": "2nd", "third": "3rd",
    "fourth": "4th", "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
    "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
    "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th", "sixteenth": "16th",
    "seventeenth": "17th", "eighteenth": "18th", "nineteenth": "19th", "twentieth": "20th",
    # French
    "rue": "r", "boulevard": "blvd", "bd": "blvd", "chemin": "ch", "che": "ch",
    "impasse": "imp", "allee": "all", "route": "rte", "quai": "qu", "cours": "crs",
    "faubourg": "fbg", "residence": "res", "bis": "bis",
}
# Filler tokens that carry no identity (house-number prefixes etc.).
ADDR_STOP = {
    "no", "nos", "num", "number", "h", "hno", "house", "door", "plot", "flat", "shop",
    "the", "of", "and", "at", "de", "la", "le", "les", "du", "des", "d", "l", "n",
    "city", "dist", "district", "po", "ps", "via", "ward", "khasra", "kh", "kno", "sr", "survey",
    "null", "none", "nan", "na",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py",
}
_STATE_PATTERNS = sorted({**US_STATES, **IN_STATES}.items(), key=lambda kv: -len(kv[0]))
STATE_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k, _ in _STATE_PATTERNS) + r")\b")
STATE_CANON = dict(_STATE_PATTERNS)
STATE_CODES = set(STATE_CANON.values())


NAME_MAP: dict = {}
ADDR_MAP: dict = {}
NATIVE_SEG = re.compile(r"[^\x00-\x7f][^,]*")


def load_translit(path: str) -> None:
    """Install native-script token maps learned by learn_translit.py."""
    import json
    m = json.load(open(path))
    NAME_MAP.clear(); NAME_MAP.update(m["name"])
    ADDR_MAP.clear(); ADDR_MAP.update(m["addr"])


@lru_cache(maxsize=1 << 16)
def _translit(s: str) -> str:
    return unidecode(s).lower()


def translit(s: str) -> str:
    if not s:
        return ""
    return _translit(s) if NON_ASCII.search(s) else s.lower()


def _ocr_fix(tok: str) -> str:
    """'chemica1s' -> 'chemicals', '5ervices' -> 'services'; leaves '4th', '446'."""
    if HAS_ALPHA.search(tok) and HAS_DIGIT.search(tok) and not ORDINAL_RE.match(tok):
        return tok.translate(OCR_MAP)
    return tok


def normalize_name(raw: str):
    """Return (tokens, core_tokens, legal_tokens, alt_core_tokens, flags)."""
    s = translit(raw)
    is_domain = False
    m = DOMAIN_RE.search(s)
    if m:
        is_domain = True
        s = DOMAIN_RE.sub(lambda mm: " " + mm.group(1).replace("-", " ") + " ", s)
    alt = ""
    parts = DBA_RE.split(s)
    has_dba = len(parts) > 1
    if has_dba:  # trade name is what other sources use; keep owner as alt
        s, alt = parts[-1], " ".join(parts[:-1])
    s = s.replace("&", " and ").replace("+", " and ")
    s = s.replace("l.l.c", "llc").replace("p.c.", "pc").replace("pvt.", "pvt ")
    toks = [_ocr_fix(t) for t in NONALNUM_RE.split(s) if t]
    if flags_native := bool(NON_ASCII.search(raw or "")):
        toks = [NAME_MAP.get(t, t) for t in toks]
    toks = [LEGAL_CANON.get(t, t) for t in toks]
    toks = [t for t in toks if t != "and"]
    core = [t for t in toks if t not in LEGAL]
    legal = sorted({t for t in toks if t in LEGAL and t not in {"the", "l", "c"}})
    if not core:  # name was only legal words; keep them
        core = toks
    alt_core = [t for t in (LEGAL_CANON.get(x, x) for x in NONALNUM_RE.split(alt) if x)
                if t not in LEGAL and t != "and"]
    flags = (int(flags_native), int(is_domain), int(has_dba))
    return toks, core, legal, alt_core, flags


def normalize_address(raw: str):
    """Return (word_tokens, number_tokens, state_code)."""
    if not raw:
        return [], [], ""
    if NON_ASCII.search(raw):
        raw = NATIVE_SEG.sub(lambda m: " " + ADDR_MAP.get(
            " ".join(t for t in NONALNUM_RE.split(translit(m.group(0))) if t), m.group(0)) + " ", raw)
    s = translit(raw)
    s = s.replace("n°", " ").replace("no.", " ").replace("#", " ")
    state = ""
    ms = STATE_RE.findall(s)
    if ms:
        state = STATE_CANON[ms[-1]]
        s = STATE_RE.sub(" ", s)
    words, nums = [], []
    for t in NONALNUM_RE.split(s):
        if not t:
            continue
        if HAS_DIGIT.search(t):
            if ORDINAL_RE.match(t):
                words.append(t)
            else:
                # split "4219" / "b3" / "124a": keep digit run as the number
                for d in re.findall(r"\d+", t):
                    nums.append(d.lstrip("0") or "0")
            continue
        t = ADDR_ABBR.get(t, t)
        if t in ADDR_STOP:
            continue
        words.append(t)
    # trailing two-letter token (e.g. ", NC" / ", MH") that looks like a state code
    if not state and words and words[-1] in STATE_CODES:
        state = words[-1]
    return words, nums, state
