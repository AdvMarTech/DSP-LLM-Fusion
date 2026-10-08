"""Merge table and paragraph extraction results into accident records.

Prefer nonempty values and use confidence to resolve field conflicts.
Match vessels by name, reconcile causes, and validate merged fields against
source documents. Write merged JSON records and a combined Excel workbook.
"""

import json
import copy
import re
import os
import argparse
import html
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Tuple

# Optional dependency for Excel output
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# LaTeX coordinate conversion

def convert_latex_coordinate(coord_str: str) -> str:
    """Convert a LaTeX coordinate to plain degree-minute notation.
    
    Accepts spaced digits, LaTeX degree and prime commands, and optional
    compass directions. Returns text such as "43°40.0'N".
    
    Args:
        coord_str: Coordinate text that may contain LaTeX formatting.
    
    Returns:
        The converted coordinate, or the input if conversion fails.
    """
    if not coord_str or not isinstance(coord_str, str):
        return coord_str
    
    # Return plain coordinates unchanged.
    if '^' not in coord_str and '\\circ' not in coord_str and '\\prime' not in coord_str:
        return coord_str
    
    original = coord_str
    
    try:
        # Remove dollar delimiters.
        coord_str = coord_str.replace('$', '')
        
        # Extract the compass direction.
        direction = ''
        # Check supported direction formats.
        direction_patterns = [
            (r'\\mathrm\s*\{\s*([NSEW])\s*\}', 1),  # \mathrm { N }
            (r'\{\s*([NSEW])\s*\}', 1),              # { N }
            (r'\b([NSEW])\s*$', 1),                  # A trailing N, S, E, or W
        ]
        
        for pattern, group in direction_patterns:
            match = re.search(pattern, coord_str, re.IGNORECASE)
            if match:
                direction = match.group(group).upper()
                coord_str = re.sub(pattern, '', coord_str, flags=re.IGNORECASE)
                break
        
        # Remove LaTeX markup.
        # Replace LaTeX degree commands and their braced variants.
        coord_str = re.sub(r'\^\s*\{\s*\\circ\s*\}', '°', coord_str)
        coord_str = re.sub(r'\^\s*\\circ', '°', coord_str)
        coord_str = re.sub(r'\\circ', '°', coord_str)
        
        # Replace LaTeX prime commands and their braced variants.
        coord_str = re.sub(r'\^\s*\{\s*\\prime\s*\}', "'", coord_str)
        coord_str = re.sub(r'\^\s*\\prime', "'", coord_str)
        coord_str = re.sub(r'\\prime', "'", coord_str)
        
        # Remove remaining LaTeX commands.
        coord_str = re.sub(r'\\[a-zA-Z]+', '', coord_str)
        
        # Remove braces.
        coord_str = coord_str.replace('{', '').replace('}', '')
        
        # Extract coordinate digits.
        # The remaining text may contain spaced digits, such as "4 3 ° 4 0 . 0 '".
        
        # Normalize whitespace before joining digit groups.
        coord_str = coord_str.strip()
        
        # Parse degree-minute notation.
        # Allow spaces between digits in both components.
        # For example, join the digits in "4 3 ° 4 0 . 0 '".
        
        # Locate the degree symbol.
        degree_pos = coord_str.find('°')
        minute_pos = coord_str.find("'")
        
        if degree_pos != -1:
            # Read the degree component.
            degree_part = coord_str[:degree_pos].strip()
            # Join spaced degree digits.
            degree_num = ''.join(degree_part.split())
            
            # Read the minute component after the degree symbol.
            if minute_pos != -1 and minute_pos > degree_pos:
                minute_part = coord_str[degree_pos+1:minute_pos].strip()
            else:
                minute_part = coord_str[degree_pos+1:].strip()
            
            # Join spaced minute digits, retaining the decimal point.
            minute_num = ''
            for char in minute_part:
                if char.isdigit() or char == '.':
                    minute_num += char
                elif char == ' ' and minute_num and minute_num[-1] != '.':
                    # Join digit groups separated by spaces.
                    continue
            
            # Build the plain-text coordinate.
            if minute_num:
                result = f"{degree_num}°{minute_num}'{direction}"
            else:
                result = f"{degree_num}°{direction}"
            
            return result.strip()
        
        # Keep the input if no degree symbol was found.
        return original
        
    except Exception as e:
        # Keep the input if conversion fails.
        return original


def convert_coordinates_in_text(text: str) -> str:
    """Convert LaTeX coordinates embedded in text, such as accident_location."""
    if not text or not isinstance(text, str):
        return text
    
    # Return text without LaTeX coordinate markers unchanged.
    if '^' not in text and '\\circ' not in text:
        return text
    
    result = text
    
    # Match dollar-delimited coordinates with an optional trailing direction.
    # For example, "$1 2 4 ^ { \circ } 1 3 . 5 ^ { \prime }$ W".
    pattern1 = r'\$[^$]*\^\s*\{?\s*\\circ[^$]*\$\s*([NSEW])?'
    
    def replace_coord_with_direction(match):
        full_match = match.group(0)
        trailing_direction = match.group(1) if match.group(1) else ''
        # Extract the dollar-delimited coordinate.
        dollar_match = re.search(r'\$[^$]*\$', full_match)
        if dollar_match:
            coord_part = dollar_match.group(0)
            converted = convert_latex_coordinate(coord_part)
            # Append the trailing direction if conversion did not include it.
            if trailing_direction and not converted.endswith(('N', 'S', 'E', 'W')):
                converted = converted.rstrip("'") + "'" + trailing_direction
            return converted
        return full_match
    
    result = re.sub(pattern1, replace_coord_with_direction, result, flags=re.IGNORECASE)
    
    # Match coordinates without dollar delimiters.
    # For example, "4 0 ^ { \circ } 3 8 . 5 ^ { \prime } N".
    pattern2 = r'([\d\s]+\^\s*\{?\s*\\circ\s*\}?[\d\s.]+\^\s*\{?\s*\\prime\s*\}?\s*\\?(?:mathrm\s*\{)?\s*[NSEW]\s*\}?)'
    
    def replace_coord(match):
        return convert_latex_coordinate(match.group(0))
    
    result = re.sub(pattern2, replace_coord, result, flags=re.IGNORECASE)
    
    return result


def create_field_metadata(
    value: Any = None,
    confidence: float = None,
    page_idx: List[int] = None,
    classification: str = None,
    section_type: str = None,
    source: str = "text",
    source_chapter: str = None
) -> Dict:
    """Create a field value with confidence and source metadata."""
    result = {
        "value": value,
        "confidence": confidence,
        "page_idx": page_idx if page_idx is not None else [],
        "classification": classification,
        "section_type": section_type,
        "source": source
    }
    # Include source_chapter when available.
    if source_chapter is not None:
        result["source_chapter"] = source_chapter
    return result


def create_location_with_coordinates(
    location: str = None,
    latitude: str = None,
    longitude: str = None,
    confidence: float = None,
    page_idx: List[int] = None,
    classification: str = None,
    section_type: str = None,
    source: str = "text"
) -> Dict:
    """Create a location field with latitude and longitude metadata."""
    return {
        "value": location,
        "Latitude and Longitude": {
            "latitude": latitude,
            "longitude": longitude
        },
        "confidence": confidence,
        "page_idx": page_idx if page_idx is not None else [],
        "classification": classification,
        "section_type": section_type,
        "source": source
    }


def create_vessel_info(vessel_name: str = None) -> Dict:
    """Create an empty vessel record."""
    return {
        "Vessel Name": create_field_metadata(value=vessel_name),
        "ship_length": create_field_metadata(),
        "ship_tonnage": create_field_metadata(),
        "vessel_built_year": create_field_metadata(),
        "flag_state": create_field_metadata(),
        "mmsi_or_imo": create_field_metadata(),
        "vessel_type": create_field_metadata(),
        "hull_material": create_field_metadata(),
        "owner": create_field_metadata(),
        "operator": create_field_metadata(),
        "crew_complement": create_field_metadata(),
        "passenger_count": create_field_metadata(),
        "casualties": create_field_metadata(),
        "property_damage": create_field_metadata(),
    }


def create_vessel_causes() -> Dict:
    """Create an empty vessel cause record."""
    return {
        "causes": [],
        "total_causes": 0
    }


def create_empty_accident_structure() -> OrderedDict:
    """Create an empty accident record."""
    result = OrderedDict()
    
    # Extraction metadata comes first.
    result["extraction_metadata"] = {
        "extraction_time": None,
        "extractor_version": "v16.0-validation-json-backfill",
        "source_files": [],
        "total_vessels": 0,
        "total_causes": 0
    }
    
    # Accident details
    result["accident_no"] = create_field_metadata()
    result["accident_time"] = create_field_metadata()
    result["accident_location"] = create_location_with_coordinates()
    result["accident_type"] = create_field_metadata()
    
    # Involved-vessel summary
    result["involved_vessels"] = create_field_metadata()
    
    # Vessel and cause records are added dynamically.
    
    # Environmental conditions
    result["weather_conditions"] = create_field_metadata()
    result["waterway_information"] = create_field_metadata()
    result["visibility"] = create_field_metadata()
    
    # Accident-level losses
    result["pollution"] = create_field_metadata()
    result["economic_loss"] = create_field_metadata()
    result["Ship_loss"] = create_field_metadata()  # The schema uses a capital S.
    
    return result


def is_field_empty(field_data: Any) -> bool:
    """Return whether a field is empty."""
    if field_data is None:
        return True
    if isinstance(field_data, dict):
        value = field_data.get("value")
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        if isinstance(value, list) and len(value) == 0:
            return True
    return False


def get_confidence(field_data: Any) -> float:
    """Return field confidence, defaulting to zero."""
    if field_data is None:
        return 0.0
    if isinstance(field_data, dict):
        conf = field_data.get("confidence")
        if conf is not None:
            return float(conf)
    return 0.0


def merge_field(field1: Any, field2: Any) -> Any:
    """Merge two fields, preferring nonempty values and then higher confidence.
    
    If both values are empty, return an empty field with the expected structure.
    """
    empty1 = is_field_empty(field1)
    empty2 = is_field_empty(field2)
    
    if empty1 and empty2:
        # Preserve the field structure when both values are empty.
        if isinstance(field1, dict):
            return field1
        elif isinstance(field2, dict):
            return field2
        else:
            return create_field_metadata()
    
    if empty1:
        return copy.deepcopy(field2)
    
    if empty2:
        return copy.deepcopy(field1)
    
    # Compare confidence when both values are present.
    conf1 = get_confidence(field1)
    conf2 = get_confidence(field2)
    
    if conf1 >= conf2:
        return copy.deepcopy(field1)
    else:
        return copy.deepcopy(field2)


def merge_location_field(loc1: Dict, loc2: Dict) -> Dict:
    """Merge a location field and its nested latitude and longitude values."""
    # Merge the main location value.
    merged = merge_field(loc1, loc2)
    
    # Ensure the coordinate substructure exists.
    if "Latitude and Longitude" not in merged:
        merged["Latitude and Longitude"] = {
            "latitude": None,
            "longitude": None
        }
    
    # Merge coordinate values.
    lat1 = loc1.get("Latitude and Longitude", {}).get("latitude") if loc1 else None
    lat2 = loc2.get("Latitude and Longitude", {}).get("latitude") if loc2 else None
    lon1 = loc1.get("Latitude and Longitude", {}).get("longitude") if loc1 else None
    lon2 = loc2.get("Latitude and Longitude", {}).get("longitude") if loc2 else None
    
    # Prefer nonempty coordinates.
    merged["Latitude and Longitude"]["latitude"] = lat1 if lat1 else lat2
    merged["Latitude and Longitude"]["longitude"] = lon1 if lon1 else lon2
    
    return merged


def merge_vessel_info(vessel1: Dict, vessel2: Dict) -> Dict:
    """Merge the fields of two vessel records."""
    # Create an empty vessel record.
    merged = create_vessel_info()
    
    # Fields to merge
    vessel_fields = [
        "Vessel Name", "ship_length", "ship_tonnage", "vessel_built_year",
        "flag_state", "mmsi_or_imo", "vessel_type", "hull_material",
        "owner", "operator", "crew_complement", "passenger_count",
        "casualties", "property_damage"
    ]
    
    for field in vessel_fields:
        field1 = vessel1.get(field) if vessel1 else None
        field2 = vessel2.get(field) if vessel2 else None
        merged[field] = merge_field(field1, field2)
    
    return merged


def get_metadata_value(field_data: Any) -> Any:
    """Unwrap a field metadata value, leaving plain values unchanged."""
    if isinstance(field_data, dict):
        return field_data.get("value")
    return field_data


def normalize_simple_name(name: Any) -> str:
    if not name:
        return ""
    text = re.sub(r"\b(?:tow|barge|vessel|ship)\b", "", str(name), flags=re.I)
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return text


def dedupe_names(names: List[Any]) -> List[str]:
    result = []
    seen = set()
    for name in names or []:
        text = re.sub(r"\s+", " ", str(name)).strip()
        key = normalize_simple_name(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def normalize_table_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;")


def normalize_money_markup(value: Any) -> str:
    """Clean monetary values affected by MinerU or LaTeX dollar-sign markup."""
    text = normalize_table_text(value)
    text = re.sub(r"\$\\?\$\s*([\d,]+(?:\.\d+)?)\s*\$\s*(million|billion|thousand|M|K)\b", r"$\1 \2", text, flags=re.I)
    text = re.sub(r"\$\\?\$\s*([\d,]+(?:\.\d+)?)\s*\$", r"$\1", text)
    text = re.sub(r"\$\\?\$\s*([\d,]+(?:\.\d+)?)", r"$\1", text)
    text = re.sub(r"\\\$", "$", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rows_from_html_table(html_text: str) -> List[List[str]]:
    if not html_text:
        return []
    rows = []
    for tr_match in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", html_text, flags=re.I | re.S):
        row_html = tr_match.group(1)
        cells = []
        for cell_match in re.finditer(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.I | re.S):
            cells.append(normalize_table_text(cell_match.group(1)))
        if cells:
            rows.append(cells)
    return rows


def extract_involved_vessels_from_rows(rows: List[List[str]]) -> List[str]:
    """Read the full vessel list from source Input tables, including later tables missed by step 4."""
    candidates: List[str] = []
    for idx, row in enumerate(rows):
        if len(row) < 2:
            continue
        label = normalize_table_text(row[0]).lower()
        label_key = re.sub(r"[^a-z]+", " ", label).strip()
        if label_key not in {"vessel name", "vessel names", "vessel", "vessels"} and "vessel names" not in label_key:
            continue

        if label_key in {"vessel", "vessels"} and len(row) > 2:
            for cell in row[1:]:
                cleaned = normalize_table_text(cell)
                if cleaned and normalize_simple_name(cleaned):
                    candidates.append(cleaned)
            continue

        value = normalize_table_text(" ".join(str(cell) for cell in row[1:]))
        if "vessel names" in label_key and not re.search(r"\b(?:tow|barge|vessel|ship|M/V|S/S|T/V|F/V)\b", value, flags=re.I):
            if idx + 1 < len(rows):
                next_label = re.sub(r"[^a-z]+", " ", normalize_table_text(rows[idx + 1][0]).lower()).strip()
                next_value = normalize_table_text(" ".join(str(cell) for cell in rows[idx + 1][1:]))
                if not next_label and next_value:
                    value = next_value
        if not value:
            continue

        parts = []
        barge_match = re.search(r"\bbarges?\s+(.+)$", value, flags=re.I)
        vessel_part = value
        if barge_match:
            vessel_part = value[:barge_match.start()].strip(" ,")
            barge_part = barge_match.group(1)
        parts.extend(re.split(r",\s+and\s+|,\s*and\s+|,\s+(?=[A-Z][A-Za-z.& ]{2,})|\band\b|;|/", vessel_part))
        if barge_match:
            parts.extend(re.split(r",\s*|\band\b|;", barge_part))
        for part in parts:
            cleaned = normalize_table_text(part)
            cleaned = re.sub(r"^(?:and\s+)?barges?\s+", "", cleaned, flags=re.I)
            cleaned = re.sub(r"\b(?:tow|tows)\b$", "", cleaned, flags=re.I).strip(" ,")
            if cleaned and normalize_simple_name(cleaned):
                candidates.append(cleaned)

    return dedupe_names(candidates)


def extract_identifier_from_sources(*json_datas: Dict) -> Optional[str]:
    for json_data in json_datas:
        metadata = (json_data or {}).get("extraction_metadata", {})
        source_files = metadata.get("source_files", [])
        if isinstance(source_files, str):
            source_files = [source_files]
        for source in source_files or []:
            pure_code = re.fullmatch(r"\s*(\d+)(?:\.0)?\s*", str(source))
            if pure_code:
                return pure_code.group(1)
            match = re.search(r"(?:^|[/\\])(\d+)_content_list", str(source))
            if match:
                return match.group(1)
    return None


def extract_source_file_code(source_files: Any) -> str:
    """For Excel source_files, retain only the accident identifier before _content_list."""
    if isinstance(source_files, str):
        sources = [source_files]
    elif isinstance(source_files, list):
        sources = source_files
    else:
        sources = []

    codes = []
    for source in sources:
        match = re.search(r"(?:^|[/\\])(\d+)_content_list", str(source))
        if not match:
            match = re.search(r"(\d+)_content_list", str(source))
        if match and match.group(1) not in codes:
            codes.append(match.group(1))

    return ", ".join(codes)


def normalize_excel_cell_value(value: Any) -> Any:
    """Represent empty extracted values as Not mentioned in the final Excel output."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return "Not mentioned"
    return value


def normalize_count_output(value: Any, field_name: str) -> Any:
    """Normalize personnel counts while retaining Not mentioned for missing evidence."""
    if value is None:
        return value

    text = str(value).strip()
    if not text:
        return text

    lower = text.lower()
    if lower in {"not mentioned", "nan", "null", "n/a"}:
        return text
    field = field_name.lower()
    if lower in {"none", "none reported"}:
        if "passenger_count" in field:
            return "Not mentioned"
        return "0"

    normalized = re.sub(r"(?<=\d),(?=\d{3}\b)", "", lower)
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20,
    }
    for word, num in word_to_num.items():
        normalized = re.sub(rf"\b{word}\b", str(num), normalized)

    if "crew_complement" in field:
        if re.search(r"\b(no crew|no crewmembers|unmanned)\b", normalized):
            return "0"
        role_pattern = (
            r"(\d+)\s*"
            r"(?:crew(?:members?)?|crewmembers?|pilots?|captains?|mates?|"
            r"engineers?|deckhands?|officers?|company representatives?|representatives?)"
        )
    elif "passenger_count" in field:
        if re.search(r"\b(no passengers?|no guests?)\b", normalized):
            return "0"
        role_pattern = r"(\d+)\s*(?:passengers?|guests?|noncrewmember guests?|non-crew guests?)"
    else:
        return text

    role_numbers = [int(n) for n in re.findall(role_pattern, normalized)]
    if role_numbers:
        return str(sum(role_numbers))

    if "crew_complement" in field:
        before_numbers = [
            int(n) for n in re.findall(
                r"(?:crew(?:members?)?|crewmembers?|pilots?|captains?|mates?|"
                r"engineers?|deckhands?|officers?)\s*(\d+)",
                normalized,
            )
        ]
        if before_numbers:
            return str(sum(before_numbers))
    elif "passenger_count" in field:
        before_numbers = [int(n) for n in re.findall(r"(?:passengers?|guests?)\s*(\d+)", normalized)]
        if before_numbers:
            return str(sum(before_numbers))

    if re.fullmatch(r"\d+", normalized):
        return normalized

    leading_number = re.match(r"^\s*(\d+)\b", normalized)
    if leading_number and not re.search(r"\b(ft|feet|meter|metre|ton|tons|year|years|mph|knots?)\b", normalized):
        return leading_number.group(1)

    return text


def normalize_pollution_output(value: Any) -> Any:
    """Normalize explicit absence of pollution to None reported for this field only."""
    if value is None:
        return value

    text = str(value).strip()
    if not text:
        return text

    lower = text.lower()
    if lower in {"not mentioned", "nan", "null", "n/a"}:
        return text

    no_pollution_patterns = [
        r"^\s*none\s*$",
        r"^\s*none reported\s*$",
        r"\bno (?:reported )?(?:pollution|environmental damage|water pollution)\b",
        r"\bno product was released\b",
        r"\bno pollution resulted\b",
        r"\bno water pollution resulted\b",
        r"\bnone observed\b",
    ]
    if any(re.search(pattern, lower) for pattern in no_pollution_patterns):
        return "None reported"

    return text


def normalize_property_damage_output(value: Any) -> Any:
    """Clean vessel damage amounts, including stray trailing dollar signs."""
    if value is None:
        return value
    text = str(value).strip()
    if not text:
        return text
    text = re.sub(r"^\$(\d[\d,]*(?:\.\d+)?)\$$", r"$\1", text)
    return text


def iter_content_list_candidate_paths(identifier: Optional[str]) -> List[Path]:
    """Find content_list JSON files in Input and any directories configured by environment variables."""
    if not identifier:
        return []

    file_name = f"{identifier}_content_list.json"
    project_dir = Path(__file__).resolve().parent
    candidates = [project_dir / "Input" / file_name]

    env_dirs = os.environ.get("MARITIME_EXTRA_INPUT_DIRS", "")
    for raw_dir in re.split(r"[:;]", env_dirs):
        raw_dir = raw_dir.strip()
        if raw_dir:
            candidates.append(Path(raw_dir) / file_name)

    seen = set()
    unique_candidates = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(path)
    return unique_candidates


def find_content_list_path(identifier: Optional[str]) -> Optional[Path]:
    for path in iter_content_list_candidate_paths(identifier):
        if path.exists():
            return path
    return None


def choose_fullest_involved_vessels_from_input(identifier: Optional[str]) -> List[str]:
    if not identifier:
        return []

    input_path = find_content_list_path(identifier)
    if not input_path:
        return []

    try:
        nodes = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    best: List[str] = []
    for node in nodes if isinstance(nodes, list) else []:
        if not isinstance(node, dict):
            continue
        html_text = (
            node.get("table_body", "")
            or node.get("html", "")
            or node.get("table_html", "")
            or node.get("text", "")
        )
        names = extract_involved_vessels_from_rows(rows_from_html_table(html_text))
        if len(names) > len(best):
            best = names

    return best


def load_input_nodes(identifier: Optional[str]) -> List[Dict]:
    if not identifier:
        return []

    input_path = find_content_list_path(identifier)
    if not input_path:
        return []

    try:
        nodes = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    return nodes if isinstance(nodes, list) else []


def normalize_table_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9/]+", " ", normalize_table_text(value).lower()).strip()


def parse_owner_operator(value: Any) -> Tuple[Optional[str], Optional[str]]:
    text = normalize_table_text(value)
    if not text or text.upper() in {"NA", "N/A"}:
        return None, None
    if "/" in text:
        owner, operator = text.split("/", 1)
        return owner.strip(" ,") or None, operator.strip(" ,") or None
    company_split = re.search(r"\b(?:Company|Co\.)\s*(?=[A-Z])", text)
    if company_split:
        split_at = company_split.end()
        owner = text[:split_at].strip(" ,")
        operator = text[split_at:].strip(" ,")
        return owner or None, operator or None
    return text, None


def canonical_vessel_detail_field(label: str) -> Optional[str]:
    """Map source vessel table row labels to schema field names."""
    label = normalize_table_label(label)
    if not label:
        return None

    if label in {"type", "vessel type", "vessel/service", "vessel service", "service"}:
        return "vessel_type"
    if label in {"property damage", "damage", "damages", "vessel damage"}:
        return "property_damage"
    if label in {"length", "length overall", "loa", "vessel length"} or "length" in label:
        return "ship_length"
    if (
        label in {"tonnage", "gross tonnage", "gross tons", "gt", "grt", "itc"}
        or "tonnage" in label
        or "gross ton" in label
        or label.endswith(" grt")
    ):
        return "ship_tonnage"
    if label in {"year built", "built", "date built", "construction year"} or "year built" in label:
        return "vessel_built_year"
    if label in {"flag", "flag state", "registry", "country"} or "flag" in label:
        return "flag_state"
    if label in {"crew", "crew complement", "crewmembers", "persons on board"} or "crew" in label:
        return "crew_complement"
    if label in {"passengers", "passenger count", "guests"} or "passenger" in label:
        return "passenger_count"
    if label.startswith("mmsi") or label in {"mmsi number"}:
        return "mmsi_or_imo"
    return None


def extract_vessel_detail_table_from_input(identifier: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Extract vessel particulars from horizontal Input tables for final field recovery."""
    best: Dict[str, Dict[str, Any]] = {}
    for node in load_input_nodes(identifier):
        html_text = (
            node.get("table_body", "")
            or node.get("html", "")
            or node.get("table_html", "")
            or node.get("text", "")
        )
        rows = rows_from_html_table(html_text)
        if not rows:
            continue

        vessel_row = None
        for row in rows:
            if len(row) > 2 and normalize_table_label(row[0]) in {"vessel", "vessels"}:
                vessel_row = row
                break
        if not vessel_row:
            continue

        vessel_names = [normalize_table_text(cell) for cell in vessel_row[1:] if normalize_table_text(cell)]
        if len(vessel_names) < 2 or len(vessel_names) <= len(best):
            continue

        details = {name: {} for name in vessel_names}
        for row in rows:
            if len(row) < 2:
                continue
            label = normalize_table_label(row[0])
            values = list(row[1:])
            for idx, name in enumerate(vessel_names):
                if idx >= len(values):
                    continue
                value = normalize_table_text(values[idx])
                # MinerU may omit an empty owner/operator cell, shifting later vessel values right.
                if not value and label == "owner/operator" and idx + 1 < len(values):
                    shifted_value = normalize_table_text(values[idx + 1])
                    if re.search(r"\b(?:Company|Co\.)\s*(?=[A-Z])", shifted_value):
                        value = shifted_value
                if not value:
                    continue
                if label == "owner/operator":
                    owner, operator = parse_owner_operator(value)
                    if owner:
                        details[name]["owner"] = owner
                    if operator:
                        details[name]["operator"] = operator
                elif label.startswith("imo number") or label.startswith("mo number"):
                    details[name]["imo_number"] = value
                elif label.startswith("official number us") or label.startswith("officialnumber us"):
                    details[name]["official_number_us"] = value
                else:
                    field = canonical_vessel_detail_field(label)
                    if field:
                        details[name][field] = value

        best = details

    return best


def find_input_vessel_detail(vessel_name: Any, details: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    key = normalize_simple_name(vessel_name)
    if not key:
        return {}
    for name, detail in details.items():
        if normalize_simple_name(name) == key:
            return detail
    return {}


def ordered_vessel_detail_items(identifier: Optional[str], details: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Reorder table vessel records to match the first-page summary when MinerU reverses columns."""
    if not details:
        return []
    ordered: List[Tuple[str, Dict[str, Any]]] = []
    used = set()
    for involved_name in choose_fullest_involved_vessels_from_input(identifier):
        involved_key = normalize_simple_name(involved_name)
        if not involved_key:
            continue
        for detail_name, detail in details.items():
            detail_key = normalize_simple_name(detail_name)
            if detail_key in used:
                continue
            if detail_key == involved_key or detail_key in involved_key or involved_key in detail_key:
                ordered.append((normalize_table_text(involved_name) or detail_name, detail))
                used.add(detail_key)
                break
    for detail_name, detail in details.items():
        detail_key = normalize_simple_name(detail_name)
        if detail_key not in used:
            ordered.append((detail_name, detail))
            used.add(detail_key)
    return ordered


def extract_vessel_no_details_from_input(identifier: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Recover core vessel details from early NTSB summaries using Vessel No. 1/2 labels."""
    text = input_plain_text(identifier)
    if not text:
        return {}
    details: Dict[str, Dict[str, Any]] = {}
    pattern = r"Vessel\s+No\.\s*(\d+)\s*:\s*(.+?)(?=Vessel\s+No\.\s*\d+\s*:|Accident\s+Type:|Location:|Date:|Time:|Owner:|Property\s+Damage:|$)"
    for _, block in re.findall(pattern, text, flags=re.I):
        block = normalize_table_text(block)
        if not block:
            continue
        name = None
        if re.search(r"\bAdvantage\b", block, flags=re.I):
            name = "Advantage"
        elif re.search(r"\bBayliner\b", block, flags=re.I):
            name = "Bayliner"
        else:
            name = re.split(r",", block, 1)[0].strip()
        if not name:
            continue
        detail: Dict[str, Any] = {}
        type_match = re.search(
            r"(?:\d+[’'′]?\s*\d*[”\"′]?\s*)?([^,.]{0,80}?(?:motorboat|sailboat|fishing vessel|towing vessel|barge|freighter|cutter|lifeboat))",
            block,
            flags=re.I,
        )
        if type_match:
            detail["vessel_type"] = normalize_table_text(type_match.group(1))
        built_match = re.search(r"\bbuilt in\s+(\d{4})\b", block, flags=re.I)
        if built_match:
            detail["vessel_built_year"] = built_match.group(1)
        details[name] = detail

    owner_match = re.search(r"Owner:\s+(.+?)\s+Property\s+Damage:", text, flags=re.I)
    if owner_match:
        owner_text = normalize_table_text(owner_match.group(1))
        for name, owner in re.findall(r"([A-Za-z][A-Za-z0-9 .'-]+?)\s*-\s*([^;]+?)(?=\s+[A-Za-z][A-Za-z0-9 .'-]+?\s*-|$)", owner_text):
            name_key = normalize_simple_name(name)
            for detail_name, detail in details.items():
                detail_key = normalize_simple_name(detail_name)
                if name_key and (name_key in detail_key or detail_key in name_key):
                    owner_value = normalize_table_text(owner).split(",", 1)[0]
                    if owner_value:
                        detail["owner"] = owner_value
                        detail["operator"] = owner_value
                    break
    return details


def input_plain_text(identifier: Optional[str]) -> str:
    """Read source text and table text from Input for deterministic validation."""
    parts: List[str] = []
    for node in load_input_nodes(identifier):
        if not isinstance(node, dict):
            continue
        for key in ["table_body", "html", "table_html", "text"]:
            text = node.get(key, "")
            if text:
                parts.append(normalize_table_text(text))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def extract_summary_property_damage_from_input(identifier: Optional[str]) -> Optional[str]:
    """Extract total property damage from the accident summary table."""
    text = normalize_money_markup(input_plain_text(identifier))
    if not text:
        return None

    # Search a short window near the opening summary label to avoid unrelated body text.
    head_text = text[:5000]
    for label_match in re.finditer(r"\b(?:Property\s+damage|Property\s+Damage|Property\s+Damage:|Damage)\b", head_text, flags=re.I):
        snippet = normalize_money_markup(head_text[label_match.start(): label_match.start() + 700])
        stop_match = re.search(r"\b(?:Weather|Waterway|Injuries|Complement|Accident Synopsis|Synopsis)\b", snippet, flags=re.I)
        if stop_match:
            snippet = snippet[:stop_match.start()]

        # Handle displaced labels such as "Property damage Environmental $957,000 est. damage None reported".
        env_money = re.search(
            r"Property\s+damage\s+Environmental\s+(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|thousand|M|K))?(?:\s*(?:est\.?|estimated))?)\s+damage",
            snippet,
            flags=re.I,
        )
        if env_money:
            return normalize_money_markup(env_money.group(1))

        money = extract_money_expression(snippet)
        if money and not is_pollution_like_damage_value(money):
            return re.sub(r"^(?:None|No(?:ne)? reported)\s+(?=\$)", "", money, flags=re.I).strip()

        qualitative = re.search(r"\b(?:total loss|constructive total loss|minor damage|substantial damage)\b", snippet, flags=re.I)
        if qualitative:
            return normalize_table_text(qualitative.group(0))

    for pattern in [
        r"vessel,\s+valued at\s+(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|thousand|M|K))?).{0,80}(?:constructive total loss|total loss)",
        r"estimated value was\s+(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|thousand|M|K))?)",
        r"valued at\s+(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|thousand|M|K))?).{0,80}(?:constructive total loss|total loss)",
    ]:
        match = re.search(pattern, head_text, flags=re.I)
        if match:
            return normalize_money_markup(match.group(1))

    return None


def extract_property_damage_pairs_from_input(identifier: Optional[str]) -> Dict[str, str]:
    """Extract per-vessel damage in "vessel/object: amount" summary entries."""
    text = normalize_money_markup(input_plain_text(identifier))
    if not text:
        return {}

    snippets = []
    head = text[:1800]
    snippets.append(head)
    for pattern in [
        r"Property\s+Damage:?\s+(.+?)(?:Injuries|Complement|Environmental|Weather|Waterway|Synopsis|$)",
        r"Property\s+damage\s+(.+?)(?:Injuries|Environmental|Weather|Waterway|$)",
        r"Damage:?\s+(.+?)(?:Injuries|Complement|Environmental|Weather|Waterway|Synopsis|$)",
    ]:
        match = re.search(pattern, text, flags=re.I)
        if match:
            snippets.append(normalize_table_text(match.group(1)))

    pairs: Dict[str, str] = {}
    for snippet in snippets:
        pair_pattern = (
            r"([A-Za-z0-9][A-Za-z0-9 .’'&/\\-]{1,80}?)"
            r"\s*(?:-|:)\s*"
            r"(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|thousand|M|K))?(?:\s*(?:est\\.?|estimated))?)"
        )
        for name, value in re.findall(pair_pattern, snippet, flags=re.I):
            clean_name = normalize_table_text(name)
            # Keep the last object name after a colon, date, or previous field to remove joined preceding text.
            clean_name = re.split(r"\b(?:Date|Time|Location|Injuries|Damage|Property Damage|Complement)\b", clean_name, flags=re.I)[-1]
            clean_name = re.sub(r"^.*?\b(?=[A-Z][A-Za-z0-9 .'&/-]{1,40}$)", "", clean_name).strip()
            clean_name = re.sub(r"^(?:and|the)\s+", "", clean_name, flags=re.I).strip()
            if clean_name and value:
                pairs[normalize_simple_name(clean_name)] = normalize_money_markup(value)
    return pairs


def has_shore_damage_context(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    shore_pattern = r"\b(?:terminal|facility|wharf|berth|pier|dock|dolphin|conveyor|gallery|shore|pipeline)\b"
    damage_pattern = r"\b(?:damage|damaged|repair|repairs|loss|lost|estimated|cost|costs|valued|destroyed)\b"
    for sentence in re.split(r"[.。;；]\s*", normalize_money_markup(text)):
        lower = sentence.lower()
        if not re.search(shore_pattern, lower):
            continue
        if extract_money_expression(sentence) or re.search(damage_pattern, lower):
            return True
    return False


def vessel_mentioned_near_damage(text: Any, vessel_name: Any) -> bool:
    """Require the target vessel name near damage, repair, loss, or an amount for vessel-level attribution."""
    if not isinstance(text, str) or not vessel_name:
        return False
    vessel_key = normalize_simple_name(vessel_name)
    if not vessel_key:
        return False
    raw_text = normalize_money_markup(text)
    for sentence in re.split(r"[.。;；]\s*", raw_text):
        if vessel_key in normalize_simple_name(sentence) and (
            extract_money_expression(sentence)
            or re.search(r"\b(?:minor damage|substantial damage|total loss|constructive total loss)\b", sentence, flags=re.I)
            or re.search(r"\b(?:damage|damaged|repair|repairs|salvage|loss|lost|value)\b", sentence, flags=re.I)
        ):
            return True
    return False


def extract_barge_property_damage_from_input(identifier: Optional[str]) -> Optional[str]:
    """Extract collective barge losses when individual barges are not named."""
    text = normalize_money_markup(input_plain_text(identifier))
    if not text:
        return None
    patterns = [
        r"(?:damage\s+(?:cost\s+)?to\s+the\s+barges?|costs?\s+to\s+(?:repair|replace)\s+the\s+barges?|"
        r"barges?\s+(?:sustained|incurred|received|had)\s+damage)"
        r"[^.。;；]{0,160}",
        r"[^.。;；]{0,120}\bbarges?\b[^.。;；]{0,80}\b(?:damage|repair|loss)\b[^.。;；]{0,120}",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            sentence = normalize_table_text(match.group(0))
            if has_shore_damage_context(sentence):
                continue
            money = extract_money_expression(sentence)
            if money:
                return money
    return None


def should_summary_damage_apply_to_vessel(summary: Any, raw_text: str, vessel: Dict, single_core_vessel: bool = False) -> bool:
    """Check whether the summary property damage total can support a core vessel's damage field."""
    if not isinstance(summary, str) or not summary.strip() or is_pollution_like_damage_value(summary):
        return False

    vessel_name = get_metadata_value(vessel.get("Vessel Name")) if isinstance(vessel, dict) else None
    vessel_type = get_metadata_value(vessel.get("vessel_type")) if isinstance(vessel, dict) else None
    name_key = normalize_simple_name(vessel_name)
    combined = f"{vessel_name or ''} {vessel_type or ''}".lower()
    lower = f"{summary} {raw_text[:2500]}".lower()

    shore_terms = [
        "terminal", "facility", "wharf", "berth", "pier", "dock", "dolphin",
        "conveyor", "gallery", "shore", "pipeline"
    ]
    has_shore_context = has_shore_damage_context(lower)
    if has_shore_context:
        # Do not assign shore, pipeline, or pier damage to a vessel from a summary total alone.
        # Require a nearby vessel name or explicit vessel/barge damage wording.
        if name_key and vessel_mentioned_near_damage(lower, vessel_name):
            return True
        if "barge" in combined and re.search(
            r"\b(?:damage (?:cost )?to the barges?|barges? (?:sustained|incurred|received|had) damage)\b",
            lower,
        ):
            return True
        return False

    if name_key and vessel_mentioned_near_damage(lower, vessel_name):
        return True
    if re.search(r"\b(?:vessel and (?:the )?barges|vessels? and barges?|both vessels|all vessels|both vessels combined)\b", lower):
        return True
    if "barge" in combined and re.search(r"\bbarges?\b", lower):
        return True
    if "barge" not in combined and re.search(r"\b(?:barge|barges)\b[^.。;；]{0,120}\b(?:damage|damaged|loss|repair|estimated)\b", lower):
        return False
    if re.search(r"\b(?:total loss|constructive total loss|estimated total loss|damage to the vessel|vessel damage)\b", lower):
        return True
    if re.search(r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|thousand|M|K))?", str(summary), flags=re.I):
        # A single summary amount without shore or third-party damage context may cover core vessel losses.
        # Use it only as an "Included in total ..." fallback.
        return single_core_vessel

    return False


def is_shore_or_third_party_damage_context(text: Any, vessel_name: Any = None) -> bool:
    """Identify damage to piers, berths, shore facilities, pipelines, and other non-vessel assets."""
    if not isinstance(text, str):
        return False
    lower = text.lower()
    shore_terms = [
        "terminal", "facility", "wharf", "berth", "pier", "dock", "dolphin",
        "conveyor", "gallery", "shore", "pipeline"
    ]
    if not any(term in lower for term in shore_terms):
        return False
    vessel_key = normalize_simple_name(vessel_name)
    return not (vessel_key and vessel_key in normalize_simple_name(text))


def damage_sentence_targets_other_object(sentence: Any, vessel_name: Any = None, vessel_type: Any = None) -> bool:
    """Reject damage attributed to another vessel or object, such as a barge or pier."""
    if not isinstance(sentence, str):
        return False
    lower = sentence.lower()
    current = f"{vessel_name or ''} {vessel_type or ''}".lower()
    current_key = normalize_simple_name(vessel_name)
    is_current_barge = "barge" in current
    if not is_current_barge and re.search(r"\b(?:damage|damaged|loss|repair|estimated)[^.。;；]{0,100}\bbarges?\b|\bbarges?\b[^.。;；]{0,100}\b(?:damage|damaged|loss|repair|estimated)\b", lower):
        return True
    target_match = re.search(r"\bdamage to\s+(?:the\s+)?([A-Z][A-Za-z0-9 .’'&/-]{2,80}?)(?:\s+was|\s+were|\s+is|\s+estimated|,|;|\\.|$)", sentence)
    if target_match:
        target_key = normalize_simple_name(target_match.group(1))
        if target_key and current_key and current_key not in target_key and target_key not in current_key:
            return True
    return False


def extract_summary_environmental_damage_from_input(identifier: Optional[str]) -> Optional[str]:
    """Extract pollution or environmental damage from the summary table."""
    text = input_plain_text(identifier)
    if not text:
        return None

    match = re.search(
        r"Environmental\s+damage\s+(.+?)(?:Weather|Waterway|$)",
        text,
        flags=re.I,
    )
    if not match:
        return None

    summary = normalize_table_text(match.group(1))
    return summary or None


def extract_money_expression(text: str) -> Optional[str]:
    if not text:
        return None
    text = normalize_money_markup(text)
    money_patterns = [
        r"(?:greater than|more than|over|about|approximately|nearly|estimated at|est\.?|between)?\s*\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand|M|K)?(?:\s*(?:and|to|–|-)\s*\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand|M|K)?)?\s*(?:est\.?|estimated)?",
        r"\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand|M|K)?",
    ]
    for pattern in money_patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            money = normalize_table_text(match.group(0))
            money = re.sub(r"^(?:estimated at|est\.?\s*)\s+", "", money, flags=re.I).strip()
            return money
    return None


def extract_vessel_amount_from_summary(summary: Any) -> Optional[str]:
    """Prefer amounts explicitly attributed to a vessel, ship, or barge in the loss summary."""
    if not isinstance(summary, str):
        return None
    text = normalize_money_markup(summary)
    patterns = [
        r"(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|K))?)\s*(?:for|to)\s+(?:the\s+)?(?:vessel|ship|barge)\b",
        r"(?:vessel|ship|barge)\s*(?:damage|loss|repairs?|repair costs?|total loss)?[^$]{0,60}(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|K))?)",
        r"\((\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|K))?)\s+for\s+(?:vessel|ship|barge)",
        r"(?:est\.?|estimated)\s*(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|K))?)\s+to\s+vessel\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return normalize_money_markup(match.group(1))
    return None


def has_vessel_damage_context(text: Any) -> bool:
    """Check whether an amount represents vessel damage rather than only the accident total."""
    if not isinstance(text, str):
        return False
    lower = text.lower()
    vessel_terms = [
        "damage to the vessel", "damage to vessel", "to the vessel", "to its vessel",
        "vessel was estimated", "vessel were", "vessel repairs", "repair the vessel",
        "salvage and repair", "salvage and repair the vessel", "vessel loss",
        "loss of vessel", "total loss of vessel", "constructive total loss",
        "total constructive loss", "estimated vessel value", "vessel value",
        "estimated total loss", "total loss", "minor damage", "substantial damage",
        "to the connor bordelon",
    ]
    return any(term in lower for term in vessel_terms)


def is_pollution_like_damage_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lower = value.lower()
    return bool(re.search(r"\b(?:environmental damage|gallons?|diesel|fuel|lube oil|oil sheen|pollution|released|discharged)\b", lower))


def is_generic_vessel_name(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    text = value.strip().lower()
    return text in {"", "not mentioned", "none", "null", "vessel", "ship", "tug", "barge", "tow"}


def is_missing_extracted_value(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"", "not mentioned", "none", "null", "n/a", "nan"}


def normalize_count_number(value: Any) -> Optional[int]:
    """Normalize a personnel count to an integer, or return None if it cannot be resolved."""
    if value is None:
        return None
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(value).strip().lower())
    if not text or text in {"not mentioned", "none", "null", "n/a", "nan"}:
        return None
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20,
    }
    for word, num in word_to_num.items():
        text = re.sub(rf"\b{word}\b", str(num), text)
    numbers = re.findall(r"\d+", text)
    return int(numbers[0]) if numbers else None


def is_passenger_service_vessel(vessel_type: Any) -> bool:
    text = str(vessel_type or "").lower()
    return bool(re.search(r"\b(?:passenger|ferry|cruise|tour|excursion)\b", text))


def is_non_passenger_work_or_cargo_vessel(vessel_type: Any, vessel_name: Any = None) -> bool:
    combined = f"{vessel_type or ''} {vessel_name or ''}".lower()
    if is_passenger_service_vessel(combined):
        return False
    return bool(re.search(
        r"\b(?:container|containership|bulk|carrier|cargo|tanker|chemical|oil|"
        r"towing|towboat|tug|barge|fishing|trawler|seiner|workboat|"
        r"supply vessel|offshore supply|osv|freight|freighter|deck cargo|"
        r"liftboat|dredge|research vessel|pilot boat)\b",
        combined,
    ))


def passenger_evidence_for_vessel(identifier: Optional[str], vessel_name: Any = None) -> List[str]:
    """Return passenger evidence associated with the target vessel."""
    text = input_plain_text(identifier)
    if not text:
        return []
    normalized = normalize_table_text(text)
    vessel_key = normalize_simple_name(vessel_name)
    snippets = []
    passenger_terms = r"passengers?|guests?|noncrewmember guests?|non-crew guests?"
    for sentence in re.split(r"(?<=[.。;；])\s+", normalized):
        lower = sentence.lower()
        if not re.search(passenger_terms, lower):
            continue
        if vessel_key and len(vessel_key) > 2:
            sentence_key = normalize_simple_name(sentence)
            # Short accident summaries may provide evidence without repeating the vessel name.
            if vessel_key not in sentence_key and len(sentence) > 220:
                continue
        snippets.append(sentence)
    return snippets


def extract_passenger_count_from_input(identifier: Optional[str], vessel_name: Any = None) -> Optional[str]:
    """Recover explicit passenger counts from Input, including guests and noncrewmember guests."""
    snippets = passenger_evidence_for_vessel(identifier, vessel_name)
    if not snippets:
        return None
    joined = " ".join(snippets).lower()
    joined = re.sub(r"(?<=\d),(?=\d{3}\b)", "", joined)
    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16",
        "seventeen": "17", "eighteen": "18", "nineteen": "19",
        "twenty": "20",
    }
    for word, number in word_to_num.items():
        joined = re.sub(rf"\b{word}\b", number, joined)
    if re.search(r"\bno\s+(?:passengers?|guests?)\s+(?:aboard|on board|were aboard|were on board)\b", joined):
        return "0"
    numbers = [
        int(n) for n in re.findall(
            r"(\d+)\s*(?:passengers?|guests?|noncrewmember guests?|non-crew guests?)",
            joined,
        )
    ]
    numbers += [
        int(n) for n in re.findall(
            r"(?:passengers?|guests?|noncrewmember guests?|non-crew guests?)\s*[:：]?\s*(\d+)",
            joined,
        )
    ]
    if numbers:
        return str(max(numbers))
    return None


def has_passenger_evidence(identifier: Optional[str], vessel_name: Any = None) -> bool:
    return bool(passenger_evidence_for_vessel(identifier, vessel_name))


def is_invalid_mmsi_or_imo(value: Any) -> bool:
    """Keep seven-digit IMO or nine-digit MMSI values; exclude registration and official numbers."""
    if value is None:
        return True
    text = str(value).strip()
    lower = text.lower()
    if lower in {"", "none", "not mentioned", "not applicable", "n/a", "na", "null"}:
        return True
    if re.search(r"\b(?:official|registration|registry|virgin island|vin|hull|year built)\b", lower):
        return True

    digits = re.findall(r"\d{6,9}", re.sub(r"(?<=\d),(?=\d{3}\b)", "", text))
    if not digits:
        return True
    return not any(len(number) in {7, 9} for number in digits)


def normalize_mmsi_or_imo_value(value: Any) -> Optional[str]:
    if is_invalid_mmsi_or_imo(value):
        return None
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(value))
    candidates = re.findall(r"\d{7}|\d{9}", text)
    return "; ".join(dict.fromkeys(candidates)) if candidates else None


def extract_vessel_property_damage_from_input(identifier: Optional[str], vessel_name: Any = None) -> Optional[str]:
    """Recover vessel-level property damage from the source Input document."""
    text = input_plain_text(identifier)
    if not text:
        return None
    vessel_key = normalize_simple_name(vessel_name)
    vessel_amount = extract_vessel_amount_from_summary(text[:5000])
    if vessel_amount and (not vessel_key or vessel_mentioned_near_damage(text[:5000], vessel_name) or re.search(r"\b(?:for|to)\s+(?:the\s+)?(?:vessel|ship|barge)\b", text[:5000], flags=re.I)):
        return vessel_amount

    # Prefer explicit vessel damage statements in the narrative.
    sentence_pattern = (
        r"[^.。;；]{0,180}"
        r"(?:damage to the vessel|damage to [^.。;；]{0,80} vessel|"
        r"damage to [A-Z][^.。;；]{0,80} was estimated|"
        r"costs? to salvage and repair the vessel|salvage and repair the vessel|"
        r"estimated vessel repair costs?|vessel repair costs?|"
        r"(?:sustained|incurred|received)\s+(?:nearly\s+)?\$?[^.。;；]{0,60}\s+in damage|"
        r"vessel (?:was|were)?\s*(?:estimated|valued)|"
        r"vessel was a total loss|"
        r"constructive total loss|total loss of vessel|loss of vessel|estimated vessel value)"
        r"[^.。;；]{0,220}"
    )
    for match in re.finditer(sentence_pattern, text, flags=re.I):
        sentence = normalize_table_text(match.group(0))
        if is_shore_or_third_party_damage_context(sentence, vessel_name):
            continue
        if damage_sentence_targets_other_object(sentence, vessel_name):
            continue
        if has_shore_damage_context(sentence) and not vessel_mentioned_near_damage(sentence, vessel_name):
            continue
        money = extract_money_expression(sentence)
        if money:
            return money
        qualitative = re.search(r"\b(?:minor damage|substantial damage|total loss|constructive total loss)\b", sentence, flags=re.I)
        if qualitative:
            return normalize_table_text(qualitative.group(0))

    # Accept sentences containing the target name, an amount, and damage/repair/loss wording.
    if vessel_key:
        for sentence in re.split(r"[.。;；]\s*", text):
            sentence = normalize_table_text(sentence)
            if not sentence or vessel_key not in normalize_simple_name(sentence):
                continue
            if not re.search(r"\b(?:damage|damaged|repair|repairs|salvage|loss|lost|value)\b", sentence, flags=re.I):
                continue
            if is_shore_or_third_party_damage_context(sentence, vessel_name):
                continue
            if damage_sentence_targets_other_object(sentence, vessel_name):
                continue
            if has_shore_damage_context(sentence) and not vessel_mentioned_near_damage(sentence, vessel_name):
                continue
            money = extract_money_expression(sentence)
            if money:
                return money
            qualitative = re.search(r"\b(?:minor damage|substantial damage|total loss|constructive total loss)\b", sentence, flags=re.I)
            if qualitative:
                return normalize_table_text(qualitative.group(0))

    # Accept summary damage attributed to the core vessel by name or "to the vessel" wording.
    summary_match = re.search(
        r"Property damage\s+(.+?)(?:Environmental\s+damage|Environmental|Weather|Waterway|$)",
        text,
        flags=re.I,
    )
    if summary_match:
        summary = normalize_table_text(summary_match.group(1))
        if not has_shore_damage_context(summary) and (
            has_vessel_damage_context(summary) or (vessel_key and vessel_key in normalize_simple_name(summary))
        ):
            money = extract_money_expression(summary)
            if money:
                return money

    return None


def fill_economic_loss_from_input(result: Dict):
    """Fill missing economic_loss from the summary table's Property damage field."""
    current = clean_money_text(get_metadata_value(result.get("economic_loss")))
    if not is_missing_extracted_value(current):
        result["economic_loss"] = set_metadata_value(result.get("economic_loss"), current)
        return

    identifier = extract_identifier_from_sources(result)
    summary = extract_summary_property_damage_from_input(identifier)
    if summary and not is_pollution_like_damage_value(summary):
        result["economic_loss"] = set_metadata_value(result.get("economic_loss"), summary)


def normalize_environmental_damage_to_pollution(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = normalize_table_text(value)
    lower = text.lower()
    pollution_terms = r"(pollution|environmental damage|oil sheen|diesel|fuel|lube oil|released|discharged|gallons?)"
    if len(text) > 500:
        for sentence in re.split(r"(?<=[.;])\s+", text):
            if re.search(pollution_terms, sentence, flags=re.I):
                text = normalize_table_text(sentence)
                lower = text.lower()
                break
        else:
            return None
    if lower.startswith("weather") and not re.search(pollution_terms, lower):
        return None
    if re.search(r"\bdiesel engines?\b", lower) and not re.search(r"\b(?:released|discharged|spilled|sank|leaked|pollution|environmental damage|oil sheen|gallons?)\b", lower):
        return None
    if lower in {"none", "none reported", "no pollution", "no environmental damage"}:
        return "None reported"
    if re.search(r"\b(?:no|none) (?:reported )?(?:pollution|environmental damage|water pollution)\b", lower):
        return "None reported"
    return text or None


def fill_pollution_from_input(result: Dict):
    """Fill pollution from Environmental damage, applying None reported only to pollution."""
    identifier = extract_identifier_from_sources(result)
    summary_candidate = normalize_environmental_damage_to_pollution(
        extract_summary_environmental_damage_from_input(identifier)
    )
    current = get_metadata_value(result.get("pollution"))
    if not is_missing_extracted_value(current):
        normalized = normalize_environmental_damage_to_pollution(current)
        if summary_candidate == "None reported" and (not normalized or len(str(current)) > 180):
            result["pollution"] = set_metadata_value(result.get("pollution"), summary_candidate)
            return
        if normalized:
            result["pollution"] = set_metadata_value(result.get("pollution"), normalized)
        return

    if summary_candidate:
        result["pollution"] = set_metadata_value(result.get("pollution"), summary_candidate)


def fill_vessel_details_from_input(result: Dict):
    """Recover core vessel names, types, owners, operators, and IMO values from Input tables."""
    identifier = extract_identifier_from_sources(result)
    input_details = extract_vessel_detail_table_from_input(identifier)
    text_details = extract_vessel_no_details_from_input(identifier)
    for name, detail in text_details.items():
        if normalize_simple_name(name) not in {normalize_simple_name(k) for k in input_details}:
            input_details[name] = detail
    if not input_details:
        return

    detail_items = ordered_vessel_detail_items(identifier, input_details)
    current_name_counts = {}
    for j in range(1, 4):
        current_vessel = result.get(f"vessel_{j}_info")
        if isinstance(current_vessel, dict):
            current_key = normalize_simple_name(get_metadata_value(current_vessel.get("Vessel Name")))
            if current_key and not is_missing_extracted_value(get_metadata_value(current_vessel.get("Vessel Name"))):
                current_name_counts[current_key] = current_name_counts.get(current_key, 0) + 1
    expected_keys = {normalize_simple_name(name) for name, _ in detail_items[:max(1, len(current_name_counts))]}
    current_keys = set(current_name_counts)
    should_reorder_by_expected = bool(expected_keys) and expected_keys.issubset(current_keys)
    used_detail_keys = set()
    for i in range(1, 4):
        vessel = result.get(f"vessel_{i}_info")
        if not isinstance(vessel, dict):
            continue

        current_name = get_metadata_value(vessel.get("Vessel Name"))
        detail = find_input_vessel_detail(current_name, input_details)
        detail_name = None
        if detail:
            for name, item in detail_items:
                if item is detail:
                    detail_name = name
                    break

        expected_name = None
        expected_detail = None
        if i <= len(detail_items):
            expected_name, expected_detail = detail_items[i - 1]
        if detail_name and normalize_simple_name(detail_name) in used_detail_keys:
            for fallback_name, fallback_detail in detail_items:
                if normalize_simple_name(fallback_name) not in used_detail_keys:
                    expected_name, expected_detail = fallback_name, fallback_detail
                    break

        # Replace generic tug/barge/vessel labels using core vessel names in table order.
        should_use_expected = False
        if expected_detail:
            current_key = normalize_simple_name(current_name)
            expected_key = normalize_simple_name(expected_name)
            detail_key = normalize_simple_name(detail_name)
            should_use_expected = (
                is_generic_vessel_name(current_name)
                or not detail
                or detail_key in used_detail_keys
                or current_name_counts.get(current_key, 0) > 1
                or (should_reorder_by_expected and current_key and expected_key and current_key != expected_key and expected_key not in used_detail_keys)
            )

        if should_use_expected and expected_name and expected_detail:
            detail_name, detail = expected_name, expected_detail
            vessel["Vessel Name"] = set_metadata_value(vessel.get("Vessel Name"), detail_name)

        if not detail:
            continue
        force_detail_values = should_use_expected
        if detail_name:
            used_detail_keys.add(normalize_simple_name(detail_name))

        for field in [
            "owner", "operator", "vessel_type", "ship_length", "ship_tonnage",
            "vessel_built_year", "flag_state", "crew_complement", "passenger_count"
        ]:
            current = get_metadata_value(vessel.get(field))
            candidate = detail.get(field)
            if field == "operator" and not candidate and detail.get("owner"):
                candidate = detail.get("owner")
            if candidate and (force_detail_values or is_missing_extracted_value(current) or is_generic_vessel_name(current)):
                vessel[field] = set_metadata_value(vessel.get(field), candidate)

        current_imo = get_metadata_value(vessel.get("mmsi_or_imo"))
        imo_candidate = normalize_mmsi_or_imo_value(detail.get("imo_number") or detail.get("mmsi_or_imo"))
        if imo_candidate and (force_detail_values or is_invalid_mmsi_or_imo(current_imo)):
            vessel["mmsi_or_imo"] = set_metadata_value(vessel.get("mmsi_or_imo"), imo_candidate)

        current_damage = clean_money_text(get_metadata_value(vessel.get("property_damage")))
        detail_damage = clean_money_text(detail.get("property_damage"))
        if detail_damage and is_missing_extracted_value(current_damage) and not is_pollution_like_damage_value(detail_damage):
            vessel["property_damage"] = set_metadata_value(vessel.get("property_damage"), detail_damage)


def cleanup_and_fill_passenger_count(result: Dict):
    """Correct passenger_count, distinguishing missing evidence from an explicit zero."""
    identifier = extract_identifier_from_sources(result)
    for i in range(1, 4):
        vessel = result.get(f"vessel_{i}_info")
        if not isinstance(vessel, dict):
            continue

        vessel_name = get_metadata_value(vessel.get("Vessel Name"))
        vessel_type = get_metadata_value(vessel.get("vessel_type"))
        crew_value = get_metadata_value(vessel.get("crew_complement"))
        current = get_metadata_value(vessel.get("passenger_count"))
        current_number = normalize_count_number(current)
        crew_number = normalize_count_number(crew_value)
        candidate = extract_passenger_count_from_input(identifier, vessel_name)

        # Explicit passengers, guests, or noncrewmember guests take precedence over inferred counts.
        # They can replace a crew count incorrectly assigned to passenger_count.
        if candidate is not None:
            if str(candidate).strip() != str(current or "").strip():
                vessel["passenger_count"] = set_metadata_value(vessel.get("passenger_count"), candidate)
            continue

        # Clear crew counts assigned to passenger_count when no passenger evidence exists.
        if (
            current_number is not None
            and crew_number is not None
            and current_number > 0
            and current_number == crew_number
            and not has_passenger_evidence(identifier, vessel_name)
        ):
            vessel["passenger_count"] = set_metadata_value(vessel.get("passenger_count"), None)
            current = None

        if not is_missing_extracted_value(current):
            continue


def nonempty_core_vessel_count(result: Dict) -> int:
    count = 0
    for i in range(1, 4):
        vessel = result.get(f"vessel_{i}_info")
        if not isinstance(vessel, dict):
            continue
        name = get_metadata_value(vessel.get("Vessel Name"))
        if name and not is_missing_extracted_value(name):
            count += 1
    return count


def is_barge_or_auxiliary_vessel(name: Any = None, vessel_info: Dict = None) -> bool:
    name_text = str(name or "").strip()
    type_text = ""
    if vessel_info:
        type_text = str(get_metadata_value(vessel_info.get("vessel_type")) or "")
    combined = f"{name_text} {type_text}"
    if re.search(r"\b(?:tank|freight|deck)?\s*barge\b", combined, flags=re.I):
        return True
    if re.fullmatch(r"\d{4,}", name_text):
        return True
    return bool(re.search(r"^AEP\s*\d+|^APEX\s*\d+|^ING\s*\d+|^IB\s*\d+", name_text, flags=re.I))


def is_cause_official(cause: Dict) -> bool:
    return bool(cause.get("is_official_cause")) or "probable cause" in str(cause.get("source_chapter", "")).lower()


def cause_mentions_name(content: str, name: str) -> bool:
    if not content or not name:
        return False
    normalized_content = normalize_simple_name(content)
    normalized_name = normalize_simple_name(name)
    return bool(normalized_name and normalized_name in normalized_content)


def is_third_party_cause(content: str, vessel_name: str = "") -> bool:
    if not content:
        return False
    lower = content.lower()
    # An explicit target vessel name in a cause supports assigning it to that vessel.
    if vessel_name and cause_mentions_name(content, vessel_name):
        return False

    third_party_patterns = [
        "coast guard", "army corps", "corps of engineers",
        "shipyard", "dry dock", "dockyard", "manufacturer",
        "port authority", "terminal", "fleeting area",
        "fleeting facility", "facility owner", "sanitary authority",
        "bae systems", "vigor industrial", "regulator", "regulatory"
    ]
    return any(pattern in lower for pattern in third_party_patterns)


def is_shared_environment_or_management_cause(content: str) -> bool:
    """Return whether a cause should be assigned to all core responsible vessels."""
    if not content:
        return False
    lower = content.lower()

    shared_terms = [
        "both vessels", "all vessels", "two vessels", "each vessel",
        "both operators", "all involved parties", "joint decision", "joint failure",
        "communication between the pilots", "risk of collision", "passing room",
    ]
    management_terms = [
        "operating company", "company", "manufacturer", "shipyard", "port authority",
        "terminal", "terminal personnel", "fleeting area", "fleeting facility",
        "facility owner", "sanitary authority", "coast guard", "army corps",
        "corps of engineers", "vessel traffic service", "vts", "harbor department",
        "lockmaster", "industry practice", "regulator", "regulatory",
        "oversight", "procedures", "monitoring", "management",
    ]
    environment_terms = [
        "weather", "severe weather", "wind", "winds", "current", "river current",
        "high water", "high-water", "flood", "flooding", "ice", "visibility",
        "fog", "heavy seas", "waterway", "river", "seafloor", "underwater protrusions",
    ]

    human_specific_terms = [
        "captain", "master", "pilot", "mate", "watchstander", "officer of the watch",
        "crew", "helmsman", "lookout", "engineer", "pic",
    ]

    if any(term in lower for term in shared_terms + management_terms):
        return True

    # Share purely environmental causes; keep environmental qualifiers of specific human errors local.
    if any(term in lower for term in environment_terms):
        return not any(term in lower for term in human_specific_terms)

    return False


def has_target_vessel_cue(content: str, vessel_name: str) -> bool:
    if not content:
        return False
    if cause_mentions_name(content, vessel_name):
        return True
    lower = content.lower()
    target_role_patterns = [
        "captain", "master", "pilot", "bridge team", "crew",
        "operator", "owner", "watchstander", "officer of the watch",
        "lookout", "engineer", "helmsman", "mate",
        "propulsion", "engine", "generator", "steering", "rudder",
        "anchor", "mooring", "navigation watch", "hull", "bulkhead"
    ]
    shared_patterns = [
        "both vessels", "all vessels", "two vessels", "each vessel",
        "passing room", "risk of collision", "communication between the pilots"
    ]
    return any(pattern in lower for pattern in target_role_patterns + shared_patterns)


def clean_cause_content(content: str) -> str:
    if not content:
        return content
    text = re.sub(r"^\s*\d+\.\s*", "", str(content)).strip()
    text = re.sub(r"\s+", " ", text)

    # Keep the causal statement and remove appended accident consequences.
    led_match = re.search(
        r"\b(?:led to|leading to|resulting in)\b\s+(?:the\s+)?(?:fire|destruction|damage|sinking|flooding|grounding|collision)\b",
        text,
        flags=re.I
    )
    if led_match and led_match.start() > 12:
        text = text[:led_match.start()].strip(" ,.;")

    return text


def cause_relevant_to_vessel(cause: Dict, vessel_name: str) -> bool:
    content = str(cause.get("content", ""))
    lower = content.lower()
    known_names = ["miss dixie", "d r boney", "dewey r", "p b shah"]
    current_key = normalize_simple_name(vessel_name)

    mentioned = [name for name in known_names if normalize_simple_name(name) in normalize_simple_name(content)]
    if mentioned and all(normalize_simple_name(name) not in current_key for name in mentioned):
        return False

    # Keep Miss Dixie's maintenance, clutch, and propulsion causes separate from D.& R. Boney and barges.
    miss_dixie_terms = [
        "clutch", "port propeller", "maintenance program", "engine room", "propulsion",
        "periodic inspection", "maintenance procedures", "manufacturer's guidance",
        "manufacturer guidance", "undetected wear"
    ]
    if any(term in lower for term in miss_dixie_terms) and "missdixie" not in current_key:
        return False

    # Keep human factors attributed to P. B. Shah separate from Dewey R.
    if ("p. b. shah" in lower or "p b shah" in lower or "p.b. shah" in lower) and "pbshah" not in current_key:
        return False
    if "dewey" in lower and "dewey" not in current_key:
        return False

    if not has_target_vessel_cue(content, vessel_name) and not is_cause_official(cause):
        return False

    return True


def postprocess_causes_for_vessel(causes_data: Dict, vessel_name: str) -> Dict:
    if not causes_data:
        return {"causes": [], "total_causes": 0}

    causes = [
        cause for cause in causes_data.get("causes", [])
        if cause_relevant_to_vessel(cause, vessel_name)
    ]
    official_causes = [cause for cause in causes if is_cause_official(cause)]
    if official_causes:
        causes = official_causes

    # Exclude observations or evidence statements that do not describe a cause.
    evidence_patterns = [
        "smoke in the area", "smelled rubber", "observed smoke",
        "postaccident inspection", "after the accident"
    ]
    filtered = []
    for cause in causes:
        text = str(cause.get("content", "")).lower()
        if any(pattern in text for pattern in evidence_patterns) and not is_cause_official(cause):
            continue
        cause = copy.deepcopy(cause)
        cause["content"] = clean_cause_content(cause.get("content", ""))
        if not cause["content"]:
            continue
        filtered.append(cause)

    seen = set()
    unique = []
    for cause in filtered:
        key = re.sub(r"\s+", " ", str(cause.get("content", "")).lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(cause)

    return {"causes": unique, "total_causes": len(unique)}


def propagate_shared_causes_to_all_vessels(vessel_causes: Dict[str, Dict]) -> Dict[str, Dict]:
    """Share environmental, management, regulatory, third-party, and joint human causes.
    
    Recover shared causes assigned to only one vessel by the model. Preserve
    the attribution of causes tied to specific people or vessel equipment.
    """
    if not vessel_causes:
        return vessel_causes

    shared_causes = []
    for causes_data in vessel_causes.values():
        for cause in causes_data.get("causes", []) if causes_data else []:
            content = str(cause.get("content", ""))
            if is_shared_environment_or_management_cause(content):
                shared_causes.append(cause)

    if not shared_causes:
        return vessel_causes

    for key, causes_data in vessel_causes.items():
        if not causes_data:
            causes_data = {"causes": [], "total_causes": 0}
            vessel_causes[key] = causes_data
        causes_list = causes_data.setdefault("causes", [])
        seen = {
            re.sub(r"\s+", " ", str(c.get("content", "")).lower()).strip()
            for c in causes_list
        }
        for cause in shared_causes:
            cause_key = re.sub(r"\s+", " ", str(cause.get("content", "")).lower()).strip()
            if not cause_key or cause_key in seen:
                continue
            copied = copy.deepcopy(cause)
            copied["shared_cause_propagated"] = True
            causes_list.append(copied)
            seen.add(cause_key)
        causes_data["total_causes"] = len(causes_list)

    return vessel_causes


def extract_probable_cause_from_input(identifier: Optional[str]) -> Optional[str]:
    """Recover the Probable Cause section from the full Input document."""
    text = input_plain_text(identifier)
    if not text:
        return None

    match = re.search(
        r"Probable Cause\s+(.+?)(?:Vessel Particulars|For more details|Adopted:|$)",
        text,
        flags=re.I,
    )
    if not match:
        return None

    cause = normalize_table_text(match.group(1))
    cause = re.sub(r"^The National Transportation Safety Board determines that the probable cause(?:s)? of .*? (?:was|were)\s+", "", cause, flags=re.I)
    cause = re.sub(r"^The probable cause(?:s)? of .*? (?:was|were)\s+", "", cause, flags=re.I)
    cause = re.sub(r"\s+", " ", cause).strip(" .;")
    if len(cause) < 12:
        return None
    return cause


def fill_missing_causes_from_input(result: Dict):
    """Use the Probable Cause section if vessel_1 causes are entirely missing."""
    causes_data = result.get("causes_for_vessel_1")
    if causes_data and causes_data.get("causes"):
        return

    identifier = extract_identifier_from_sources(result)
    cause_text = extract_probable_cause_from_input(identifier)
    if not cause_text:
        return

    cause = {
        "content": clean_cause_content(cause_text),
        "source_chapter": "Probable Cause",
        "is_official_cause": True,
    }
    result["causes_for_vessel_1"] = {"causes": [cause], "total_causes": 1}


def get_involved_vessel_list(json_data: Dict) -> List[str]:
    involved = get_metadata_value(json_data.get("involved_vessels")) if json_data else None
    if isinstance(involved, list):
        return dedupe_names(involved)

    vessels = collect_vessels_from_json(json_data or {})
    return dedupe_names([
        get_metadata_value(info.get("Vessel Name"))
        for info in vessels.values()
        if get_metadata_value(info.get("Vessel Name"))
    ])


def choose_fullest_involved_vessels(json1: Dict, json2: Dict) -> List[str]:
    list1 = get_involved_vessel_list(json1)
    list2 = get_involved_vessel_list(json2)
    best = list1 if len(list1) >= len(list2) else list2
    input_best = choose_fullest_involved_vessels_from_input(extract_identifier_from_sources(json1, json2))
    return input_best if len(input_best) > len(best) else best


def clean_money_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return re.sub(r"^\s*(?:None|No(?:ne)? reported)\s+(?=\$)", "", value, flags=re.I).strip()


def normalize_compare_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9$.,]+", "", text)
    return text.strip()


def clean_accident_time(value: Any) -> Any:
    """Restore date/time spacing lost in table extraction, such as September2,2015,1959."""
    if not isinstance(value, str):
        return value

    months = (
        "January|February|March|April|May|June|July|August|"
        "September|October|November|December"
    )
    text = re.sub(r"\s+", " ", value).strip()
    text = re.sub(
        rf"\b({months})\s*(\d{{1,2}})\s*,\s*(\d{{4}})\s*,?\s*",
        r"\1 \2, \3, ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def is_suspicious_owner_operator(value: Any) -> bool:
    """Detect owner/operator fragments displaced by MinerU or table column shifts."""
    if not isinstance(value, str):
        return False

    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return False

    lower = text.lower()
    suspicious_patterns = [
        "american commercial american marine",
        "commercial american marine",
        "jrc american commercial",
        "barge line / inland",
        "barge line inland",
        "association cincinnati",
        "line st. louis",
    ]
    if any(pattern in lower for pattern in suspicious_patterns):
        return True

    # An owner/operator value should not consist only of an industry or location suffix.
    if re.fullmatch(r"(?:barge line|inland|marine|association|cincinnati|st\.?\s*louis)(?:\s*/\s*(?:inland|marine))?", lower):
        return True

    return False


def cleanup_vessel_owner_operator(result: Dict):
    for i in range(1, 4):
        vessel = result.get(f"vessel_{i}_info")
        if not isinstance(vessel, dict):
            continue
        for field in ["owner", "operator"]:
            if is_suspicious_owner_operator(get_metadata_value(vessel.get(field))):
                vessel[field] = set_metadata_value(vessel.get(field), None)


def cleanup_vessel_mmsi_or_imo(result: Dict):
    """Keep only MMSI/IMO identifiers in mmsi_or_imo; exclude US official numbers."""
    identifier = extract_identifier_from_sources(result)
    input_details = extract_vessel_detail_table_from_input(identifier)

    for i in range(1, 4):
        vessel = result.get(f"vessel_{i}_info")
        if not isinstance(vessel, dict):
            continue

        vessel_name = get_metadata_value(vessel.get("Vessel Name"))
        detail = find_input_vessel_detail(vessel_name, input_details)
        current_value = get_metadata_value(vessel.get("mmsi_or_imo"))
        if not current_value:
            continue

        current_key = normalize_compare_text(current_value)
        official_key = normalize_compare_text(detail.get("official_number_us"))
        imo_value = normalize_table_text(detail.get("imo_number"))
        imo_is_empty = not imo_value or imo_value.upper() in {"NA", "N/A"}

        if official_key and current_key == official_key and imo_is_empty:
            vessel["mmsi_or_imo"] = set_metadata_value(vessel.get("mmsi_or_imo"), "Not mentioned")
            continue

        normalized = normalize_mmsi_or_imo_value(current_value)
        if normalized:
            vessel["mmsi_or_imo"] = set_metadata_value(vessel.get("mmsi_or_imo"), normalized)
        else:
            vessel["mmsi_or_imo"] = set_metadata_value(vessel.get("mmsi_or_imo"), "Not mentioned")


def fill_vessel_owner_operator_from_input(result: Dict):
    """Fill missing core vessel owner/operator fields from complete Input tables."""
    identifier = extract_identifier_from_sources(result)
    input_details = extract_vessel_detail_table_from_input(identifier)
    for name, detail in extract_vessel_no_details_from_input(identifier).items():
        if normalize_simple_name(name) not in {normalize_simple_name(k) for k in input_details}:
            input_details[name] = detail
    if not input_details:
        return

    for i in range(1, 4):
        vessel = result.get(f"vessel_{i}_info")
        if not isinstance(vessel, dict):
            continue
        detail = find_input_vessel_detail(get_metadata_value(vessel.get("Vessel Name")), input_details)
        if not detail:
            continue
        if any(is_suspicious_owner_operator(detail.get(field)) for field in ["owner", "operator"]):
            continue
        for field in ["owner", "operator"]:
            current = get_metadata_value(vessel.get(field))
            candidate = detail.get(field)
            if field == "operator" and not candidate and detail.get("owner"):
                candidate = detail.get("owner")
            if not current and candidate and not is_suspicious_owner_operator(candidate):
                vessel[field] = set_metadata_value(vessel.get(field), candidate)


def cleanup_vessel_property_damage(result: Dict):
    """Remove accident totals duplicated across vessels while preserving explicit vessel losses."""
    economic_loss = get_metadata_value(result.get("economic_loss"))
    economic_key = normalize_compare_text(clean_money_text(economic_loss))
    identifier = extract_identifier_from_sources(result)
    raw_text = input_plain_text(identifier)
    summary_damage = extract_summary_property_damage_from_input(identifier)
    summary_key = normalize_compare_text(clean_money_text(summary_damage))
    single_core_vessel = nonempty_core_vessel_count(result) == 1
    economic_is_vessel_damage = (
        has_vessel_damage_context(economic_loss)
        or (has_vessel_damage_context(raw_text) and not has_shore_damage_context(raw_text[:3000]))
    )

    values_by_key: Dict[str, List[Tuple[Dict, str]]] = {}
    for i in range(1, 4):
        vessel = result.get(f"vessel_{i}_info")
        if not isinstance(vessel, dict):
            continue
        value = clean_money_text(get_metadata_value(vessel.get("property_damage")))
        if is_pollution_like_damage_value(value):
            vessel["property_damage"] = set_metadata_value(vessel.get("property_damage"), None)
            continue
        if value:
            vessel["property_damage"] = set_metadata_value(vessel.get("property_damage"), value)
        key = normalize_compare_text(value)
        if key:
            values_by_key.setdefault(key, []).append((vessel, "property_damage"))

    for key, fields in values_by_key.items():
        # Retain single-vessel property damage when the source explicitly describes vessel damage.
        if economic_key and key == economic_key and single_core_vessel and economic_is_vessel_damage:
            continue
        if any(
            vessel_mentioned_near_damage(raw_text, get_metadata_value(vessel.get("Vessel Name")))
            and not damage_sentence_targets_other_object(raw_text, get_metadata_value(vessel.get("Vessel Name")), get_metadata_value(vessel.get("vessel_type")))
            for vessel, _ in fields
        ):
            continue
        if any(
            re.search(r"\bbarge\b", f"{get_metadata_value(vessel.get('Vessel Name')) or ''} {get_metadata_value(vessel.get('vessel_type')) or ''}", flags=re.I)
            and re.search(r"\bbarges?\b[^.。;；]{0,160}\b(?:damage|damaged|loss|repair|estimated)\b", raw_text, flags=re.I)
            for vessel, _ in fields
        ):
            continue
        if summary_key and key == summary_key and single_core_vessel and should_summary_damage_apply_to_vessel(summary_damage, raw_text, fields[0][0], single_core_vessel):
            continue

        # In multi-vessel cases, repeated amounts often represent a shared accident total.
        should_clear = False
        if economic_key and key == economic_key and not (single_core_vessel and economic_is_vessel_damage):
            should_clear = True
        if summary_key and key == summary_key:
            for vessel, _ in fields:
                if should_summary_damage_apply_to_vessel(summary_damage, raw_text, vessel, single_core_vessel):
                    continue
                vessel["property_damage"] = set_metadata_value(vessel.get("property_damage"), None)
            continue
        if len(fields) > 1:
            should_clear = True

        if should_clear:
            for vessel, field in fields:
                vessel[field] = set_metadata_value(vessel.get(field), None)


def fill_vessel_property_damage_from_input(result: Dict):
    """Recover explicit vessel property damage from Input when extraction missed it."""
    identifier = extract_identifier_from_sources(result)
    if not identifier:
        return

    single_core_vessel = nonempty_core_vessel_count(result) == 1
    summary_damage = extract_summary_property_damage_from_input(identifier)
    damage_pairs = extract_property_damage_pairs_from_input(identifier)
    raw_text = input_plain_text(identifier)
    input_details = extract_vessel_detail_table_from_input(identifier)
    for name, detail in extract_vessel_no_details_from_input(identifier).items():
        if normalize_simple_name(name) not in {normalize_simple_name(k) for k in input_details}:
            input_details[name] = detail
    for i in range(1, 4):
        vessel = result.get(f"vessel_{i}_info")
        if not isinstance(vessel, dict):
            continue
        vessel_name = get_metadata_value(vessel.get("Vessel Name"))
        if not vessel_name or is_missing_extracted_value(vessel_name):
            continue

        current = get_metadata_value(vessel.get("property_damage"))

        vessel_key = normalize_simple_name(vessel_name)
        vessel_type_key = normalize_simple_name(get_metadata_value(vessel.get("vessel_type")))
        detail = find_input_vessel_detail(vessel_name, input_details)
        candidate = damage_pairs.get(vessel_key)
        if not candidate:
            for pair_key, pair_value in damage_pairs.items():
                if pair_key and (pair_key in vessel_key or vessel_key in pair_key):
                    candidate = pair_value
                    break
        if not candidate:
            for pair_key, pair_value in damage_pairs.items():
                if pair_key and vessel_type_key and (pair_key in vessel_type_key or vessel_type_key in pair_key):
                    candidate = pair_value
                    break
        if not candidate:
            recreational_current = "recreational" in f"{vessel_name or ''} {get_metadata_value(vessel.get('vessel_type')) or ''}".lower()
            for pair_key, pair_value in damage_pairs.items():
                if recreational_current and pair_key in {"recreationalboat", "recreationalvessel"}:
                    candidate = pair_value
                    break

        if not is_missing_extracted_value(current):
            if candidate and normalize_compare_text(candidate) != normalize_compare_text(current):
                vessel["property_damage"] = set_metadata_value(vessel.get("property_damage"), candidate)
            continue

        if not candidate:
            candidate = clean_money_text(detail.get("property_damage")) if detail else None
        if candidate and is_pollution_like_damage_value(candidate):
            candidate = None

        if not candidate and re.search(r"\bbarge\b", f"{vessel_name or ''} {get_metadata_value(vessel.get('vessel_type')) or ''}", flags=re.I):
            candidate = extract_barge_property_damage_from_input(identifier)

        if not candidate:
            candidate = extract_vessel_property_damage_from_input(identifier, vessel_name)

        if (
            not candidate
            and summary_damage
            and should_summary_damage_apply_to_vessel(summary_damage, raw_text, vessel, single_core_vessel)
        ):
            candidate = summary_damage

        if not candidate and single_core_vessel:
            economic_loss = clean_money_text(get_metadata_value(result.get("economic_loss")))
            if economic_loss and (
                has_vessel_damage_context(economic_loss)
                or (has_vessel_damage_context(raw_text) and not has_shore_damage_context(raw_text[:3000]))
            ):
                candidate = economic_loss
            elif (
                summary_damage
                and not is_pollution_like_damage_value(summary_damage)
                and should_summary_damage_apply_to_vessel(summary_damage, raw_text, vessel, single_core_vessel)
            ):
                candidate = summary_damage

        if candidate:
            vessel["property_damage"] = set_metadata_value(vessel.get("property_damage"), candidate)


def split_accident_type_and_no(accident_type: Any, accident_no: Any) -> Tuple[Any, Any]:
    type_value = accident_type
    no_value = accident_no
    for value in [accident_no, accident_type]:
        if not isinstance(value, str):
            continue
        match = re.search(r"\bNo\.?\s*([A-Z]{2,}\d{2}[A-Z]{0,3}\d{3,})\b", value, flags=re.I)
        if match:
            no_value = match.group(1).strip()
            prefix = value[:match.start()].strip(" ,;-")
            if prefix:
                type_value = prefix
            break
    return type_value, no_value


def dedupe_repeated_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return text
    half = len(text) // 2
    left = text[:half].strip()
    right = text[half:].strip()
    if left and left == right:
        return left
    words = text.split()
    if len(words) % 2 == 0:
        mid = len(words) // 2
        if words[:mid] == words[mid:]:
            return " ".join(words[:mid])
    return text


def set_metadata_value(field_data: Any, value: Any) -> Any:
    if isinstance(field_data, dict):
        updated = copy.deepcopy(field_data)
        updated["value"] = value
        return updated
    return create_field_metadata(value=value)


def postprocess_merged_fields(result: Dict):
    """Validate merged fields to remove displaced table text and duplicated values."""
    accident_type, accident_no = split_accident_type_and_no(
        get_metadata_value(result.get("accident_type")),
        get_metadata_value(result.get("accident_no"))
    )
    if accident_type:
        result["accident_type"] = set_metadata_value(result.get("accident_type"), accident_type)
    if accident_no:
        result["accident_no"] = set_metadata_value(result.get("accident_no"), accident_no)

    for field in ["economic_loss", "Ship_loss"]:
        value = clean_money_text(get_metadata_value(result.get(field)))
        if is_pollution_like_damage_value(value):
            result[field] = set_metadata_value(result.get(field), None)
        elif not is_missing_extracted_value(value):
            result[field] = set_metadata_value(result.get(field), value)
    fill_economic_loss_from_input(result)
    fill_pollution_from_input(result)

    waterway = dedupe_repeated_text(get_metadata_value(result.get("waterway_information")))
    if waterway:
        result["waterway_information"] = set_metadata_value(result.get("waterway_information"), waterway)

    accident_time = clean_accident_time(get_metadata_value(result.get("accident_time")))
    if accident_time:
        result["accident_time"] = set_metadata_value(result.get("accident_time"), accident_time)

    cleanup_vessel_owner_operator(result)
    fill_vessel_details_from_input(result)
    fill_vessel_owner_operator_from_input(result)
    cleanup_vessel_mmsi_or_imo(result)
    cleanup_and_fill_passenger_count(result)
    fill_vessel_property_damage_from_input(result)
    cleanup_vessel_property_damage(result)
    fill_missing_causes_from_input(result)


def merge_causes(causes1: Dict, causes2: Dict) -> Dict:
    """Merge and deduplicate accident causes."""
    merged = {
        "causes": [],
        "total_causes": 0
    }
    
    # Collect cause entries from both records.
    all_causes = []
    
    if causes1 and causes1.get("causes"):
        all_causes.extend(causes1.get("causes", []))
    
    if causes2 and causes2.get("causes"):
        all_causes.extend(causes2.get("causes", []))
    
    # Deduplicate by content.
    seen_contents = set()
    unique_causes = []
    for cause in all_causes:
        content = cause.get("content", "")
        if content and content not in seen_contents:
            seen_contents.add(content)
            unique_causes.append(cause)
    
    merged["causes"] = unique_causes
    merged["total_causes"] = len(unique_causes)
    
    return merged


# Vessel name matching

def normalize_vessel_name(name: str) -> str:
    """Normalize vessel names for matching by removing prefixes and descriptions."""
    if not name:
        return ""
    
    # Convert to uppercase and trim whitespace.
    name_upper = name.upper().strip()
    
    # Remove common vessel prefixes with or without slashes.
    prefixes = [
        "M/V", "MV", "M.V.", "M.V",
        "S/S", "SS", "S.S.", "S.S",
        "UTV", "MT", "M/T", "M.T.", "M.T",
        "F/V", "FV", "F.V.", "F.V",
        "R/V", "RV", "R.V.", "R.V",
        "USCGC", "USS", "HMS", "HMCS",
        "THE"
    ]
    
    for prefix in prefixes:
        # Check for prefixes followed by a space.
        if name_upper.startswith(prefix + " "):
            name_upper = name_upper[len(prefix):].strip()
            break
        elif name_upper.startswith(prefix) and len(name_upper) > len(prefix):
            # Also accept prefixes directly attached to the name.
            rest = name_upper[len(prefix):]
            if rest[0].isalpha():
                name_upper = rest.strip()
                break
    
    # Remove nationality and type descriptions, such as "Malaysian-registered bulk carrier".
    # Vessel type terms
    vessel_type_keywords = [
        "BULK CARRIER", "CONTAINER SHIP", "TANKER", "CARGO SHIP", "CARGO VESSEL",
        "FERRY", "TUG", "TUGBOAT", "BARGE", "FISHING VESSEL", "FISHING BOAT",
        "PASSENGER SHIP", "CRUISE SHIP", "FREIGHTER", "VESSEL", "SHIP", "BOAT"
    ]
    
    for keyword in vessel_type_keywords:
        # Locate a type term.
        idx = name_upper.find(keyword)
        if idx != -1:
            # Inspect the text after the type term.
            after_keyword = name_upper[idx + len(keyword):].strip()
            # Use the remaining text as a candidate name.
            if after_keyword:
                name_upper = after_keyword
                break
    
    # Remove parentheses and their contents.
    name_upper = re.sub(r'\([^)]*\)', '', name_upper).strip()
    
    # Remove quotation marks.
    name_upper = name_upper.replace('"', '').replace("'", '').strip()
    
    return name_upper


def extract_vessel_name_words(name: str) -> set:
    """Return the set of words in a vessel name for similarity comparison."""
    if not name:
        return set()
    
    normalized = normalize_vessel_name(name)
    # Split into words and discard empty tokens.
    words = set(w for w in re.split(r'[\s\-_]+', normalized) if w)
    return words


def calculate_vessel_name_similarity(name1: str, name2: str) -> float:
    """Return vessel name similarity on a scale from 0.0 to 1.0.
    
    Combine normalized exact matching, substring coverage, word-level
    Jaccard similarity, and shared-word character coverage.
    """
    if not name1 or not name2:
        return 0.0
    
    # Normalize both names.
    norm1 = normalize_vessel_name(name1)
    norm2 = normalize_vessel_name(name2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Exact match
    if norm1 == norm2:
        return 1.0
    
    # Substring match
    if norm1 in norm2 or norm2 in norm1:
        # Measure substring coverage.
        longer = max(len(norm1), len(norm2))
        shorter = min(len(norm1), len(norm2))
        return shorter / longer
    
    # Word-level Jaccard similarity
    words1 = extract_vessel_name_words(name1)
    words2 = extract_vessel_name_words(name2)
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    if not union:
        return 0.0
    
    jaccard = len(intersection) / len(union)
    
    # Use character coverage when the names share words.
    if intersection:
        # Measure the character count of shared words.
        common_chars = sum(len(w) for w in intersection)
        total_chars = max(len(norm1), len(norm2))
        char_ratio = common_chars / total_chars if total_chars > 0 else 0
        
        # Combine Jaccard similarity and character coverage.
        return max(jaccard, char_ratio)
    
    return jaccard


def is_same_vessel(name1: str, name2: str, threshold: float = 0.6) -> bool:
    """Return whether two names match at the given similarity threshold.
    
    The default threshold is 0.6.
    """
    similarity = calculate_vessel_name_similarity(name1, name2)
    return similarity >= threshold


def find_matching_vessel_improved(vessel_name: str, vessels_dict: Dict, threshold: float = 0.6) -> Optional[Tuple[str, float]]:
    """Find the best matching vessel in vessels_dict at the given threshold.
    
    Return a (vessel record key, similarity) pair, or None if no match exists.
    """
    if not vessel_name:
        return None
    
    best_match = None
    best_similarity = 0.0
    
    for key, vessel_info in vessels_dict.items():
        if not key.startswith("vessel_") or not key.endswith("_info"):
            continue
        
        v_name = vessel_info.get("Vessel Name", {}).get("value", "")
        if not v_name:
            continue
        
        similarity = calculate_vessel_name_similarity(vessel_name, v_name)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = key
    
    if best_match and best_similarity >= threshold:
        return (best_match, best_similarity)
    
    return None


def collect_vessels_from_json(json_data: Dict) -> Dict[str, Dict]:
    """Collect vessel records from an accident JSON object."""
    vessels = {}
    for key, value in json_data.items():
        if key.startswith("vessel_") and key.endswith("_info"):
            vessels[key] = value
    return vessels


def collect_causes_from_json(json_data: Dict) -> Dict[str, Dict]:
    """Collect cause records from an accident JSON object."""
    causes = {}
    for key, value in json_data.items():
        if key.startswith("causes_for_vessel_"):
            causes[key] = value
    return causes


def build_vessel_name_to_causes_map(json_data: Dict) -> Dict[str, Dict]:
    """Map vessel names to cause records using their shared vessel number.
    
    Associate vessel_X_info with causes_for_vessel_X and index the causes
    under both original and normalized vessel names.
    """
    name_to_causes = {}
    
    # Visit each vessel record.
    for key, vessel_info in json_data.items():
        if not key.startswith("vessel_") or not key.endswith("_info"):
            continue
        
        # Read the vessel name.
        vessel_name = vessel_info.get("Vessel Name", {}).get("value", "")
        if not vessel_name:
            continue
        
        # Normalize the name for matching.
        normalized_name = normalize_vessel_name(vessel_name)
        
        # Extract X from vessel_X_info.
        idx = get_vessel_key_index(key)
        if not idx:
            continue
        
        # Find the corresponding cause record.
        causes_key = f"causes_for_vessel_{idx}"
        causes = json_data.get(causes_key)
        
        if causes:
            # Index by both the original and normalized names.
            name_to_causes[vessel_name] = causes
            if normalized_name and normalized_name != vessel_name:
                name_to_causes[normalized_name] = causes
    
    return name_to_causes


def find_causes_by_vessel_name(vessel_name: str, name_to_causes: Dict[str, Dict]) -> Optional[Dict]:
    """Find a cause record by exact, normalized, substring, or fuzzy vessel-name matching.
    
    Return None when no matching record exists.
    """
    if not vessel_name or not name_to_causes:
        return None
    
    # Try the original name.
    if vessel_name in name_to_causes:
        return name_to_causes[vessel_name]
    
    # Try the normalized name.
    normalized_name = normalize_vessel_name(vessel_name)
    if normalized_name in name_to_causes:
        return name_to_causes[normalized_name]
    
    # Allow descriptive names such as "UTV Alliance,United States" to match "Alliance".
    vessel_name_upper = vessel_name.upper()
    normalized_upper = normalized_name.upper() if normalized_name else ""
    
    for name_key, causes in name_to_causes.items():
        name_key_upper = name_key.upper()
        name_key_normalized = normalize_vessel_name(name_key).upper()
        
        # Check substring containment.
        if name_key_upper and (
            name_key_upper in vessel_name_upper or 
            vessel_name_upper in name_key_upper or
            (name_key_normalized and name_key_normalized in normalized_upper) or
            (name_key_normalized and normalized_upper in name_key_normalized)
        ):
            return causes
    
    # Use a lower similarity threshold for the final fallback.
    best_match = None
    best_similarity = 0.0
    
    for name_key, causes in name_to_causes.items():
        similarity = calculate_vessel_name_similarity(vessel_name, name_key)
        if similarity > best_similarity and similarity >= 0.4:  # Fallback similarity threshold
            best_similarity = similarity
            best_match = causes
    
    return best_match


def get_vessel_key_index(key: str) -> Optional[str]:
    """Extract X from vessel_X_info or causes_for_vessel_X."""
    if key.startswith("vessel_") and key.endswith("_info"):
        return key.split("_")[1]
    if key.startswith("causes_for_vessel_"):
        return key.split("_")[-1]
    return None


def merge_json_files(json1: Dict, json2: Dict) -> OrderedDict:
    """Merge two accident JSON objects using the shared schema and vessel-name matching."""
    # Create an empty result record.
    result = create_empty_accident_structure()
    
    # Merge extraction metadata.
    result["extraction_metadata"] = {
        "extraction_time": datetime.now().isoformat(),
        "extractor_version": "v16.0-validation-json-backfill",
        "source_files": [],
        "total_vessels": 0,
        "total_causes": 0
    }
    
    # Collect source filenames.
    src1 = json1.get("extraction_metadata", {}).get("source_files", [])
    src2 = json2.get("extraction_metadata", {}).get("source_files", [])
    result["extraction_metadata"]["source_files"] = list(set(src1 + src2))
    
    # Merge accident details.
    basic_fields = ["accident_no", "accident_time", "accident_type"]
    for field in basic_fields:
        result[field] = merge_field(json1.get(field), json2.get(field))

    fullest_involved = choose_fullest_involved_vessels(json1, json2)
    result["involved_vessels"] = create_field_metadata(
        value=fullest_involved if fullest_involved else None,
        confidence=0.99 if fullest_involved else None,
        source="table"
    )
    
    # Merge location and its nested coordinates.
    result["accident_location"] = merge_location_field(
        json1.get("accident_location", {}),
        json2.get("accident_location", {})
    )
    
    # Merge environmental conditions.
    env_fields = ["weather_conditions", "waterway_information", "visibility"]
    for field in env_fields:
        result[field] = merge_field(json1.get(field), json2.get(field))
    
    # Merge accident-level losses.
    result["pollution"] = merge_field(json1.get("pollution"), json2.get("pollution"))
    result["economic_loss"] = merge_field(json1.get("economic_loss"), json2.get("economic_loss"))
    # Accept both ship_loss and Ship_loss keys.
    ship_loss1 = json1.get("Ship_loss") or json1.get("ship_loss")
    ship_loss2 = json2.get("Ship_loss") or json2.get("ship_loss")
    result["Ship_loss"] = merge_field(ship_loss1, ship_loss2)
    
    # Match and merge vessel records.
    vessels1 = collect_vessels_from_json(json1)
    vessels2 = collect_vessels_from_json(json2)
    
    # Map causes by vessel name so differing vessel numbers do not affect attribution.
    name_to_causes1 = build_vessel_name_to_causes_map(json1)
    name_to_causes2 = build_vessel_name_to_causes_map(json2)
    
    # Group matching vessels from both inputs.
    vessel_groups = []
    processed_vessels2 = set()  # Track matched vessels in json2.
    
    for key1, info1 in vessels1.items():
        name1 = info1.get("Vessel Name", {}).get("value", "")
        if not name1:
            continue
        
        # Find the corresponding vessel in json2.
        matched_key2 = None
        matched_name2 = None
        best_similarity = 0.0
        
        for key2, info2 in vessels2.items():
            if key2 in processed_vessels2:
                continue
            
            name2 = info2.get("Vessel Name", {}).get("value", "")
            if not name2:
                continue
            
            similarity = calculate_vessel_name_similarity(name1, name2)
            if similarity > best_similarity and similarity >= 0.6:
                best_similarity = similarity
                matched_key2 = key2
                matched_name2 = name2
        
        if matched_key2:
            processed_vessels2.add(matched_key2)
            vessel_groups.append({
                "name": name1,
                "name2": matched_name2,  # Keep the json2 name for cause lookup.
                "info1": info1,
                "info2": vessels2[matched_key2]
            })
        else:
            vessel_groups.append({
                "name": name1,
                "name2": None,
                "info1": info1,
                "info2": None
            })
    
    # Prefer table vessel details; use paragraph records only when tables contain no vessels.
    if not vessel_groups:
        for key2, info2 in vessels2.items():
            if key2 in processed_vessels2:
                continue
            
            name2 = info2.get("Vessel Name", {}).get("value", "")
            if not name2:
                continue
            
            vessel_groups.append({
                "name": name2,
                "name2": name2,
                "info1": None,
                "info2": info2
            })
    
    # Add vessel records and their causes in order.
    merged_vessels = {}
    vessel_causes = {}
    
    # Output details for at most three core vessels; keep generic barges only in involved_vessels.
    core_vessel_groups = [
        group for group in vessel_groups
        if not is_barge_or_auxiliary_vessel(group.get("name"), group.get("info1") or group.get("info2"))
    ]
    if core_vessel_groups:
        vessel_groups = core_vessel_groups
    vessel_groups = vessel_groups[:3]

    for i, group in enumerate(vessel_groups, 1):
        # Merge vessel particulars.
        merged_vessel = merge_vessel_info(group["info1"], group["info2"])
        merged_vessels[f"vessel_{i}_info"] = merged_vessel
        
        # Look up causes by vessel name.
        vessel_name = group["name"]
        vessel_name2 = group.get("name2")
        
        # Find causes in json1.
        cause_info1 = find_causes_by_vessel_name(vessel_name, name_to_causes1)
        
        # Use the json2 name for its causes when available, otherwise use the json1 name.
        cause_info2 = None
        if vessel_name2:
            cause_info2 = find_causes_by_vessel_name(vessel_name2, name_to_causes2)
        if not cause_info2:
            cause_info2 = find_causes_by_vessel_name(vessel_name, name_to_causes2)
        
        merged_cause = merge_causes(cause_info1, cause_info2)
        merged_cause = postprocess_causes_for_vessel(merged_cause, get_metadata_value(merged_vessel.get("Vessel Name")) or vessel_name)
        vessel_causes[f"causes_for_vessel_{i}"] = merged_cause

    vessel_causes = propagate_shared_causes_to_all_vessels(vessel_causes)
    
    # Update extraction metadata.
    result["extraction_metadata"]["total_vessels"] = len(fullest_involved) if fullest_involved else len(merged_vessels)
    total_causes = sum(c.get("total_causes", 0) for c in vessel_causes.values())
    result["extraction_metadata"]["total_causes"] = total_causes
    
    # Restore the schema's output field order.
    final_result = OrderedDict()
    
    # Extraction metadata comes first.
    final_result["extraction_metadata"] = result["extraction_metadata"]
    
    # Accident details
    final_result["accident_no"] = result["accident_no"]
    final_result["accident_time"] = result["accident_time"]
    final_result["accident_location"] = result["accident_location"]
    final_result["accident_type"] = result["accident_type"]
    final_result["involved_vessels"] = result["involved_vessels"]
    
    # Vessel records
    for i in range(1, len(merged_vessels) + 1):
        key = f"vessel_{i}_info"
        if key in merged_vessels:
            final_result[key] = merged_vessels[key]
    
    # Cause records in corresponding vessel order
    for i in range(1, len(merged_vessels) + 1):
        key = f"causes_for_vessel_{i}"
        if key in vessel_causes:
            final_result[key] = vessel_causes[key]
    
    # Environmental conditions
    final_result["weather_conditions"] = result["weather_conditions"]
    final_result["waterway_information"] = result["waterway_information"]
    final_result["visibility"] = result["visibility"]
    
    # Accident-level losses
    final_result["pollution"] = result["pollution"]
    final_result["economic_loss"] = result["economic_loss"]
    final_result["Ship_loss"] = result["Ship_loss"]

    postprocess_merged_fields(final_result)
    
    return final_result


def load_json_file(filepath: str) -> Dict:
    """Load JSON from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Dict, filepath: str):
    """Save JSON to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Excel output

def get_field_value(field_data: Any) -> Any:
    """Extract the value from a field record."""
    if field_data is None:
        return None
    if isinstance(field_data, dict):
        return field_data.get("value")
    return field_data


def convert_value_to_string(value: Any, convert_latex: bool = False) -> str:
    """Convert a value to text for Excel output.
    
    When convert_latex is true, also convert embedded LaTeX coordinates.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        result = ", ".join(str(v) for v in value)
    else:
        result = str(value)
    
    # Convert LaTeX coordinates when requested.
    if convert_latex and result:
        result = convert_coordinates_in_text(result)
    
    return result


def get_causes_string(causes_data: Dict, vessel_num: int) -> Tuple[str, int]:
    """Return the combined cause text and cause count for a vessel."""
    if not causes_data:
        return "", 0
    
    causes_list = causes_data.get("causes", [])
    total = causes_data.get("total_causes", len(causes_list))
    
    # Join cause entries into a single cell value.
    cause_contents = []
    for i, cause in enumerate(causes_list, 1):
        content = cause.get("content", "")
        if content:
            cause_contents.append(f"{i}. {content}")
    
    return "\n".join(cause_contents), total


def json_to_excel_row(merged_data: Dict) -> Dict[str, Any]:
    """Convert a merged accident record to a dictionary keyed by Excel column names."""
    row = {}
    
    # Source files
    source_files = merged_data.get("extraction_metadata", {}).get("source_files", [])
    row["source_files"] = extract_source_file_code(source_files)
    
    # Accident details
    row["accident_no"] = convert_value_to_string(get_field_value(merged_data.get("accident_no")))
    row["accident_time"] = convert_value_to_string(get_field_value(merged_data.get("accident_time")))
    # Convert LaTeX coordinates embedded in accident_location.
    row["accident_location"] = convert_value_to_string(get_field_value(merged_data.get("accident_location")), convert_latex=True)
    
    # Latitude and longitude
    location_data = merged_data.get("accident_location", {})
    lat_lon = location_data.get("Latitude and Longitude", {})
    row["accident_latitude"] = convert_value_to_string(lat_lon.get("latitude"), convert_latex=True)
    row["accident_longitude"] = convert_value_to_string(lat_lon.get("longitude"), convert_latex=True)
    
    # Accident type
    row["accident_type"] = convert_value_to_string(get_field_value(merged_data.get("accident_type")))
    
    # Involved-vessel summary
    involved = get_field_value(merged_data.get("involved_vessels"))
    if isinstance(involved, list):
        row["involved_vessels_count"] = len(involved)
        row["involved_vessels_list"] = ", ".join(str(v) for v in involved)
    elif isinstance(involved, (int, float)):
        row["involved_vessels_count"] = int(involved)
        row["involved_vessels_list"] = ""
    else:
        row["involved_vessels_count"] = merged_data.get("extraction_metadata", {}).get("total_vessels", 0)
        row["involved_vessels_list"] = convert_value_to_string(involved)
    
    # Vessel 1 particulars
    vessel_1 = merged_data.get("vessel_1_info", {})
    row["vessel_1_name"] = convert_value_to_string(get_field_value(vessel_1.get("Vessel Name")))
    row["vessel_1_ship_length"] = convert_value_to_string(get_field_value(vessel_1.get("ship_length")))
    row["vessel_1_ship_tonnage"] = convert_value_to_string(get_field_value(vessel_1.get("ship_tonnage")))
    row["vessel_1_vessel_built_year"] = convert_value_to_string(get_field_value(vessel_1.get("vessel_built_year")))
    row["vessel_1_flag_state"] = convert_value_to_string(get_field_value(vessel_1.get("flag_state")))
    row["vessel_1_mmsi_or_imo"] = convert_value_to_string(get_field_value(vessel_1.get("mmsi_or_imo")))
    row["vessel_1_vessel_type"] = convert_value_to_string(get_field_value(vessel_1.get("vessel_type")))
    row["vessel_1_owner"] = convert_value_to_string(get_field_value(vessel_1.get("owner")))
    row["vessel_1_operator"] = convert_value_to_string(get_field_value(vessel_1.get("operator")))
    row["vessel_1_crew_complement"] = normalize_count_output(convert_value_to_string(get_field_value(vessel_1.get("crew_complement"))), "vessel_1_crew_complement")
    row["vessel_1_passenger_count"] = normalize_count_output(convert_value_to_string(get_field_value(vessel_1.get("passenger_count"))), "vessel_1_passenger_count")
    row["vessel_1_casualties"] = convert_value_to_string(get_field_value(vessel_1.get("casualties")))
    row["vessel_1_property_damage"] = normalize_property_damage_output(convert_value_to_string(get_field_value(vessel_1.get("property_damage"))))
    
    # Vessel 2 particulars
    vessel_2 = merged_data.get("vessel_2_info", {})
    row["vessel_2_name"] = convert_value_to_string(get_field_value(vessel_2.get("Vessel Name")))
    row["vessel_2_ship_length"] = convert_value_to_string(get_field_value(vessel_2.get("ship_length")))
    row["vessel_2_ship_tonnage"] = convert_value_to_string(get_field_value(vessel_2.get("ship_tonnage")))
    row["vessel_2_vessel_built_year"] = convert_value_to_string(get_field_value(vessel_2.get("vessel_built_year")))
    row["vessel_2_flag_state"] = convert_value_to_string(get_field_value(vessel_2.get("flag_state")))
    row["vessel_2_mmsi_or_imo"] = convert_value_to_string(get_field_value(vessel_2.get("mmsi_or_imo")))
    row["vessel_2_vessel_type"] = convert_value_to_string(get_field_value(vessel_2.get("vessel_type")))
    row["vessel_2_owner"] = convert_value_to_string(get_field_value(vessel_2.get("owner")))
    row["vessel_2_operator"] = convert_value_to_string(get_field_value(vessel_2.get("operator")))
    row["vessel_2_crew_complement"] = normalize_count_output(convert_value_to_string(get_field_value(vessel_2.get("crew_complement"))), "vessel_2_crew_complement")
    row["vessel_2_passenger_count"] = normalize_count_output(convert_value_to_string(get_field_value(vessel_2.get("passenger_count"))), "vessel_2_passenger_count")
    row["vessel_2_casualties"] = convert_value_to_string(get_field_value(vessel_2.get("casualties")))
    row["vessel_2_property_damage"] = normalize_property_damage_output(convert_value_to_string(get_field_value(vessel_2.get("property_damage"))))
    
    # Vessel 3 particulars
    vessel_3 = merged_data.get("vessel_3_info", {})
    row["vessel_3_name"] = convert_value_to_string(get_field_value(vessel_3.get("Vessel Name")))
    row["vessel_3_ship_length"] = convert_value_to_string(get_field_value(vessel_3.get("ship_length")))
    row["vessel_3_ship_tonnage"] = convert_value_to_string(get_field_value(vessel_3.get("ship_tonnage")))
    row["vessel_3_vessel_built_year"] = convert_value_to_string(get_field_value(vessel_3.get("vessel_built_year")))
    row["vessel_3_flag_state"] = convert_value_to_string(get_field_value(vessel_3.get("flag_state")))
    row["vessel_3_mmsi_or_imo"] = convert_value_to_string(get_field_value(vessel_3.get("mmsi_or_imo")))
    row["vessel_3_vessel_type"] = convert_value_to_string(get_field_value(vessel_3.get("vessel_type")))
    row["vessel_3_owner"] = convert_value_to_string(get_field_value(vessel_3.get("owner")))
    row["vessel_3_operator"] = convert_value_to_string(get_field_value(vessel_3.get("operator")))
    row["vessel_3_crew_complement"] = normalize_count_output(convert_value_to_string(get_field_value(vessel_3.get("crew_complement"))), "vessel_3_crew_complement")
    row["vessel_3_passenger_count"] = normalize_count_output(convert_value_to_string(get_field_value(vessel_3.get("passenger_count"))), "vessel_3_passenger_count")
    row["vessel_3_casualties"] = convert_value_to_string(get_field_value(vessel_3.get("casualties")))
    row["vessel_3_property_damage"] = normalize_property_damage_output(convert_value_to_string(get_field_value(vessel_3.get("property_damage"))))
    
    # Vessel 1 causes
    causes_1 = merged_data.get("causes_for_vessel_1", {})
    causes_str_1, causes_count_1 = get_causes_string(causes_1, 1)
    row["vessel_1_causes"] = causes_str_1
    row["vessel_1_causes_count"] = causes_count_1
    
    # Vessel 2 causes
    causes_2 = merged_data.get("causes_for_vessel_2", {})
    causes_str_2, causes_count_2 = get_causes_string(causes_2, 2)
    row["vessel_2_causes"] = causes_str_2
    row["vessel_2_causes_count"] = causes_count_2
    
    # Vessel 3 causes
    causes_3 = merged_data.get("causes_for_vessel_3", {})
    causes_str_3, causes_count_3 = get_causes_string(causes_3, 3)
    row["vessel_3_causes"] = causes_str_3
    row["vessel_3_causes_count"] = causes_count_3
    
    # Environmental conditions
    row["weather_conditions"] = convert_value_to_string(get_field_value(merged_data.get("weather_conditions")))
    row["waterway_information"] = convert_value_to_string(get_field_value(merged_data.get("waterway_information")))
    
    # Accident-level losses
    row["pollution"] = normalize_pollution_output(convert_value_to_string(get_field_value(merged_data.get("pollution"))))
    row["economic_loss"] = convert_value_to_string(get_field_value(merged_data.get("economic_loss")))
    row["ship_loss"] = convert_value_to_string(get_field_value(merged_data.get("Ship_loss")))
    
    # Notes and data quality
    row["notes"] = ""
    row["data_quality"] = ""
    
    return {key: normalize_excel_cell_value(value) for key, value in row.items()}


# Column order follows the output template.
EXCEL_COLUMNS = [
    "source_files",
    "accident_no",
    "accident_time",
    "accident_location",
    "accident_latitude",
    "accident_longitude",
    "accident_type",
    "involved_vessels_count",
    "involved_vessels_list",
    "vessel_1_name",
    "vessel_1_ship_length",
    "vessel_1_ship_tonnage",
    "vessel_1_vessel_built_year",
    "vessel_1_flag_state",
    "vessel_1_mmsi_or_imo",
    "vessel_1_vessel_type",
    "vessel_1_owner",
    "vessel_1_operator",
    "vessel_1_crew_complement",
    "vessel_1_passenger_count",
    "vessel_1_casualties",
    "vessel_1_property_damage",
    "vessel_2_name",
    "vessel_2_ship_length",
    "vessel_2_ship_tonnage",
    "vessel_2_vessel_built_year",
    "vessel_2_flag_state",
    "vessel_2_mmsi_or_imo",
    "vessel_2_vessel_type",
    "vessel_2_owner",
    "vessel_2_operator",
    "vessel_2_crew_complement",
    "vessel_2_passenger_count",
    "vessel_2_casualties",
    "vessel_2_property_damage",
    "vessel_3_name",
    "vessel_3_ship_length",
    "vessel_3_ship_tonnage",
    "vessel_3_vessel_built_year",
    "vessel_3_flag_state",
    "vessel_3_mmsi_or_imo",
    "vessel_3_vessel_type",
    "vessel_3_owner",
    "vessel_3_operator",
    "vessel_3_crew_complement",
    "vessel_3_passenger_count",
    "vessel_3_casualties",
    "vessel_3_property_damage",
    "vessel_1_causes",
    "vessel_1_causes_count",
    "vessel_2_causes",
    "vessel_2_causes_count",
    "vessel_3_causes",
    "vessel_3_causes_count",
    "weather_conditions",
    "waterway_information",
    "pollution",
    "economic_loss",
    "ship_loss",
    "notes",
    "data_quality"
]


def save_batch_excel(all_merged_data: List[Dict], filepath: str):
    """Write all merged accident records to one Excel workbook at filepath."""
    if not OPENPYXL_AVAILABLE:
        print("警告: openpyxl未安装，无法输出Excel文件")
        return
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Maritime Accident Data"
    
    # Style the header row.
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="DAEEF3", end_color="DAEEF3", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    # Write column headings.
    for col_idx, col_name in enumerate(EXCEL_COLUMNS, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
    
    # Write accident rows.
    for row_idx, merged_data in enumerate(all_merged_data, 2):
        row_data = json_to_excel_row(merged_data)
        for col_idx, col_name in enumerate(EXCEL_COLUMNS, 1):
            value = row_data.get(col_name, "")
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    
    # Set column widths.
    for col_idx, col_name in enumerate(EXCEL_COLUMNS, 1):
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = 15
    
    # Freeze the header row.
    ws.freeze_panes = "A2"
    
    wb.save(filepath)


# Batch processing

def extract_identifier(filename: str, pattern: str) -> Optional[str]:
    """Extract a filename identifier with pattern, returning None if unmatched.
    
    filename is the basename without its directory. For example, a pattern
    of three uppercase letters followed by four digits matches MAB1201.
    """
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


def find_matching_files(
    input_dir1: str,
    input_dir2: str,
    pattern: str
) -> List[Tuple[str, str, str]]:
    """Find JSON file pairs with matching identifiers in two directories.
    
    Use pattern to extract identifiers. Return a list of
    (identifier, file1_path, file2_path) tuples.
    """
    # Find JSON files in both input directories.
    dir1_files = {f: os.path.join(input_dir1, f) 
                  for f in os.listdir(input_dir1) 
                  if f.endswith('.json')}
    dir2_files = {f: os.path.join(input_dir2, f) 
                  for f in os.listdir(input_dir2) 
                  if f.endswith('.json')}
    
    # Map identifiers to file paths.
    dir1_by_id = {}
    for filename, filepath in dir1_files.items():
        identifier = extract_identifier(filename, pattern)
        if identifier:
            dir1_by_id[identifier] = filepath
    
    dir2_by_id = {}
    for filename, filepath in dir2_files.items():
        identifier = extract_identifier(filename, pattern)
        if identifier:
            dir2_by_id[identifier] = filepath
    
    # Pair files with matching identifiers.
    matched_pairs = []
    common_ids = set(dir1_by_id.keys()) & set(dir2_by_id.keys())
    
    for identifier in sorted(common_ids):
        matched_pairs.append((
            identifier,
            dir1_by_id[identifier],
            dir2_by_id[identifier]
        ))
    
    return matched_pairs


def batch_merge(
    input_dir1: str,
    input_dir2: str,
    output_dir: str,
    pattern: str,
    output_prefix: str = "merged_",
    output_suffix: str = "",
    output_excel: bool = True
) -> Dict[str, Any]:
    """Merge matching JSON files and optionally write one combined Excel workbook.
    
    Args:
        input_dir1: First input directory.
        input_dir2: Second input directory.
        output_dir: Destination for JSON records and the workbook.
        pattern: Regular expression used to extract filename identifiers.
        output_prefix: Prefix for output filenames.
        output_suffix: Suffix for output filenames.
        output_excel: Whether to write the combined workbook.
    
    Returns:
        Batch processing statistics.
    """
    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)
    
    # Find matching file pairs.
    matched_pairs = find_matching_files(input_dir1, input_dir2, pattern)
    
    results = {
        "total_pairs": len(matched_pairs),
        "successful": 0,
        "failed": 0,
        "details": []
    }
    
    # Collect successful merge results for the combined workbook.
    all_merged_data = []
    
    print("=" * 80)
    print("海事事故信息JSON批量融合程序")
    print("=" * 80)
    print(f"\n输入文件夹1: {input_dir1}")
    print(f"输入文件夹2: {input_dir2}")
    print(f"输出文件夹: {output_dir}")
    print(f"匹配模式: {pattern}")
    print(f"\n找到 {len(matched_pairs)} 对匹配的文件")
    print("-" * 80)
    
    for identifier, file1, file2 in matched_pairs:
        output_filename = f"{output_prefix}{identifier}{output_suffix}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            print(f"\n处理: {identifier}")
            print(f"  文件1: {os.path.basename(file1)}")
            print(f"  文件2: {os.path.basename(file2)}")
            
            # Load and merge the file pair.
            json1 = load_json_file(file1)
            json2 = load_json_file(file2)
            merged = merge_json_files(json1, json2)
            
            # Save the merged JSON record.
            save_json_file(merged, output_path)
            
            
            # Add the record to the workbook data.
            all_merged_data.append(merged)
            
            results["successful"] += 1
            results["details"].append({
                "identifier": identifier,
                "status": "success",
                "input_files": [file1, file2],
                "output_file": output_path,
                "total_vessels": merged["extraction_metadata"]["total_vessels"],
                "total_causes": merged["extraction_metadata"]["total_causes"]
            })
            
            print(f"  [OK] 成功 -> {output_filename}")
            print(f"    船舶数量: {merged['extraction_metadata']['total_vessels']}")
            print(f"    原因数量: {merged['extraction_metadata']['total_causes']}")
            
        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "identifier": identifier,
                "status": "failed",
                "input_files": [file1, file2],
                "error": str(e)
            })
            print(f"  [ERROR] 失败: {str(e)}")
    
    # Write all successful merges to one workbook in the output directory.
    if output_excel and OPENPYXL_AVAILABLE and all_merged_data:
        summary_excel_path = os.path.join(output_dir, "all_merged_accidents.xlsx")
        save_batch_excel(all_merged_data, summary_excel_path)
        print(f"\n汇总Excel文件已保存: {summary_excel_path}")
    
    # Print batch totals.
    print("\n" + "=" * 80)
    print("处理完成!")
    print(f"  成功: {results['successful']}")
    print(f"  失败: {results['failed']}")
    print(f"  总计: {results['total_pairs']}")
    if output_excel and OPENPYXL_AVAILABLE:
        print(f"  汇总Excel: all_merged_accidents.xlsx")
    print("=" * 80)
    
    return results


def main():
    """Parse command-line options and run batch merging."""
    
    # Default paths and parameters
    
    # Resolve paths relative to this script.
    import pathlib
    PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
    
    # Default directories for batch processing
    DEFAULT_INPUT_DIR1 = str(PROJECT_ROOT / "Table_Information_Extraction_Output")      # First input directory
    DEFAULT_INPUT_DIR2 = str(PROJECT_ROOT / "Paragraph_Information_Extraction_Output")      # Second input directory
    DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "Merge_Information_output")      # Destination for JSON records and the combined workbook
    
    # Choose a pattern that extracts the same identifier from both input filename formats.
    # For example, MAB1201_content_list_extracted.json and extracted_MAB1201_content_list_annotated.json
    # share the identifier MAB1201.
    DEFAULT_PATTERN = r"(?:extracted_)?(\d+)(?=_content_list_annotated)"
    
    # Output filename configuration
    DEFAULT_OUTPUT_PREFIX = "merged_"                  # Output filename prefix
    DEFAULT_OUTPUT_SUFFIX = ""                         # Output filename suffix
    
    
    parser = argparse.ArgumentParser(
        description="海事事故信息JSON融合程序（批量处理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  使用默认配置直接运行:
    python merge_v4.py

  指定输入输出路径:
    python merge_v4.py --dir1 /path/to/folder1 --dir2 /path/to/folder2 --output-dir /path/to/output

  自定义文件匹配模式:
    python merge_v4.py --pattern "([A-Z]{3}\\d{4})"

  文件匹配模式说明:
    --pattern 参数用于从文件名中提取标识符以匹配两个文件夹中的对应文件
    例如: MAB1201_content_list_extracted.json 和 extracted_MAB1201_content_list_annotated.json
    都包含 "MAB1201" 作为标识符，可以使用模式 "([A-Z]{3}\\d{4})" 来提取

  Excel输出:
    默认输出汇总Excel文件到output文件夹，使用 --no-excel 禁用Excel输出
        """
    )
    
    # Batch processing arguments
    parser.add_argument("--dir1", default=DEFAULT_INPUT_DIR1, help="第一个输入文件夹路径")
    parser.add_argument("--dir2", default=DEFAULT_INPUT_DIR2, help="第二个输入文件夹路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出文件夹路径（JSON和Excel都输出到这里）")
    parser.add_argument(
        "--pattern", 
        default=DEFAULT_PATTERN,
        help="文件匹配模式（正则表达式），用于从文件名中提取标识符"
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="输出文件名前缀"
    )
    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help="输出文件名后缀"
    )
    
    # Excel output option
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="禁用Excel输出，仅输出JSON"
    )
    
    args = parser.parse_args()
    
    # Check Excel support.
    output_excel = not args.no_excel
    if output_excel and not OPENPYXL_AVAILABLE:
        print("警告: openpyxl未安装，将仅输出JSON格式")
        print("安装命令: pip install openpyxl")
        output_excel = False
    
    # Run batch merging.
    results = batch_merge(
        input_dir1=args.dir1,
        input_dir2=args.dir2,
        output_dir=args.output_dir,
        pattern=args.pattern,
        output_prefix=args.output_prefix,
        output_suffix=args.output_suffix,
        output_excel=output_excel
    )
    
    return results


if __name__ == "__main__":
    main()
