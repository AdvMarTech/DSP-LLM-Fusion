"""Extract maritime accident information from document paragraphs with Qwen.

Group text by section, identify vessels, and extract vessel fields, causes,
weather, sea state, and pollution. Vessel extraction visits sections in
priority order until the requested fields are complete. Cause entries
retain page, confidence, classification, and source-section metadata.
"""

import json
import re
import os
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Runtime configuration
# Resolve paths relative to this script.
import pathlib
_PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()

# Directory containing input JSON documents
INPUT_FOLDER = str(_PROJECT_ROOT / "Document_Structural_Parsing_Output")

# Directory for extracted JSON records
OUTPUT_FOLDER = str(_PROJECT_ROOT / "Paragraph_Information_Extraction_Output")

# Local model path
MODEL_PATH = "/mnt/data/LLM/models/Qwen/Qwen2.5-7B-Instruct"

# Extraction field configuration
# Fields extracted from paragraph text
# Vessel fields
VESSEL_EXTRACT_FIELDS = [
    "ship_length",
    "ship_tonnage", 
    "vessel_built_year",
    "flag_state",
    "mmsi_or_imo"
]

# Environmental and loss fields
GENERAL_EXTRACT_FIELDS = [
    "weather_conditions",
    "sea_state",
    "pollution"
]

# Causes are handled separately to retain detailed provenance.

# Priority sections
KEY_SECTIONS = [
    "Summary", "Vessel Information", "Weather, Tides and Currents",
    "Waterway Information", "Accident Narrative", "Introduction",
    "Accident Description", "Synopsis", "Background", "Accident Events",
    "Probable Cause", "Contributing Factors", "Analysis", "Conclusions",
    "Findings", "Safety Issues"
]

# Priority sections for vessel particulars
VESSEL_INFO_SECTIONS = [
    "Vessel Information", "vessel information",
    "Summary", "summary", 
    "Introduction", "introduction",
    "Accident Description", "Accident Description and Timeline of Events",
    "Background", "Synopsis"
]

# Generation parameters
GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "do_sample": False,
    "repetition_penalty": 1.1
}


class QwenExtractor:
    """Generate extraction responses with a local Qwen model."""
    
    def __init__(self, model_path: str = MODEL_PATH):
        """Load the Qwen tokenizer and model from model_path."""
        print(f"正在加载Qwen模型: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        print("模型加载完成")
    
    def generate(self, prompt: str) -> str:
        """Generate a text response for the extraction prompt."""
        messages = [
            {"role": "system", "content": "You are a professional maritime accident information extraction assistant. Extract information accurately from the given text and output in JSON format. Be very careful to distinguish between different vessels and do not confuse barge information with main vessel information."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **GENERATION_CONFIG
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


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
    return {
        "value": value,
        "confidence": confidence,
        "page_idx": page_idx if page_idx is not None else [],
        "classification": classification,
        "section_type": section_type,
        "source": source,
        "source_chapter": source_chapter
    }


def create_location_with_coordinates(
    location: str = None,
    latitude: str = None,
    longitude: str = None,
    confidence: float = None,
    page_idx: List[int] = None,
    classification: str = None,
    section_type: str = None,
    source: str = "text",
    source_chapter: str = None
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
        "source": source,
        "source_chapter": source_chapter
    }


def create_vessel_info(vessel_name: str = None) -> Dict:
    """Create a vessel record for paragraph extraction.
    
    Extract ship_length, ship_tonnage, vessel_built_year, flag_state, and
    mmsi_or_imo; retain the remaining schema fields without extracting them.
    """
    return {
        "Vessel Name": create_field_metadata(value=vessel_name),
        "ship_length": create_field_metadata(),
        "ship_tonnage": create_field_metadata(),
        "vessel_built_year": create_field_metadata(),
        "flag_state": create_field_metadata(),
        "mmsi_or_imo": create_field_metadata(),
        # Retain these schema fields without extracting their values.
        "vessel_type": create_field_metadata(),
        "owner": create_field_metadata(),
        "operator": create_field_metadata(),
        "crew_complement": create_field_metadata(),
        "passenger_count": create_field_metadata(),
        "casualties": create_field_metadata(),
        "property_damage": create_field_metadata(),
    }


def create_cause_item(
    content: str,
    page_idx: List[int] = None,
    confidence: float = None,
    level: int = None,
    classification: str = None,
    section_type: str = None,
    source_chapter: str = None
) -> Dict:
    """Create a cause entry with source metadata."""
    return {
        "content": content,
        "page_idx": page_idx if page_idx is not None else [],
        "confidence": confidence,
        "level": level,
        "classification": classification,
        "section_type": section_type,
        "source_chapter": source_chapter
    }


def create_empty_accident_structure(vessel_count: int = 1) -> Dict:
    """Create an empty accident record."""
    result = OrderedDict()
    
    result["extraction_metadata"] = {
        "extraction_time": None,
        "extractor_version": "v3.0-qwen7b-limited-fields-fixed-v13-pollution-normalized",
        "source_files": [],
        "total_vessels": vessel_count,
        "total_causes": 0,
        "extracted_fields": VESSEL_EXTRACT_FIELDS + GENERAL_EXTRACT_FIELDS + ["causes"]
    }
    
    # Retain accident fields without extracting their values here.
    result["accident_no"] = create_field_metadata()
    result["accident_time"] = create_field_metadata()
    result["accident_location"] = create_location_with_coordinates()
    result["accident_type"] = create_field_metadata()
    result["involved_vessels"] = create_field_metadata()
    
    # Create one record for each detected vessel.
    for i in range(1, vessel_count + 1):
        result[f"vessel_{i}_info"] = create_vessel_info()
    
    # Create one cause collection for each vessel.
    for i in range(1, vessel_count + 1):
        result[f"causes_for_vessel_{i}"] = {
            "causes": [],
            "total_causes": 0
        }
    
    # Environmental fields to extract
    result["weather_conditions"] = create_field_metadata()
    result["sea_state"] = create_field_metadata()
    
    # Extract pollution; retain the other loss fields for schema compatibility.
    result["pollution"] = create_field_metadata()
    result["economic_loss"] = create_field_metadata()
    result["ship_loss"] = create_field_metadata()
    
    return result


def filter_and_group_content(content_list: List[Dict]) -> Dict[str, List[Dict]]:
    """Filter document nodes and group paragraph text by section.
    
    Skip page 0, tables, and nontext nodes. Nodes with text_level=1
    start a new section.
    """
    sections = {}
    current_section = "Untitled"
    current_section_content = []
    
    for item in content_list:
        # Skip page 0; paragraph extraction uses later pages.
        page_idx = item.get("page_idx", 0)
        if page_idx == 0:
            continue
        
        # Skip table nodes.
        if item.get("type") == "table":
            continue
        
        # Only process text nodes.
        if item.get("type") != "text":
            continue
        
        text = item.get("text", "").strip()
        if not text:
            continue
        
        # A node with text_level=1 starts a section.
        if item.get("text_level") == 1:
            # Save the preceding section.
            if current_section_content:
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].extend(current_section_content)
            
            # Start a new section.
            current_section = text.strip()
            current_section_content = []
        else:
            # Record paragraph text and its source section.
            item_with_chapter = item.copy()
            item_with_chapter["_source_chapter"] = current_section
            current_section_content.append(item_with_chapter)
    
    # Save the final section.
    if current_section_content:
        if current_section not in sections:
            sections[current_section] = []
        sections[current_section].extend(current_section_content)
    
    return sections


def detect_vessel_count(extractor: QwenExtractor, sections: Dict[str, List[Dict]], content_list: List[Dict]) -> tuple:
    """Detect vessel names and count, preferring the official first-page summary."""
    # Prefer vessel information from page 0.
    first_page_text = ""
    for item in content_list:
        if item.get("page_idx") == 0 and item.get("type") == "text":
            text = item.get("text", "").strip()
            if text:
                first_page_text += f"\n{text}"
    
    # Use first-page text when available.
    if first_page_text.strip():
        relevant_text = first_page_text
    else:
        # Fall back to relevant sections when page 0 has no text.
        relevant_text = ""
        priority_sections = ["Summary", "Synopsis", "Introduction", "Vessel Information", 
                            "Background", "Accident Description", "Accident Narrative"]
        
        for sec_name in priority_sections:
            for key, content in sections.items():
                if sec_name.lower() in key.lower():
                    text = "\n".join([item.get("text", "") for item in content[:10]])
                    relevant_text += f"\n{text}"
                    break
        
        # If no priority section is found, use the first few paragraphs of all sections.
        if not relevant_text:
            for sec_name, content in list(sections.items())[:3]:
                text = "\n".join([item.get("text", "") for item in content[:5]])
                relevant_text += f"\n{text}"
    
    if not relevant_text.strip():
        return 1, [{"name": None}]
    
    # Build the vessel detection prompt.
    prompt = f"""Please analyze the following maritime accident text and determine how many MAIN vessels are involved in this accident.

## IMPORTANT: This text is from the FIRST PAGE (page_idx=0) which contains the OFFICIAL vessel names and count.
The first page typically contains the cover page or summary with the official list of vessels involved.

## Text Content:
{relevant_text[:3000]}

## Task:
Count the number of distinct MAIN vessels mentioned as being directly involved in this accident.

## CRITICAL RULES:
1. Only count vessels that are PRIMARY participants in the accident (e.g., vessels that collided, vessel that grounded, etc.)
2. Do NOT count rescue vessels, coast guard vessels, or vessels that arrived after the accident
3. A towing vessel (tug) and its barges should be counted as ONE unit if they are acting together
   - Example: "Alliance pushing barges MMI 3024 and MMI 3025" = count as 1 vessel (the tug Alliance)
   - The barges are not separate vessels for accident attribution purposes
4. For collision accidents, there are usually 2 main vessels involved (e.g., a tug with barges vs. a tankship = 2 vessels)

## IMPORTANT - Vessel Naming Convention:
- Use the MAIN vessel name, not the barge names
- For a towing vessel with barges, use the TUG name (e.g., "Alliance"), not the barge names
- For tankships/cargo ships, use the ship name (e.g., "Naticina")

## IMPORTANT - Vessel Order:
- List vessels in the order they appear in the official document/header if available
- If the document lists "Vessel, Flag" column with multiple entries, use that order
- Otherwise, list alphabetically

## Output Format:
Return a JSON object with the following format:
{{
  "vessel_count": <number>,
  "vessels": [
    {{"name": "<vessel 1 name>"}},
    {{"name": "<vessel 2 name>"}}
  ],
  "accident_type": "<collision/grounding/fire/capsizing/other>",
  "reasoning": "<brief explanation of how you identified the vessels>"
}}

Only output the JSON object, no other text."""
    
    response = extractor.generate(prompt)
    
    # Parse the response.
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
            vessel_count = result.get("vessel_count", 1)
            vessels = result.get("vessels", [{"name": None}])
            
            # Keep vessel_count consistent with the vessel list.
            if vessel_count != len(vessels):
                vessel_count = len(vessels)
            
            return vessel_count, vessels
    except:
        pass
    
    return 1, [{"name": None}]


# Extraction prompts

def build_vessel_specific_extraction_prompt(
    vessel_name: str, 
    vessel_idx: int,
    section_name: str, 
    section_content: List[Dict],
    all_vessel_info: List[Dict] = None
) -> str:
    """Build an extraction prompt for one target vessel.
    
    Args:
        vessel_name: Target vessel name.
        vessel_idx: Vessel number, starting at 1.
        section_name: Source section heading.
        section_content: Paragraphs in the section.
        all_vessel_info: All vessel records, used to distinguish the target.
    
    Returns:
        The extraction prompt.
    """
    # Combine section paragraphs with their page indices.
    text_parts = []
    for item in section_content:
        text = item.get("text", "")
        page_idx = item.get("page_idx", "?")
        text_parts.append(f"[Page {page_idx}] {text}")
    
    text_content = "\n".join(text_parts)
    
    # Describe other vessels so the model can distinguish the target.
    other_vessels_info = ""
    if all_vessel_info:
        other_vessels = [v.get("name", "") for v in all_vessel_info if v and v.get("name") and v.get("name") != vessel_name]
        if other_vessels:
            other_vessels_info = f"""
## OTHER VESSELS IN THIS ACCIDENT (DO NOT extract their information):
{', '.join(other_vessels)}
- Information about these other vessels should be IGNORED
- Only extract information for "{vessel_name}"
"""

    prompt = f"""## TASK: Extract vessel specifications for ONE SPECIFIC VESSEL ONLY

## TARGET VESSEL: "{vessel_name}"
You must ONLY extract information that is EXPLICITLY about the vessel named "{vessel_name}".

## Section: {section_name}

## Text Content:
{text_content}
{other_vessels_info}
## CRITICAL RULES - READ VERY CAREFULLY:

### Rule 1 - ONLY EXTRACT FOR "{vessel_name}":
- You are extracting information for "{vessel_name}" ONLY
- Every piece of information you extract MUST be explicitly stated as belonging to "{vessel_name}" in the text
- If the text says "The {vessel_name} is 900 feet long", then extract "900 feet"
- If the text says something about another vessel, IGNORE IT completely

### Rule 2 - DISTINGUISH BARGES FROM MAIN VESSELS (VERY IMPORTANT):
- A "barge" (like MMI 3024, MMI 3025) is NOT the same as the "towing vessel" or "tug"
- If "{vessel_name}" is a towing vessel/tug, DO NOT extract barge specifications for it
- Barge information includes: barge length, barge tonnage, barge draft, barge build year
- Example: "The MMI 3024... is 297 feet in length" - this is BARGE info, NOT the tug's length
- Example: "1,619-gross-registered-ton (GRT) tank barge" - this is BARGE tonnage, NOT the tug's tonnage
- The tug/towing vessel specifications are usually stated separately (e.g., "The vessel was built in 2008 and is 72 feet long")

### Rule 3 - BEWARE OF CONTEXT CONFUSION:
- "26 feet from the bow" or "approximately 26 feet from the tow's bow" refers to ANTENNA POSITION, not vessel length
- "AIS antenna was located approximately X feet from..." is NOT the vessel's length
- "X feet aft of the bow" refers to equipment position, not vessel length
- Read the FULL sentence to understand the context before extracting

### Rule 4 - NO FABRICATION (STRICT):
- ONLY extract information that is EXPLICITLY stated in the text
- If ship length is not mentioned for "{vessel_name}", return null
- Do NOT guess, infer, or use typical values
- Do NOT invent IMO numbers or MMSI numbers - only extract if explicitly stated
- If you see "IMO 9123456" as an example in your training, do NOT use it - only extract real numbers from the text

### Rule 5 - VERIFY VESSEL NAME BEFORE EXTRACTING:
- Before extracting any value, check: "Does the text explicitly say this is about {vessel_name}?"
- Look for phrases like: "The {vessel_name} is...", "{vessel_name}'s length is...", "...of the {vessel_name}"
- Also look for context like: heading "{vessel_name}" followed by specifications
- If the text doesn't explicitly connect a specification to "{vessel_name}", DO NOT extract it

## Fields to Extract for "{vessel_name}":
1. ship_length: The overall length of "{vessel_name}" ONLY (e.g., "900 feet", "72 feet", "185.9 meters")
2. ship_tonnage: The tonnage of "{vessel_name}" ONLY (e.g., "32,474 GT", "1,619 GRT", "10,000 DWT")
3. vessel_built_year: The year "{vessel_name}" was built/delivered ONLY (e.g., "2008", "2010", "March 2010")
4. flag_state: The flag state/registry of "{vessel_name}" ONLY (e.g., "Marshall Islands", "United States", "Panama")
5. mmsi_or_imo: The MMSI or IMO number of "{vessel_name}" ONLY - ONLY if explicitly stated in the text

## Output Format:
Return a JSON object. For each field:
- "value": the extracted value (or null if not found for "{vessel_name}")
- "found": true ONLY if explicitly found for "{vessel_name}", false otherwise
- "page_idx": the page number(s) where found (empty list if not found)
- "context": brief quote showing the value is about "{vessel_name}" (helps verify correctness)

{{
  "target_vessel": "{vessel_name}",
  "ship_length": {{"value": null, "found": false, "page_idx": [], "context": ""}},
  "ship_tonnage": {{"value": null, "found": false, "page_idx": [], "context": ""}},
  "vessel_built_year": {{"value": null, "found": false, "page_idx": [], "context": ""}},
  "flag_state": {{"value": null, "found": false, "page_idx": [], "context": ""}},
  "mmsi_or_imo": {{"value": null, "found": false, "page_idx": [], "context": ""}}
}}

## EXAMPLES:

### Example 1 - Correct extraction for a tankship:
Text: "The Naticina is a Marshall Islands–flagged... tankship... The vessel is 900 feet in length and 150 feet in beam"
Target: "Naticina"
Output:
{{
  "target_vessel": "Naticina",
  "ship_length": {{"value": "900 feet", "found": true, "page_idx": [3], "context": "The vessel [Naticina] is 900 feet in length"}},
  "flag_state": {{"value": "Marshall Islands", "found": true, "page_idx": [3], "context": "Naticina is a Marshall Islands–flagged"}}
}}

### Example 2 - Correct extraction for a towing vessel (avoiding barge confusion):
Text: "The Alliance is a U.S.-flagged... towing vessel... The vessel was built in 2008 and is 72 feet long... The MMI 3024... is 297 feet in length and 54 feet in beam"
Target: "Alliance"
Output:
{{
  "target_vessel": "Alliance",
  "ship_length": {{"value": "72 feet", "found": true, "page_idx": [3], "context": "The vessel was built in 2008 and is 72 feet long"}},
  "vessel_built_year": {{"value": "2008", "found": true, "page_idx": [3], "context": "The vessel was built in 2008"}},
  "flag_state": {{"value": "United States", "found": true, "page_idx": [3], "context": "The Alliance is a U.S.-flagged"}}
}}
Note: 297 feet is NOT extracted because that's the barge MMI 3024's length, NOT the Alliance's length.

### Example 3 - Avoiding antenna position confusion:
Text: "The AIS antenna on the Alliance was located approximately 26 feet from the tow's bow"
Target: "Alliance"
ship_length should be null - "26 feet" refers to antenna position, not vessel length!

Only output the JSON object, no other text."""

    return prompt


def build_general_fields_extraction_prompt(section_name: str, section_content: List[Dict]) -> str:
    """Build a prompt for weather, sea state, and pollution extraction."""
    # Combine section paragraphs.
    text_parts = []
    for item in section_content:
        text = item.get("text", "")
        page_idx = item.get("page_idx", "?")
        text_parts.append(f"[Page {page_idx}] {text}")
    
    text_content = "\n".join(text_parts)
    
    prompt = f"""## TASK: Extract environmental and condition information from maritime accident report

## Section: {section_name}

## Text Content:
{text_content}

## CRITICAL RULES:
1. ONLY extract information that is EXPLICITLY stated in the text
2. If a field is not mentioned, set "found" to false and "value" to null
3. Do NOT invent, guess, or infer any information

## Fields to Extract:
1. weather_conditions: Weather at the time of the accident (visibility, wind, sky conditions, temperature, etc.)
   - Example: "clear, with 7 to 8 miles of visibility", "wind southwesterly at 4 knots"
   
2. sea_state: Sea/wave conditions at the time of the accident
   - Example: "mean wave heights less than 1 foot", "calm seas", "rough seas with 6-foot swells"
   
3. pollution: Any pollution or environmental damage from the accident
   - Example: "No pollution", "oil spill of approximately 100 gallons", "No product was released"
   - If the text explicitly states no pollution, no water pollution, no product released, or no environmental damage, output "None reported"
   - If pollution is not mentioned at all, set "found" to false and "value" to null
   - The "None reported" normalization applies only to pollution, not to other fields

## Output Format:
{{
  "weather_conditions": {{"value": null or "description", "found": true/false, "page_idx": []}},
  "sea_state": {{"value": null or "description", "found": true/false, "page_idx": []}},
  "pollution": {{"value": null or "description", "found": true/false, "page_idx": []}}
}}

Only output the JSON object, no other text."""

    return prompt


def build_causes_extraction_prompt(section_name: str, section_content: List[Dict], vessel_count: int = 1, vessel_info: List[Dict] = None) -> str:
    """Build a cause extraction prompt with vessel and source context."""
    # Combine section text and collect source page indices.
    text_parts = []
    for item in section_content:
        text = item.get("text", "")
        page_idx = item.get("page_idx", "?")
        text_level = item.get("text_level", "?")
        text_parts.append(f"[Page {page_idx}, Level {text_level}] {text}")
    
    text_content = "\n".join(text_parts)
    
    # Describe the mapping between vessel names and records.
    vessel_desc = ""
    if vessel_info and len(vessel_info) > 0:
        vessel_desc = "\n## CRITICAL - Vessel Name to Number Mapping (USE THIS FOR CAUSE ATTRIBUTION):\n"
        for i, v in enumerate(vessel_info, 1):
            vessel_name = v.get("name", "Unknown") if v and v.get("name") else "Unknown"
            vessel_desc += f"- Vessel {i} = \"{vessel_name}\"\n"
        vessel_desc += """
## IMPORTANT - How to attribute causes:
- If the cause mentions a specific vessel by name (e.g., "the master of Alliance"), look up which vessel number it is
- Set vessel_number to match the vessel number from the mapping above
- If the cause is about environmental conditions or a joint action/failure explicitly affecting all core vessels, set vessel_number to 0
- If the cause is a management, regulatory, manufacturer, shipyard, port/terminal/fleeting facility, VTS, Coast Guard, Army Corps, industry-practice, or other shared organizational factor, set vessel_number to 0
- If the cause is specifically about one vessel's crew, pilot, master, watchstander, equipment, structure, maintenance, or operation, set vessel_number to that vessel only
"""
    
    # Identify the official Probable Cause section.
    is_probable_cause_section = any(kw in section_name.lower() for kw in ["probable cause", "probable_cause"])
    
    if is_probable_cause_section:
        section_priority_note = """
## IMPORTANT - THIS IS THE "PROBABLE CAUSE" SECTION:
This section contains the OFFICIAL and DEFINITIVE causes determined by the investigation.
- Extract causes from this section with HIGH PRIORITY
- These are the confirmed, official causes of the accident
- Mark all causes from this section with "is_official_cause": true
- Prefer the concise official probable cause and explicitly stated contributing factors.
- Keep the wording short and cause-centered. Do not append consequences such as fire damage, sinking, flooding results, or destruction unless those words are part of the official cause phrase.
- Do NOT split supporting evidence, symptoms, inspection findings, or consequences into separate causes when they merely support the official cause.
"""
    else:
        section_priority_note = """
## Note on Cause Extraction:
- Only extract statements that are ACTUAL CAUSES of the accident
- Do NOT extract general descriptions, background information, or observations
- A cause must have a direct causal relationship to the accident
- Look for keywords like: "caused by", "due to", "resulted from", "contributing factor", "failure to", "led to"
"""
    
    prompt = f"""Please extract ONLY the ACTUAL CAUSES of this maritime accident from the following text.

## Section: {section_name}
{section_priority_note}
{vessel_desc}
## Text Content:
{text_content}

## CRITICAL INSTRUCTIONS:
1. This accident involves {vessel_count} vessel(s)
2. ONLY extract statements that DIRECTLY CAUSED or CONTRIBUTED TO the accident, preferably from an official "probable cause" or explicitly stated "contributing factor" sentence
3. Do NOT extract:
   - General descriptions or background information
   - Observations that did not cause the accident
   - Consequences or results of the accident
   - Recommendations or lessons learned
   - Supporting evidence or symptoms such as smoke, smell, postaccident inspection findings, or damage descriptions unless the text explicitly says they caused the accident
4. A valid cause must answer: "What action, failure, or condition LED TO the accident?"
5. For multi-vessel accidents (e.g., collision), CAREFULLY attribute each cause to the correct vessel:
   - FIRST: Find the vessel name mentioned in the cause statement
   - SECOND: Look up which vessel number corresponds to that name in the mapping above
   - THIRD: Set vessel_number to that number
   - If the cause is environmental, management/organizational, regulatory, third-party, or affects all main responsible vessels, set vessel_number to 0
   - If the cause explicitly says "both operators", "both vessels", "all involved parties", or a joint decision/failure, set vessel_number to 0
6. Each returned "content" must be a concise cause phrase, not a full narrative sentence. If the text says "X led to Y", return only "X" unless "Y" is necessary to identify the cause.

## STRICT RULE - NO FABRICATION:
- Only extract causes that are EXPLICITLY stated in the text
- If no causes are found, return an empty causes array
- Do NOT invent, infer, or fabricate any causes

## Output Format:
Return a JSON object with a "causes" array. Each cause should include:
- "content": The cause description (exactly as stated or closely paraphrased from the text)
- "page_idx": List of page numbers where this cause is mentioned
- "vessel_number": Which vessel this cause relates to (1, 2, ..., or 0 for shared/environmental/management/third-party causes)
- "cause_type": Type of cause (probable_cause, contributing_factor, root_cause, human_error, technical_failure, environmental, organizational)
- "is_official_cause": true if from Probable Cause section, false otherwise

Example:
{{
  "causes": [
    {{
      "content": "The encroachment by the master of the Alliance into the Texas City Channel",
      "page_idx": [12],
      "vessel_number": 2,
      "cause_type": "human_error",
      "is_official_cause": true
    }}
  ]
}}

If no actual causes are found in this section, return:
{{
  "causes": []
}}

Only output the JSON object, no other text."""

    return prompt


def parse_llm_response(response: str) -> Dict:
    """Parse JSON from the model response."""
    # Locate the JSON payload.
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    return {}


# Field extraction

def extract_vessel_info_separately(
    extractor: QwenExtractor,
    sections: Dict[str, List[Dict]],
    vessel_count: int,
    vessel_info: List[Dict],
    result: Dict
) -> Dict:
    """Extract fields separately for each vessel across all document sections.
    
    Visit vessel-specific and vessel-information sections first, then other
    sections. Stop once all requested fields are populated.
    
    Args:
        extractor: Model extractor instance.
        sections: Section groups with page 0 excluded.
        vessel_count: Number of detected vessels.
        vessel_info: Vessel records containing name fields.
        result: Accumulated accident record.
    
    Returns:
        The updated accident record.
    """
    # Extract each vessel separately.
    for vessel_idx, v_info in enumerate(vessel_info, 1):
        vessel_name = v_info.get("name") if v_info else None
        if not vessel_name:
            print(f"        跳过船舶 {vessel_idx}: 无船舶名称")
            continue
        
        print(f"\n      === 提取船舶 {vessel_idx} 信息: {vessel_name} ===")
        vessel_info_key = f"vessel_{vessel_idx}_info"
        
        # Visit sections most likely to contain this vessel's particulars first.
        def section_priority_for_vessel(item):
            section_name = item[0]
            section_name_normalized = section_name.lower().replace(" ", "")
            
            # First priority: headings containing the target vessel name.
            if vessel_name:
                vessel_name_normalized = vessel_name.lower().replace(" ", "")
                if vessel_name_normalized in section_name_normalized or section_name_normalized in vessel_name_normalized:
                    return (0, section_name)
            
            # Second priority: known vessel-information sections.
            is_vessel_section = any(
                vs.lower().replace(" ", "") in section_name_normalized 
                for vs in VESSEL_INFO_SECTIONS
            )
            if is_vessel_section:
                return (1, section_name)
            
            # Third priority: all other sections.
            return (2, section_name)
        
        sorted_sections = sorted(sections.items(), key=section_priority_for_vessel)
        
        # Search all sections because vessel particulars may be scattered.
        for section_name, section_content in sorted_sections:
            if not section_content:
                continue
            
            # Check whether all requested fields have values.
            all_fields_extracted = True
            for field in VESSEL_EXTRACT_FIELDS:
                current = result[vessel_info_key].get(field, {})
                if current.get("value") is None:
                    all_fields_extracted = False
                    break
            
            # Stop once the vessel record is complete.
            if all_fields_extracted:
                print(f"        船舶 {vessel_idx} ({vessel_name}) 所有字段已提取完成，跳过剩余章节")
                break
            
            # Include sections even when their headings are outside the priority list.
            
            # Determine section priority for logging.
            section_name_normalized = section_name.lower().replace(" ", "")
            is_priority_section = any(
                vs.lower().replace(" ", "") in section_name_normalized 
                for vs in VESSEL_INFO_SECTIONS
            )
            if vessel_name:
                vessel_name_normalized = vessel_name.lower().replace(" ", "")
                if vessel_name_normalized in section_name_normalized or section_name_normalized in vessel_name_normalized:
                    is_priority_section = True
            
            priority_mark = "【优先】" if is_priority_section else ""
            print(f"        {priority_mark}处理章节: {section_name}")
            
            # Read the section's source metadata.
            first_item = section_content[0]
            source_metadata = {
                "confidence": first_item.get("confidence"),
                "page_idx": list(set(item.get("page_idx") for item in section_content if item.get("page_idx"))),
                "classification": first_item.get("classification"),
                "section_type": first_item.get("section_type"),
                "text_level": first_item.get("text_level")
            }
            
            # Build the target vessel prompt.
            prompt = build_vessel_specific_extraction_prompt(
                vessel_name=vessel_name,
                vessel_idx=vessel_idx,
                section_name=section_name,
                section_content=section_content,
                all_vessel_info=vessel_info
            )
            
            try:
                response = extractor.generate(prompt)
                extracted = parse_llm_response(response)
                
                if extracted:
                    # Check that the response refers to the target vessel.
                    target = extracted.get("target_vessel", "")
                    if target and target.lower() != vessel_name.lower():
                        print(f"          [WARNING] 响应目标船舶不匹配: 期望 {vessel_name}, 得到 {target}")
                        continue
                    
                    # Merge the extracted fields.
                    extracted_any = False
                    for field in VESSEL_EXTRACT_FIELDS:
                        if field in extracted:
                            field_data = extracted[field]
                            if isinstance(field_data, dict) and field_data.get("found") and field_data.get("value"):
                                # Only fill empty fields.
                                current = result[vessel_info_key].get(field, {})
                                if current.get("value") is None:
                                    page_idx = field_data.get("page_idx", [])
                                    if not isinstance(page_idx, list):
                                        page_idx = [page_idx] if page_idx else []
                                    
                                    result[vessel_info_key][field] = create_field_metadata(
                                        value=field_data["value"],
                                        confidence=source_metadata.get("confidence"),
                                        page_idx=page_idx,
                                        classification=source_metadata.get("classification"),
                                        section_type=source_metadata.get("section_type"),
                                        source="text",
                                        source_chapter=section_name
                                    )
                                    print(f"          [OK] {field}: {field_data['value']}")
                                    extracted_any = True
                    
                    if not extracted_any:
                        print(f"          - 本章节未发现 {vessel_name} 的新信息")
                else:
                    print(f"          [ERROR] 未能解析响应")
                    
            except Exception as e:
                print(f"          [ERROR] 提取失败: {str(e)}")
    
    return result


def extract_general_fields(
    extractor: QwenExtractor,
    sections: Dict[str, List[Dict]],
    result: Dict
) -> Dict:
    """Extract weather, sea state, and pollution fields."""
    # Sections likely to describe weather and sea state
    weather_sections = ["Weather", "Tides", "Currents", "Environmental", "Conditions",
                        "Summary", "Accident Description", "Background"]
    
    for section_name, section_content in sections.items():
        if not section_content:
            continue
        
        # Select relevant sections.
        # Ignore spaces when matching headings to tolerate OCR word splitting.
        section_name_normalized = section_name.lower().replace(" ", "")
        is_relevant = any(
            ws.lower().replace(" ", "") in section_name_normalized 
            for ws in weather_sections
        )
        if not is_relevant:
            continue
        
        # Read the section's source metadata.
        first_item = section_content[0]
        source_metadata = {
            "confidence": first_item.get("confidence"),
            "page_idx": list(set(item.get("page_idx") for item in section_content if item.get("page_idx"))),
            "classification": first_item.get("classification"),
            "section_type": first_item.get("section_type"),
            "text_level": first_item.get("text_level")
        }
        
        # Build the accident-level extraction prompt.
        prompt = build_general_fields_extraction_prompt(section_name, section_content)
        
        try:
            response = extractor.generate(prompt)
            extracted = parse_llm_response(response)
            
            if extracted:
                for field in GENERAL_EXTRACT_FIELDS:
                    if field in extracted:
                        field_data = extracted[field]
                        if isinstance(field_data, dict) and field_data.get("found") and field_data.get("value"):
                            current = result.get(field, {})
                            if current.get("value") is None:
                                page_idx = field_data.get("page_idx", source_metadata.get("page_idx", []))
                                if not isinstance(page_idx, list):
                                    page_idx = [page_idx] if page_idx else []
                                
                                result[field] = create_field_metadata(
                                    value=field_data["value"],
                                    confidence=source_metadata.get("confidence"),
                                    page_idx=page_idx,
                                    classification=source_metadata.get("classification"),
                                    section_type=source_metadata.get("section_type"),
                                    source="text",
                                    source_chapter=section_name
                                )
                                print(f"        [OK] {field}: {field_data['value']}")
        except Exception as e:
            print(f"        [ERROR] 通用字段提取失败: {str(e)}")
    
    return result


def merge_causes_data(
    accumulated: Dict,
    new_data: Dict,
    source_metadata: Dict,
    section_name: str,
    vessel_count: int = 1
) -> Dict:
    """Merge cause entries into the relevant vessel records."""
    if "causes" not in new_data:
        return accumulated
    
    causes_list = new_data.get("causes", [])
    
    # Check whether the source is the official Probable Cause section.
    is_from_probable_cause = any(kw in section_name.lower() for kw in ["probable cause", "probable_cause"])
    
    for cause_item in causes_list:
        if not cause_item.get("content"):
            continue
        
        # Determine which vessel owns this cause.
        vessel_number = cause_item.get("vessel_number", 1)
        if vessel_number == -1:
            # Treat third-party, shore-based, regulatory, and management factors as shared causes for legacy responses.
            vessel_number = 0
        if vessel_number == 0:
            content_lower = str(cause_item.get("content", "")).lower()
            shared_patterns = [
                "both vessels", "all vessels", "two vessels", "each vessel",
                "both operators", "all involved parties", "joint decision", "joint failure",
                "weather", "current", "visibility", "fog", "river", "waterway",
                "high water", "high-water", "wind", "winds", "severe weather",
                "channel", "traffic", "passing room", "meeting", "risk of collision",
                "company", "manufacturer", "shipyard", "port authority", "terminal",
                "fleeting area", "coast guard", "army corps", "vessel traffic service",
                "vts", "harbor department", "lockmaster", "industry practice",
                "oversight", "procedures", "monitoring", "management"
            ]
            if vessel_count > 1 and not any(pattern in content_lower for pattern in shared_patterns):
                continue
            # Assign explicitly shared, environmental, or management causes to all vessels.
            vessel_numbers = list(range(1, vessel_count + 1))
        else:
            vessel_numbers = [min(vessel_number, vessel_count)]
        
        for v_num in vessel_numbers:
            cause_key = f"causes_for_vessel_{v_num}"
            
            # Create the cause collection if needed.
            if cause_key not in accumulated:
                accumulated[cause_key] = {
                    "causes": [],
                    "total_causes": 0
                }
            
            # Read source page indices.
            page_idx = cause_item.get("page_idx", source_metadata.get("page_idx", []))
            if not isinstance(page_idx, list):
                page_idx = [page_idx] if page_idx else []
            
            # Check for an existing identical or similar cause.
            existing_contents = [c.get("content", "").lower().strip() for c in accumulated[cause_key]["causes"]]
            new_content_lower = cause_item["content"].lower().strip()
            
            # Compare normalized cause text.
            is_duplicate = False
            for existing in existing_contents:
                if new_content_lower == existing:
                    is_duplicate = True
                    break
                if len(new_content_lower) > 50 and len(existing) > 50:
                    if new_content_lower in existing or existing in new_content_lower:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                is_official = is_from_probable_cause or cause_item.get("is_official_cause", False)
                existing_official = any(
                    c.get("is_official_cause") for c in accumulated[cause_key]["causes"]
                )
                if existing_official and not is_official:
                    continue

                # Create the cause entry.
                new_cause = create_cause_item(
                    content=cause_item["content"],
                    page_idx=page_idx,
                    confidence=source_metadata.get("confidence"),
                    level=source_metadata.get("text_level"),
                    classification=source_metadata.get("classification"),
                    section_type=source_metadata.get("section_type"),
                    source_chapter=section_name
                )
                
                # Include the cause type when provided.
                if cause_item.get("cause_type"):
                    new_cause["cause_type"] = cause_item["cause_type"]
                
                # Record whether this is an official cause.
                new_cause["is_official_cause"] = is_official
                
                # Place official causes first.
                if is_official:
                    accumulated[cause_key]["causes"] = [
                        c for c in accumulated[cause_key]["causes"]
                        if c.get("is_official_cause")
                    ]
                    accumulated[cause_key]["causes"].insert(0, new_cause)
                else:
                    accumulated[cause_key]["causes"].append(new_cause)
                    
                accumulated[cause_key]["total_causes"] = len(accumulated[cause_key]["causes"])
    
    # Update the cause count.
    total_causes = 0
    for i in range(1, vessel_count + 1):
        cause_key = f"causes_for_vessel_{i}"
        if cause_key in accumulated:
            total_causes += accumulated[cause_key].get("total_causes", 0)
    accumulated["extraction_metadata"]["total_causes"] = total_causes
    
    return accumulated


def extract_from_document(
    input_file: str,
    output_file: str,
    model_path: str = MODEL_PATH,
    extractor: QwenExtractor = None
) -> Dict:
    """Extract accident information from an annotated document JSON file."""
    print(f"\n{'='*60}")
    print(f"处理文件: {input_file}")
    print(f"{'='*60}")
    
    # Read the input document.
    print(f"\n[1/7] 读取输入文件...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content_list = json.load(f)
    print(f"      共 {len(content_list)} 个内容节点")
    
    # Filter and group paragraph content.
    print(f"\n[2/7] 过滤和分组内容 (跳过page_idx=0和type=table)...")
    sections = filter_and_group_content(content_list)
    print(f"      识别到 {len(sections)} 个章节:")
    for sec_name, sec_content in sections.items():
        is_key = "★" if any(key.lower() in sec_name.lower() for key in KEY_SECTIONS) else " "
        print(f"      {is_key} {sec_name}: {len(sec_content)} 个段落")
    
    # Load the model.
    if extractor is None:
        print(f"\n[3/7] 加载Qwen模型...")
        extractor = QwenExtractor(model_path)
    else:
        print(f"\n[3/7] 使用已加载的Qwen模型...")
    
    # Detect the vessels.
    print(f"\n[4/7] 检测船舶数量 (优先从第一页获取官方信息)...")
    vessel_count, vessel_info = detect_vessel_count(extractor, sections, content_list)
    print(f"      确定船舶数量: {vessel_count}")
    for i, v in enumerate(vessel_info, 1):
        print(f"      船舶 {i}: {v.get('name', 'Unknown')}")
    
    # Create an empty result record.
    print(f"\n[5/7] 创建结果结构...")
    result = create_empty_accident_structure(vessel_count)
    result["extraction_metadata"]["extraction_time"] = datetime.now().isoformat()
    result["extraction_metadata"]["source_files"] = [input_file]
    
    # Store the detected vessels.
    result["extraction_metadata"]["vessel_info"] = vessel_info
    
    # Populate names in the vessel records.
    for i, v_info in enumerate(vessel_info, 1):
        vessel_key = f"vessel_{i}_info"
        if vessel_key in result and v_info.get("name"):
            result[vessel_key]["Vessel Name"]["value"] = v_info["name"]
            result[vessel_key]["Vessel Name"]["source"] = "auto_detection"
    
    # Extract the configured fields.
    print(f"\n[6/7] 抽取指定字段...")
    
    # Extract particulars separately for each vessel.
    print(f"\n      [6.1] 逐个船舶提取信息...")
    result = extract_vessel_info_separately(
        extractor=extractor,
        sections=sections,
        vessel_count=vessel_count,
        vessel_info=vessel_info,
        result=result
    )
    
    # Extract weather, sea state, and pollution.
    print(f"\n      [6.2] 提取通用字段...")
    result = extract_general_fields(
        extractor=extractor,
        sections=sections,
        result=result
    )
    
    # Extract causes.
    print(f"\n      [6.3] 提取事故原因...")
    
    # Visit sections in priority order.
    def section_priority(item):
        section_name = item[0].lower()
        if "probable cause" in section_name or "probable_cause" in section_name:
            return (0, 0, item[0])
        elif any(key.lower() in section_name for key in KEY_SECTIONS):
            return (0, 1, item[0])
        else:
            return (1, 0, item[0])
    
    sorted_sections = sorted(sections.items(), key=section_priority)
    
    for section_name, section_content in sorted_sections:
        if not section_content:
            continue
        
        # Extract causes from relevant sections.
        cause_related_keywords = ["probable cause", "cause", "factor", "finding", "analysis", "conclusion", "safety"]
        if any(kw in section_name.lower() for kw in cause_related_keywords):
            is_probable_cause = "probable cause" in section_name.lower() or "probable_cause" in section_name.lower()
            if is_probable_cause:
                print(f"        -> 【重点】抽取官方Probable Cause...")
            else:
                print(f"        -> 从章节 '{section_name}' 抽取causes...")
            
            # Read the section's source metadata.
            first_item = section_content[0]
            source_metadata = {
                "confidence": first_item.get("confidence"),
                "page_idx": list(set(item.get("page_idx") for item in section_content if item.get("page_idx"))),
                "classification": first_item.get("classification"),
                "section_type": first_item.get("section_type"),
                "text_level": first_item.get("text_level")
            }
            
            causes_prompt = build_causes_extraction_prompt(section_name, section_content, vessel_count, vessel_info)
            
            try:
                causes_response = extractor.generate(causes_prompt)
                causes_extracted = parse_llm_response(causes_response)
                
                if causes_extracted and "causes" in causes_extracted:
                    result = merge_causes_data(result, causes_extracted, source_metadata, section_name, vessel_count)
                    cause_count = len(causes_extracted.get("causes", []))
                    print(f"        [OK] 成功抽取 {cause_count} 个causes")
            except Exception as e:
                print(f"        [ERROR] causes抽取失败: {str(e)}")
    
    # Save the result.
    print(f"\n[7/7] 保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("抽取完成!")
    print(f"{'='*60}")
    
    # Print an extraction summary.
    print("\n抽取结果摘要:")
    print(f"  - 船舶数量: {vessel_count}")
    
    for i in range(1, vessel_count + 1):
        vessel_key = f"vessel_{i}_info"
        cause_key = f"causes_for_vessel_{i}"
        
        if vessel_key in result:
            vessel = result[vessel_key]
            print(f"\n  船舶 {i} ({vessel.get('Vessel Name', {}).get('value', 'Unknown')}) 信息:")
            for field in VESSEL_EXTRACT_FIELDS:
                value = vessel.get(field, {}).get("value", "N/A")
                source_chapter = vessel.get(field, {}).get("source_chapter", "N/A")
                if value and value != "N/A":
                    print(f"    - {field}: {value} (来源: {source_chapter})")
        
        if cause_key in result:
            causes_data = result[cause_key]
            cause_count = causes_data.get("total_causes", 0)
            official_count = sum(1 for c in causes_data.get("causes", []) if c.get("is_official_cause"))
            print(f"\n  船舶 {i} 事故原因 (共 {cause_count} 个, 其中官方确定原因 {official_count} 个):")
            for idx, cause in enumerate(causes_data.get("causes", []), 1):
                official_mark = "【官方】" if cause.get("is_official_cause") else ""
                print(f"    [{idx}] {official_mark}{cause.get('content', 'N/A')}")
                print(f"        page_idx: {cause.get('page_idx', [])}")
                print(f"        来源章节: {cause.get('source_chapter', 'N/A')}")
                if cause.get('cause_type'):
                    print(f"        类型: {cause.get('cause_type')}")
    
    # Print environmental fields.
    print(f"\n  环境和损失信息:")
    for field in GENERAL_EXTRACT_FIELDS:
        value = result.get(field, {}).get("value", "N/A")
        source_chapter = result.get(field, {}).get("source_chapter", "N/A")
        if value and value != "N/A":
            print(f"    - {field}: {value} (来源: {source_chapter})")
    
    return result


def process_folder(
    input_folder: str = INPUT_FOLDER,
    output_folder: str = OUTPUT_FOLDER,
    model_path: str = MODEL_PATH
):
    """Extract information from every JSON file in a directory."""
    # Create the output directory.
    os.makedirs(output_folder, exist_ok=True)
    
    # Find input JSON files.
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    if not json_files:
        print(f"警告: 输入文件夹 {input_folder} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件待处理")
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"模型路径: {model_path}")

    print("\n[初始化] 加载Qwen模型（批处理期间只加载一次）...")
    extractor = QwenExtractor(model_path)
    
    for idx, json_file in enumerate(json_files, 1):
        print(f"\n{'#'*60}")
        print(f"处理文件 [{idx}/{len(json_files)}]: {json_file}")
        print(f"{'#'*60}")
        
        input_path = os.path.join(input_folder, json_file)
        output_path = os.path.join(output_folder, f"extracted_{json_file}")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"输出已存在，跳过: {output_path}")
            continue
        
        try:
            extract_from_document(input_path, output_path, model_path, extractor=extractor)
        except Exception as e:
            print(f"处理文件 {json_file} 失败: {str(e)}")
            continue


def main():
    """Parse command-line options and run paragraph extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="海事事故信息抽取工具 - Qwen7B (修复版)")
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=INPUT_FOLDER,
        help=f"输入文件/文件夹路径 (默认: {INPUT_FOLDER})"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=OUTPUT_FOLDER,
        help=f"输出文件/文件夹路径 (默认: {OUTPUT_FOLDER})"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=MODEL_PATH,
        help=f"Qwen模型路径 (默认: {MODEL_PATH})"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量处理模式（处理输入文件夹中的所有JSON文件）"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("海事事故段落信息抽取工具")
    print("="*60)
    print("抽取规则:")
    print("  1. 按船舶名称分别提取信息")
    print("  2. 区分驳船和主船舶")
    print("  3. 根据上下文区分AIS天线位置和船舶长度等数值")
    print("  4. 仅记录原文中提供的IMO/MMSI号码")
    print("  5. 匹配章节名称时忽略OCR产生的多余空格")
    print("  6. 按优先级遍历所有章节提取船舶信息")
    print("="*60)
    print("配置信息:")
    print(f"  输入路径: {args.input}")
    print(f"  输出路径: {args.output}")
    print(f"  模型路径: {args.model}")
    print(f"  批量模式: {args.batch}")
    print(f"\n只抽取以下字段:")
    print(f"  船舶字段: {VESSEL_EXTRACT_FIELDS}")
    print(f"  通用字段: {GENERAL_EXTRACT_FIELDS}")
    print(f"  事故原因: causes (带详细元数据)")
    print("="*60)
    
    if args.batch or os.path.isdir(args.input):
        # Batch mode
        process_folder(
            input_folder=args.input,
            output_folder=args.output,
            model_path=args.model
        )
    else:
        # Single-file mode
        extract_from_document(
            input_file=args.input,
            output_file=args.output,
            model_path=args.model
        )


if __name__ == "__main__":
    main()
