#!/usr/bin/env python3
"""Annotate document nodes with section confidence and source metadata.

Detect front-page metadata and section headings, classify headings with
Qwen, and attach confidence values to the associated paragraph nodes.
"""

import json
import os
from typing import List, Dict, Tuple
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import re


class SectionCredibilityAnnotator:
    """Assign confidence metadata to document sections and front-page fields."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Load the tokenizer and model."""
        print(f"正在加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).eval()
        self.device = device
        print("[OK] 模型加载完成！\n")
        
        # Fixed confidence for metadata fields.
        self.METADATA_CONFIDENCE = 0.95
    
    def _is_metadata_node(self, node: Dict, index: int, total_nodes: int) -> bool:
        """Return whether a node resembles front-page metadata.
        
        Checks position, page index, field labels, and common metadata value formats.
        Only nodes on page 0 within the first 15% or 30 nodes are considered.
        
        Args:
            node: Document node.
            index: Node index.
            total_nodes: Total number of document nodes.
        """
        text = node.get('text', '').strip()
        
        # Empty text can belong to the initial metadata region.
        if not text:
            return index < 20 and node.get('page_idx', 0) == 0
        
        # Restrict metadata to the first 15% or 30 nodes.
        if index > min(int(total_nodes * 0.15), 30):
            return False
        
        # Metadata must be on the first page.
        if node.get('page_idx', 0) != 0:
            return False
        
        # Field labels typically end with a colon.
        if text.endswith(':'):
            # Common metadata field names
            metadata_fields = [
                'accident no', 'vessel', 'accident type', 'location', 'date', 
                'time', 'owner', 'operator', 'property damage', 'damages',
                'injuries', 'complement', 'fatalities', 'casualties',
                'synopsis', 'incident no', 'event type', 'aircraft', 'vehicle',
                'flight', 'route', 'registration', 'manufacturer', 'model'
            ]
            
            text_lower = text.lower().rstrip(':').strip()
            if any(field in text_lower for field in metadata_fields):
                return True
        
        # Multiple colons can indicate several fields in one node.
        if text.count(':') >= 2:
            return True
        
        # Date formats commonly used in metadata
        date_patterns = [
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # MM/DD/YYYY
            r'^[A-Z][a-z]+ \d{1,2}, \d{4}',  # September 19, 2005
            r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD
        ]
        for pattern in date_patterns:
            if re.match(pattern, text):
                return True
        
        # Time formats
        if re.match(r'^\d{4} ', text):  # For example, "2030 Pacific daylight time".
            return True
        if re.match(r'^\d{1,2}:\d{2}', text):  # For example, "10:30".
            return True
        
        # Identifiers, such as accident numbers
        if re.match(r'^[A-Z]{2,}-\d+-[A-Z]{2,}-\d+', text):
            return True
        
        # Short values near the beginning may belong to metadata fields.
        # Consider short text within the first 15 nodes.
        if index < 15 and len(text) < 50:
            # Check common value forms.
            # These include single words, short phrases, and numbers.
            words = text.split()
            if len(words) <= 5:  # At most five words.
                return True
        
        # Latitude and longitude formats
        if 'latitude' in text.lower() or 'longitude' in text.lower():
            return True
        if re.search(r'\d+[º°]\d+', text):  # For example, "43º40.0".
            return True
        
        # Currency amounts
        if re.search(r'\$[\d,]+', text):  # For example, "$120,000".
            return True
        
        # Personnel counts
        if re.search(r'\d+\s+(crew|passenger|person|people)', text.lower()):
            return True
        
        # Casualty counts
        if re.search(r'(fatalit|injur|death)', text.lower()):
            if len(text) < 100:  # Short casualty descriptions.
                return True
        
        return False
    
    def _identify_metadata_region(self, nodes: List[Dict]) -> Tuple[int, int]:
        """Return the start and end indices of the metadata region.
        
        Scan up to 50 nodes, stopping at the first section heading. A heading
        has text_level=1, a section keyword, and 5-100 characters, and does not
        end with a colon or match a date, time, or identifier.
        """
        metadata_end = 0
        
        # A fixed 50-node limit also covers short documents.
        scan_limit = min(50, len(nodes))
        
        # Section heading keywords
        section_keywords = [
            'description', 'narrative', 'synopsis', 'summary',
            'background', 'introduction', 'overview', 'information',
            'analysis', 'discussion', 'findings', 'conclusion',
            'chronology', 'timeline', 'sequence'
        ]
        
        for i in range(scan_limit):
            if i >= len(nodes):
                break
                
            node = nodes[i]
            text = node.get('text', '').strip()
            
            # Stop the metadata region at a recognized section heading.
            if text and node.get('text_level') == 1:
                # Check heading characteristics.
                text_lower = text.lower()
                
                
                if not text.endswith(':'):  # Exclude field labels.
                    # Require a section keyword.
                    has_keyword = any(kw in text_lower for kw in section_keywords)
                    
                    # Check heading length.
                    is_reasonable_length = 5 <= len(text) <= 100
                    
                    # Exclude dates, times, and identifiers.
                    is_date = re.match(r'^[A-Z][a-z]+ \d{1,2}, \d{4}', text)
                    is_time = re.match(r'^\d{4} ', text)
                    is_code = re.match(r'^[A-Z]{2,}-\d+-[A-Z]{2,}-\d+', text)
                    
                    if has_keyword and is_reasonable_length and not is_date and not is_time and not is_code:
                        # The section heading marks the end of metadata.
                        metadata_end = i
                        break
            
            # Continue through nodes that resemble metadata.
            if self._is_metadata_node(node, i, len(nodes)):
                metadata_end = i + 1
            elif metadata_end > 0:
                # After metadata begins, allow intervening first-page text.
                if i < 30 and node.get('page_idx', 0) == 0:
                    # Keep first-page nodes within the initial 30-node region.
                    metadata_end = i + 1
                else:
                    # Stop when the node no longer fits the metadata region.
                    pass
        
        return (0, metadata_end)
    
    def _annotate_metadata(self, nodes: List[Dict]) -> int:
        """Annotate all nodes in the metadata region and return their count.
        
        Nodes do not need a text_level attribute.
        """
        metadata_start, metadata_end = self._identify_metadata_region(nodes)
        
        if metadata_end == 0:
            return 0
        
        print(f"识别到元数据区域: 节点 {metadata_start}-{metadata_end-1} (共{metadata_end-metadata_start}个节点)")
        
        annotated_count = 0
        for i in range(metadata_start, metadata_end):
            if i < len(nodes):
                # Annotate metadata regardless of text_level.
                nodes[i]['confidence'] = self.METADATA_CONFIDENCE
                nodes[i]['level'] = 1
                nodes[i]['classification'] = 'Authoritative'
                nodes[i]['section_type'] = 'Metadata'
                annotated_count += 1
        
        print(f"[OK] 已标注 {annotated_count} 个元数据节点 (置信度: {self.METADATA_CONFIDENCE})")
        
        # Print the first and last five metadata nodes.
        print(f"   元数据示例:")
        for i in range(min(3, metadata_end)):
            if i < len(nodes):
                text = nodes[i].get('text', '')[:40]
                print(f"     [{i}] {text}")
        if metadata_end > 6:
            print(f"     ...")
            for i in range(max(3, metadata_end-3), metadata_end):
                if i < len(nodes):
                    text = nodes[i].get('text', '')[:40]
                    print(f"     [{i}] {text}")
        print()
        
        return annotated_count
    
    def _build_section_credibility_prompt(self, section_title: str, document_org: str = "Unknown") -> str:
        """Build the prompt for section heading confidence classification."""
        
        prompt = f"""You are a credibility assessment system for sections within accident investigation reports. Analyze the SECTION TITLE to determine the credibility level of information in this specific section.

CLASSIFICATION SYSTEM (4 Levels):

- Level 1 (0.90-0.98): Authoritative - Factual, verified information sections
- Level 2 (0.75-0.89): Highly Credible - Analytical and interpretive sections
- Level 3 (0.50-0.74): Limited Credibility - Preliminary, predictive, or procedural sections
- Level 4 (0.15-0.49): Low Credibility - Party statements, defensive content

DECISION RULES BY SECTION TYPE (Ordered by Confidence Score):

LEVEL 1 (0.90-0.98) - FACTUAL/CONCLUSIVE SECTIONS:
IF section title contains (AND does NOT contain "Preliminary"|"Draft"|"Interim"):
→ 0.98: "Probable Cause"|"Determination"|"Conclusion"
→ 0.97: "Findings"|"Causal Factors"
→ 0.96: "Accident Description"|"Incident Description"|"Casualty Description"|"Accident Events"|"Casualty Events"|"Synopsis"|"Marine Accident Brief"|"Marine Accident Report"|"Summary"
→ 0.95: "Sequence of Events"|"Event Sequence"|"Chronology"|"Timeline"|"Accident Narrative"
→ 0.94: "Factual Information"|"Physical Evidence"|"Grounding"|"Fire"|"Collision"|"Sinking"|"Contact"|"Bar Restrictions"|"Bar Status"
→ 0.93: "Test Results"|"Examination Results"|"Laboratory Results"|"Toxicological Testing"|"Toxicological Tests"
→ 0.92: "Vessel Particulars"|"Aircraft Information"|"Ship Particulars"|"Vessel Information"|"Temporary Hull Repair"|"Hull Repair"|"Weather Conditions"|"Heavy Weather"|"Personnel Information"|"Crew Information"|"Equipment Information"|"Systems Information"
→ 0.91: "Wreckage"|"Debris"|"Wreckage and Impact Information"|"Structural Damage"|"Equipment Damage"|"Damage Assessment"
→ 0.90: "Damage"|"Injuries"|"Fatalities"|"Casualties"|"Origin of Fire"|"Fire Origin"|"Search and Rescue"|"Introduction"

LEVEL 2 (0.75-0.89) - ANALYTICAL SECTIONS:
IF section title contains (AND does NOT contain "Forecast"|"Predicted"|"My"|"Personal"):
→ 0.89: "Safety Issues"|"Safety Concerns"
→ 0.87: "Safety Recommendations"|"Recommendations"|"Conclusions" (when NOT containing "Probable Cause")
→ 0.85: "Technical Analysis"|"Forensic Analysis"
→ 0.83: "Analysis"|"Assessment"|"Evaluation"
→ 0.81: "Contributing Factors"|"Systemic Factors"
→ 0.80: "Waterway Information"|"Route Information"|"Background"|"River Conditions"|"Training"|"Instruction"|"Drills"|"Safety Orientation"|"Maintenance"|"Inspections"|"History"
→ 0.78: "Operational Information"|"Flight History"|"Voyage Information"|"Bridge Resource Management"|"Additional Information"|"Operational Procedures"|"Protocols"
→ 0.76: "Safety Board Actions"|"Regulatory Actions"
→ 0.75: "Discussion"|"Safety Discussion"|"Fire Protection Regulations"|"Human Factors"|"Crew Performance"

LEVEL 3 (0.50-0.74) - PRELIMINARY/PREDICTIVE SECTIONS:
IF section title contains (AND does NOT contain "My"|"CONFIDENTIAL"|"Attorney"):
→ 0.74: "Investigation Process"|"Methodology"
→ 0.72: "Witness Statements"|"Interviews"|"Accounts"|"Testimonies" | "Engine Failure"
→ 0.70: "Safety Board Actions"|"Board Meeting"|"Lessons Learned"
→ 0.68: "Emergency Response"|"Crew Evacuation"|"Towing Attempts"|"Attempts to Anchor"|"Vessel Adrift"
→ 0.66: "Corrective Actions"|"Postaccident Action"|"Post-accident Action"
→ 0.64: "Preliminary"|"Initial"|"Interim"|"Observations"
→ 0.62: "Planned"|"Intended"|"Scheduled"
→ 0.60: "Appendix"|"Annex"|"Attachment"|"References"|"Buoy Position"
→ 0.55: "Forecast"|"Weather Forecast"|"Projected"|"Expected"

LEVEL 4 (0.15-0.49) - LOW CREDIBILITY SECTIONS:
IF section title contains:
→ 0.45: "Captain's Statement"|"Operator's Statement"
→ 0.40: "Company Response"|"Company's Submission"|"Owner's Response"
→ 0.35: "My Statement"|"My View"|"Personal Statement"
→ 0.30: "Response to Allegations"|"Rebuttal"
→ 0.25: "Defense"|"Objections"|"Challenges to Findings"
→ 0.20: "CONFIDENTIAL"|"PRIVILEGED"|"Attorney-Client"
→ 0.15: "Refusal to Cooperate"|"Non-Compliance"

ENHANCED EXCLUSION CRITERIA:

Any title containing "Preliminary"|"Draft"|"Interim" automatically downgrades to LEVEL 3

Any title containing "Forecast"|"Predicted" automatically downgrades to LEVEL 3

Any title containing "My"|"Personal"|"Confidential" automatically downgrades to LEVEL 4

IMPORTANT CONSIDERATIONS:
1. Official agency reports (NTSB, AAIB, ATSB, TSB, Coast Guard) use scores as listed above
2. Company or party reports: reduce score by 0.10-0.15 from the matched title score
3. If multiple keywords match, use the highest applicable score
4. "Discussion" sections are analytical (Level 2), not factual (Level 1)
5. "Forecast" or "Predicted" information is Level 3, even in official reports
6. Procedural/administrative sections are Level 3, not Level 1
7. "Accident Events" is factual (Level 1 - 0.96), similar to "Accident Description"
8. "Background" is contextual/analytical (Level 2 - 0.80), not purely factual
9. "Additional Information" is analytical/contextual (Level 2 - 0.78)
10. "Lessons Learned" is interpretive guidance (Level 3 - 0.70)
11. "Training" or "Drills" sections are procedural (Level 3 - 0.68)
12. "Conclusions" without "Probable Cause" is analytical (Level 2 - 0.87)

INPUT:
Section Title: "{section_title}"
Issuing Organization: "{document_org}"

OUTPUT (JSON only, no explanation):
{{
  "level": [1|2|3|4],
  "confidence": [exact score like 0.98, 0.85, 0.60, 0.35],
  "classification": "[Authoritative|Highly Credible|Limited Credibility|Low Credibility]",
  "section_type": "[section type in 4 levles]",
  "reasoning": "[brief reason in 10-15 words]"
}}
"""
        return prompt
    
    def _detect_document_organization(self, nodes: List[Dict]) -> str:
        """Detect the organization that published the document."""
        # Collect text from the first 30 nodes.
        text_content = []
        for node in nodes[:30]:
            if node.get('type') == 'text' and node.get('text'):
                text_content.append(node['text'].strip())
        
        combined_text = ' '.join(text_content).lower()
        
        # Match known issuing organizations.
        if 'ntsb' in combined_text or 'national transportation safety board' in combined_text:
            return "NTSB"
        elif 'aaib' in combined_text or 'air accidents investigation branch' in combined_text:
            return "AAIB"
        elif 'atsb' in combined_text or 'australian transport safety bureau' in combined_text:
            return "ATSB"
        elif 'tsb' in combined_text or 'transportation safety board' in combined_text:
            return "TSB"
        elif 'bea' in combined_text or "bureau d'enquêtes" in combined_text:
            return "BEA"
        elif 'jtsb' in combined_text or 'japan transport safety board' in combined_text:
            return "JTSB"
        elif 'coast guard' in combined_text or 'uscg' in combined_text:
            return "Coast Guard"
        elif 'faa' in combined_text or 'federal aviation' in combined_text:
            return "FAA"
        elif 'easa' in combined_text or 'european aviation safety' in combined_text:
            return "EASA"
        elif 'maib' in combined_text or 'marine accident investigation' in combined_text:
            return "MAIB"
        else:
            return "Unknown Organization"
    
    def _is_valid_section_title(self, text: str, paragraph_count: int, metadata_end_idx: int, current_idx: int) -> bool:
        """Return whether text is a valid section heading.
        
        Args:
            text: Candidate heading text.
            paragraph_count: Number of following paragraphs.
            metadata_end_idx: End index of the metadata region.
            current_idx: Candidate node index.
        """
        text = text.strip()
        
        # Reject empty text.
        if not text:
            return False
        
        # Exclude nodes inside the metadata region.
        if current_idx < metadata_end_idx:
            return False
        
        # Reject very short text.
        if len(text) < 3:
            return False
        
        # Exclude metadata labels ending with a colon.
        if text.endswith(':'):
            return False
        
        # Exclude table fields containing multiple colons.
        if text.count(':') > 1:
            return False
        
        # Exclude date formats.
        date_patterns = [
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',
            r'^[A-Z][a-z]+ \d{1,2}, \d{4}$',
            r'^\d{4}-\d{2}-\d{2}$'
        ]
        for pattern in date_patterns:
            if re.match(pattern, text):
                return False
        
        # Exclude time formats.
        if re.match(r'^\d{4} ', text):
            return False
        
        # Exclude identifier formats.
        if re.match(r'^[A-Z]{2,}-\d+-[A-Z]{2,}-\d+$', text):
            return False
        
        # Exclude labels such as "Adopted:" and "Issued:".
        if re.match(r'^(Adopted|Approved|Issued|Published|Effective):', text, re.IGNORECASE):
            return False
        
        # Recognize common section heading keywords.
        section_keywords = [
            'description', 'information', 'background', 'analysis', 'discussion',
            'findings', 'conclusion', 'summary', 'overview', 'details', 'narrative',
            'investigation', 'assessment', 'evaluation', 'recommendations',
            'actions', 'cause', 'factors', 'circumstances',
            'weather', 'vessel', 'aircraft', 'crew', 'personnel', 'timeline', 
            'sequence', 'chronology', 'response', 'search', 'rescue', 'damage', 
            'injuries', 'fatalities', 'casualties', 'examination', 'tests',
            'waterway', 'route', 'forecast', 'conditions', 'witness', 'statements',
            'probable', 'determination', 'regulatory', 'safety', 'human factors',
            'wreckage', 'fire protection', 'origin', 'aftermath', 'evacuation',
            'events', 'contact', 'collision', 'grounding', 'sinking', 'fire',
            'repair', 'training', 'instruction', 'drills', 'orientation',
            'lessons', 'postaccident', 'river', 'hull', 'temporary',
            'introduction', 'synopsis',  # Common section headings
            # Common single-word headings
            'operation', 'operations', 'maintenance', 'design', 'construction',
            'testing', 'inspection', 'procedure', 'procedures', 'system', 'systems',
            'equipment', 'machinery', 'engine', 'boiler', 'propulsion',
            'navigation', 'communication', 'cargo', 'passenger', 'medical',
            'emergency', 'survival', 'lifeboat', 'lifesaving'
        ]
        
        text_lower = text.lower()
        has_keyword = any(keyword in text_lower for keyword in section_keywords)
        
        # Accept keyword headings of suitable length, even without following paragraphs.
        if has_keyword and 5 <= len(text) <= 100:
            return True
        
        # Other headings generally require following paragraphs.
        if paragraph_count == 0:
            # A section keyword allows paragraph_count to be zero.
            if has_keyword:
                return True
            else:
                return False
        
        # Without keywords, require title-like casing, length, and following paragraphs.
        words = text.split()
        if (len(words) >= 1 and len(words) <= 10 and 
            text[0].isupper() and 
            paragraph_count >= 2):
            return True
        
        return False
    
    def _call_qwen_model(self, prompt: str) -> Dict:
        """Ask Qwen to classify the heading confidence."""
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract the JSON response.
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                print(f"[WARNING] 无法解析模型输出: {response[:100]}...")
                return {
                    "confidence": 0.70, 
                    "level": 2,
                    "classification": "Highly Credible",
                    "section_type": "General",
                    "reasoning": "Unable to parse model output"
                }
                
        except Exception as e:
            print(f"[ERROR] 模型调用失败: {e}")
            return {
                "confidence": 0.70, 
                "level": 2,
                "classification": "Highly Credible",
                "section_type": "General",
                "reasoning": "Model call failed"
            }
    
    def _identify_title_paragraphs(self, nodes: List[Dict], metadata_end_idx: int) -> List[Dict]:
        """Find the paragraph range for each valid section heading."""
        title_ranges = []
        
        for i, node in enumerate(nodes):
            if node.get('text_level') == 1:
                title_text = node.get('text', '').strip()
                
                # Find the next heading.
                next_title_idx = None
                for j in range(i + 1, len(nodes)):
                    if nodes[j].get('text_level') == 1:
                        next_title_idx = j
                        break
                
                if next_title_idx is None:
                    next_title_idx = len(nodes)
                
                # Paragraph range
                paragraph_start = i + 1
                paragraph_end = next_title_idx
                paragraph_count = paragraph_end - paragraph_start
                
                # Validate the heading and exclude metadata.
                if self._is_valid_section_title(title_text, paragraph_count, metadata_end_idx, i):
                    title_ranges.append({
                        'title_index': i,
                        'title_text': title_text,
                        'paragraph_start': paragraph_start,
                        'paragraph_end': paragraph_end,
                        'paragraph_count': paragraph_count
                    })
        
        return title_ranges
    
    def process_file(self, input_file: str, output_file: str = None) -> Dict:
        """Annotate confidence values in a single JSON document."""
        
        print(f"读取文件: {input_file}")
        
        # Read document nodes.
        with open(input_file, 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        
        print(f"[OK] 成功读取 {len(nodes)} 个节点")
        
        # Detect the issuing organization.
        document_org = self._detect_document_organization(nodes)
        print(f"检测到发布组织: {document_org}")
        
        # Identify and annotate metadata.
        print("\n" + "="*60)
        print("第一步：识别元数据区域")
        print("="*60)
        metadata_count = self._annotate_metadata(nodes)
        metadata_start, metadata_end = self._identify_metadata_region(nodes)
        
        # Find section headings and paragraph ranges outside metadata.
        print("="*60)
        print("第二步：识别章节标题")
        print("="*60)
        print("识别章节标题和段落范围...")
        title_ranges = self._identify_title_paragraphs(nodes, metadata_end)
        print(f"[OK] 发现 {len(title_ranges)} 个有效章节标题\n")
        
        if len(title_ranges) == 0:
            print("[WARNING] 未找到有效的章节标题\n")
        else:
            # Classify each section heading.
            print("="*60)
            print("第三步：评估章节置信度")
            print("="*60)
            for idx, title_info in enumerate(title_ranges, 1):
                title_text = title_info['title_text']
                
                print(f"[{idx}/{len(title_ranges)}] 评估章节: {title_text}")
                
                # Request the heading confidence.
                prompt = self._build_section_credibility_prompt(title_text, document_org)
                credibility_result = self._call_qwen_model(prompt)
                
                confidence_value = credibility_result.get('confidence', 0.70)
                level = credibility_result.get('level', 2)
                classification = credibility_result.get('classification', 'Highly Credible')
                section_type = credibility_result.get('section_type', 'General')
                reasoning = credibility_result.get('reasoning', '')
                
                print(f"   [OK] 置信度: {confidence_value:.2f} (Level {level} - {classification})")
                print(f"   类型: {section_type}")
                print(f"   原因: {reasoning}")
                
                # Attach confidence metadata to the section nodes.
                paragraph_start = title_info['paragraph_start']
                paragraph_end = title_info['paragraph_end']
                title_index = title_info['title_index']
                
                # Count newly annotated nodes.
                annotated_count = 0
                
                # Consecutive headings have an empty paragraph range.
                # Annotate the heading itself if it has no confidence metadata.
                if paragraph_start == paragraph_end:
                    if 'confidence' not in nodes[title_index]:
                        nodes[title_index]['confidence'] = confidence_value
                        nodes[title_index]['level'] = level
                        nodes[title_index]['classification'] = classification
                        nodes[title_index]['section_type'] = section_type
                        annotated_count += 1
                        print(f"   [WARNING] 连续标题：将置信度标注到标题节点本身 (索引 {title_index})")
                else:
                    # Annotate paragraph content.
                    for i in range(paragraph_start, paragraph_end):
                        if i < len(nodes):
                            # Preserve existing annotations, including metadata confidence.
                            if 'confidence' not in nodes[i]:
                                nodes[i]['confidence'] = confidence_value
                                nodes[i]['level'] = level
                                nodes[i]['classification'] = classification
                                nodes[i]['section_type'] = section_type
                                annotated_count += 1
                
                print(f"   [OK] 已标注 {annotated_count} 个节点\n")
        
        # Save the annotated document.
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_annotated.json"
        
        print(f"保存结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(nodes, f, ensure_ascii=False, indent=2)
        
        # Annotation statistics
        annotated_count = sum(1 for node in nodes if 'confidence' in node)
        confidence_stats = {}
        section_type_stats = {}
        
        for node in nodes:
            if 'confidence' in node:
                # Counts by confidence level
                level = node.get('level', 0)
                if level not in confidence_stats:
                    confidence_stats[level] = {'count': 0, 'total_conf': 0}
                confidence_stats[level]['count'] += 1
                confidence_stats[level]['total_conf'] += node['confidence']
                
                # Counts by section type
                section_type = node.get('section_type', 'Unknown')
                if section_type not in section_type_stats:
                    section_type_stats[section_type] = 0
                section_type_stats[section_type] += 1
        
        print("\n" + "=" * 60)
        print("置信度分布统计:")
        print("=" * 60)
        for level in sorted(confidence_stats.keys()):
            stats = confidence_stats[level]
            avg_conf = stats['total_conf'] / stats['count']
            print(f"Level {level}: {stats['count']} 个节点, 平均置信度 {avg_conf:.3f}")
        
        print("\n" + "=" * 60)
        print("章节类型分布:")
        print("=" * 60)
        for section_type, count in sorted(section_type_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / annotated_count) * 100
            print(f"{section_type}: {count} 个节点 ({percentage:.1f}%)")
        print("=" * 60 + "\n")
        
        stats = {
            'input_file': input_file,
            'output_file': output_file,
            'document_org': document_org,
            'total_nodes': len(nodes),
            'metadata_count': metadata_count,
            'total_sections': len(title_ranges),
            'annotated_nodes': annotated_count,
            'unannotated_nodes': len(nodes) - annotated_count,
            'confidence_distribution': confidence_stats,
            'section_type_distribution': section_type_stats
        }
        
        return stats


class BatchProcessor:
    """Process batches of JSON documents."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize the batch processor."""
        self.annotator = SectionCredibilityAnnotator(model_path, device)
        self.processing_log = []
    
    def process_folder(
        self, 
        input_folder: str, 
        output_folder: str = None,
        pattern: str = "*_content_list.json"
    ) -> Dict:
        """Annotate all matching JSON files in a directory."""
        
        print("=" * 80)
        print("批量处理开始")
        print("=" * 80)
        print()
        
        input_path = Path(input_folder)
        
        # Select the output directory.
        if output_folder is None:
            output_path = input_path / "annotated_results"
        else:
            output_path = Path(output_folder)
        
        # Create the output directory.
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"输入文件夹: {input_path}")
        print(f"输出文件夹: {output_path}\n")
        
        # Find matching JSON files.
        json_files = list(input_path.glob(pattern))
        
        if not json_files:
            print(f"[WARNING] 未找到匹配的文件: {pattern}")
            return {}
        
        print(f"[OK] 找到 {len(json_files)} 个文件\n")
        
        # Process each file.
        all_stats = []
        success_count = 0
        error_count = 0
        
        for idx, json_file in enumerate(json_files, 1):
            print("=" * 80)
            print(f"处理文件 [{idx}/{len(json_files)}]: {json_file.name}")
            print("=" * 80)
            print()
            
            try:
                # Build the output filename.
                output_file = output_path / f"{json_file.stem}_annotated.json"

                if output_file.exists() and output_file.stat().st_size > 0:
                    print(f"⏭️ 输出已存在，跳过: {output_file.name}\n")
                    continue
                
                # Annotate the file.
                stats = self.annotator.process_file(
                    str(json_file),
                    str(output_file)
                )
                
                stats['status'] = 'success'
                stats['error'] = None
                all_stats.append(stats)
                success_count += 1
                
                print(f"[OK] 文件处理成功！")
                print(f"   - 发布组织: {stats['document_org']}")
                print(f"   - 总节点: {stats['total_nodes']}")
                print(f"   - 元数据节点: {stats['metadata_count']}")
                print(f"   - 章节数: {stats['total_sections']}")
                print(f"   - 已标注: {stats['annotated_nodes']}\n")
                
            except Exception as e:
                print(f"[ERROR] 文件处理失败: {e}\n")
                import traceback
                traceback.print_exc()
                all_stats.append({
                    'input_file': str(json_file),
                    'status': 'error',
                    'error': str(e)
                })
                error_count += 1
        
        # Save the processing log.
        log_file = output_path / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'input_folder': str(input_path),
                'output_folder': str(output_path),
                'pattern': pattern,
                'total_files': len(json_files),
                'success_count': success_count,
                'error_count': error_count,
                'file_stats': all_stats
            }, f, ensure_ascii=False, indent=2)
        
        # Print batch totals.
        print("\n" + "=" * 80)
        print("批量处理总结")
        print("=" * 80)
        print(f"[OK] 成功: {success_count} 个文件")
        print(f"[ERROR] 失败: {error_count} 个文件")
        print(f"处理日志: {log_file}")
        print("=" * 80)
        
        return {
            'success_count': success_count,
            'error_count': error_count,
            'all_stats': all_stats,
            'log_file': str(log_file)
        }


def main():
    """Run batch annotation using the configured paths."""
    
    # Runtime configuration
    # Resolve paths relative to this script.
    import pathlib
    PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
    
    MODEL_PATH = "/mnt/data/LLM/models/Qwen/Qwen2.5-7B-Instruct"
    INPUT_FOLDER = str(PROJECT_ROOT / "Input")
    OUTPUT_FOLDER = str(PROJECT_ROOT / "Document_Structural_Parsing_Output")
    PATTERN = "*_content_list.json"
    
    print("=" * 80)
    print("JSON文档章节与元数据置信度标注工具")
    print("=" * 80)
    print()
    
    # Check the input directory.
    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] 错误: 找不到输入文件夹 {INPUT_FOLDER}")
        return
    
    # Initialize the batch processor.
    processor = BatchProcessor(
        model_path=MODEL_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Process the batch.
    result = processor.process_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        pattern=PATTERN
    )
    
    print("\n批量处理完成！")


if __name__ == "__main__":
    main()
