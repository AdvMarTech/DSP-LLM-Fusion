"""Parse HTML tables and text-based tables from document JSON.

Use BeautifulSoup for HTML structure and Qwen for text and irregular
front-page tables. Preserve multi-part values and split combined fields
such as Owner/Operator into separate output keys.
"""

import json
import re
import os
from typing import Dict, List, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup


class EnhancedQwenTableParser:
    """Parse document tables with HTML parsing and Qwen extraction."""
    
    def __init__(
        self, 
        model_path: str = "/mnt/data/LLM/models/Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        newline_threshold: int = 8
    ):
        """Initialize the table parser.
        
        Args:
            model_path: Path to the Qwen model.
            device: Inference device.
            newline_threshold: Newline count used to identify text tables.
        """
        self.newline_threshold = newline_threshold
        self.device = device
        
        print(f"正在加载Qwen模型: {model_path}")
        
        # Load the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load the model.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).eval()
        
        print("[OK] 模型加载完成!\n")
        
        # Base field keywords, without trailing colons.
        # Matching accepts keywords with or without colons.
        self.table_keywords_base = [
            "Date", "Time", "Owner", "Operator", "Owner/Operator", "Damages", 
            "Crew Complement", "Complement", "Injuries", "Vessel", "Location", 
            "Accident Type", "Property Damage", "Accident No.",
            "Fatalities", "Fatalities/Injuries"  # Include the combined casualty label.
        ]
        
        # Keep colon-suffixed forms for matching contexts that need them.
        self.table_keywords = [kw + ":" for kw in self.table_keywords_base] + \
                              [kw for kw in self.table_keywords_base]
    
    def is_table(self, json_node: Dict) -> tuple[bool, str]:
        """Return whether a node is a table and its type.
        
        Types are html_table, text_table, irregular_table, and not_table.
        """
        # Accept explicitly marked table nodes.
        if json_node.get("type") == "table":
            return True, "html_table"
        
        # Check irregular front-page tables before generic text tables.
        # Front-page field labels identify irregular tables.
        # This precedence prevents the same page from being parsed twice.
        if json_node.get("page_idx") == 0:
            text = json_node.get("text", "")
            
            # Look for a base field keyword.
            for keyword in self.table_keywords_base:
                # Accept a colon, space, or newline after the keyword.
                if keyword + ":" in text or keyword + " " in text or keyword + "\n" in text:
                    return True, "irregular_table"
        
        # Use the newline threshold for text nodes outside page 0.
        if json_node.get("type") == "text":
            text = json_node.get("text", "")
            newline_count = text.count("\n")
            
            if newline_count >= self.newline_threshold:
                return True, "text_table"
        
        return False, "not_table"
    
    def _get_field_extraction_hints(self, keyword: str) -> str:
        """Return extraction instructions that distinguish easily confused fields."""
        keyword_lower = keyword.lower().rstrip(':').strip()
        
        # Field-specific extraction constraints
        hints = {
            "complement": """
6. 【重要】"Complement"或"Crew Complement"字段只包含人员配置数量信息，例如：
   - "22 crew, 2 pilots"
   - "911 crew; 2,135 passengers"
   - "3 crew, 4 passengers"
   【注意】不要包含伤亡信息（fatalities, injuries, deaths）！伤亡信息属于"Injuries"或"Fatalities"字段。""",
            
            "crew complement": """
6. 【重要】"Crew Complement"字段只包含船员/乘客配置数量，例如：
   - "22 crew"
   - "911 crew; 2,135 passengers"
   【注意】不要包含伤亡信息（fatalities, injuries, deaths）！""",
            
            "injuries": """
6. 【重要】"Injuries"字段包含伤亡信息，例如：
   - "8 fatalities; 10 serious injuries; 7 minor injuries"
   - "1 serious, 6 fatal"
   - "3 fatalities"
   【注意】不要包含人员配置数量（crew complement）！""",
            
            "fatalities": """
6. 【重要】"Fatalities"字段包含死亡信息，例如：
   - "8 fatalities"
   - "3 fatalities"
   【注意】不要包含人员配置数量（crew, passengers）！""",
            
            "fatalities/injuries": """
6. 【重要】"Fatalities/Injuries"字段包含伤亡信息，例如：
   - "8 fatalities; 10 serious injuries; 7 minor injuries"
   - "3 fatalities; 2 serious injuries"
   - "1 fatal, 2 serious"
   【注意】不要包含人员配置数量（crew, passengers, complement）！这是伤亡统计字段。""",
            
            "crew": """
6. 【重要】如果是单独的"Crew"字段（不是Crew Complement），它通常指船员数量，例如：
   - "22"
   - "8 crew members"
   【注意】不要包含伤亡信息！伤亡信息属于"Injuries"或"Fatalities"字段。""",
            
            "accident no": """
6. 【重要】"Accident No."字段是事故编号，格式通常为：
   - "DCA-06-MF-016"
   - "DCA-03-MM-032"
   【注意】不要包含"Vessel:"或其他字段名前缀！只提取编号本身。""",
            
            "accident no.": """
6. 【重要】"Accident No."字段是事故编号，格式通常为：
   - "DCA-06-MF-016"
   - "DCA-03-MM-032"
   【注意】不要包含"Vessel:"或其他字段名前缀！只提取编号本身。""",
            
            "owner/operator": """
6. 【重要】"Owner/Operator"是一个组合字段，表示船舶的所有者和运营者可能是同一个实体。
   该字段的值就是公司/组织的名称，例如：
   - "Fire Island Ferries"
   - "Maersk Line"
   - "Carnival Corporation"
   【注意】只提取公司/组织名称，不要包含其他字段信息。"""
        }
        
        return hints.get(keyword_lower, """
6. 对于Complement等字段，值通常是人员配置信息（不含伤亡）
7. 对于Injuries/Fatalities字段，值是伤亡信息""")
    
    def extract_semantic_value(self, keyword: str, context_nodes: List[Dict], keyword_node_idx: int) -> str:
        """Use Qwen to extract a keyword value from neighboring front-page nodes.
        
        Args:
            keyword: Field label, such as "Date:" or "Vessel:".
            context_nodes: Nodes on page 0.
            keyword_node_idx: Index of the node containing the keyword.
        
        Returns:
            The extracted value.
        """
        # Use neighboring nodes as extraction context.
        context_window = 8  # Include up to eight nodes on either side.
        start_idx = max(0, keyword_node_idx - context_window)
        end_idx = min(len(context_nodes), keyword_node_idx + context_window + 1)
        
        context_texts = []
        for i in range(start_idx, end_idx):
            node = context_nodes[i]
            text = node.get("text", "").strip()
            if text:
                # Mark the node containing the field label.
                if i == keyword_node_idx:
                    context_texts.append(f"[KEYWORD_NODE]{text}[/KEYWORD_NODE]")
                else:
                    context_texts.append(f"[NODE_{i}]{text}[/NODE_{i}]")
        
        context_str = "\n".join(context_texts)
        
        # Build the extraction prompt.
        # Include constraints for easily confused fields.
        field_hints = self._get_field_extraction_hints(keyword)
        
        prompt = f"""请从以下上下文中提取字段"{keyword}"对应的完整值。

上下文文本（每个节点用标签标记）:
{context_str}

重要说明:
1. [KEYWORD_NODE]标记了包含字段名"{keyword}"的节点
2. 字段名可能以冒号":"结尾，也可能直接后跟空格或换行
3. 对应的值可能：
   - 在同一节点中，紧跟字段名之后
   - 在相邻的独立节点中
   - 跨越多个节点（需要合并）
   - 在同一节点的多行文本中（换行符分隔）
4. 对于Property Damage等字段，值可能包含多个条目（如多艘船的损失），请完整提取所有条目
5. 如果值有多行，用换行符\\n连接或用逗号分隔
{field_hints}

要求:
- 只输出提取的完整值，不要任何解释
- 不要包含字段名本身（如不要以"Vessel:"、"Accident No."等开头）
- 不要添加引号或其他修饰
- 如果值包含多个部分，用分号或换行连接
- 如果确实没有找到值，返回空字符串
- 【重要】只提取属于"{keyword}"字段的值，不要混入其他字段的内容

现在请提取"{keyword}"的完整值:"""

        try:
            # Build the chat messages.
            messages = [
                {"role": "system", "content": "你是一个专业的信息提取专家，擅长从文本中准确、完整地提取特定字段的值。你会仔细分析上下文，确保提取的值是完整的，不会遗漏任何部分。"},
                {"role": "user", "content": prompt}
            ]
            
            # Apply the tokenizer's chat template.
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer(
                [text_input],
                return_tensors="pt"
            ).to(self.device)
            
            # Allow enough output tokens for multi-part values.
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,  # Allow longer field values.
                do_sample=False
            )
            
            # Remove the input tokens from the generated sequence.
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    model_inputs.input_ids, 
                    generated_ids
                )
            ]
            
            # Decode the generated tokens.
            response = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            return response
            
        except Exception as e:
            print(f"  [WARNING] 语义提取失败 ({keyword}): {e}")
            return ""
    
    def parse_irregular_table_with_semantic(self, all_page0_nodes: List[Dict]) -> Dict[str, str]:
        """Extract fields from irregular tables using all page-0 nodes."""
        print("  使用语义判断解析不规范表格...")
        
        result = {}
        keyword_nodes = {}  # Record the node index for each keyword.
        
        # Find nodes containing base field keywords.
        for idx, node in enumerate(all_page0_nodes):
            text = node.get("text", "")
            for keyword in self.table_keywords_base:
                # Accept keywords followed by a colon, space, or newline, or at the end of text.
                if (keyword + ":" in text or 
                    keyword + " " in text or 
                    keyword + "\n" in text or
                    text.strip().endswith(keyword) or
                    text.strip() == keyword):
                    # A node can contain several field labels.
                    if idx not in keyword_nodes:
                        keyword_nodes[idx] = []
                    # Store the base keyword without a colon.
                    if keyword not in keyword_nodes[idx]:
                        keyword_nodes[idx].append(keyword)
        
        print(f"  找到 {len(keyword_nodes)} 个包含关键词的节点")
        
        # Print detected keywords for diagnostics.
        for node_idx, keywords in keyword_nodes.items():
            node_text = all_page0_nodes[node_idx].get("text", "")[:50]
            print(f"    节点 {node_idx}: {keywords} (文本: {node_text}...)")
        
        # Extract a value for each keyword.
        processed_keywords = set()
        for node_idx, keywords in keyword_nodes.items():
            for keyword in keywords:
                if keyword in processed_keywords:
                    continue
                
                print(f"  提取 {keyword} 的值...")
                value = self.extract_semantic_value(keyword, all_page0_nodes, node_idx)
                
                if value:
                    # Remove any trailing colon from the key.
                    clean_key = keyword.rstrip(':').strip()
                    # Normalize whitespace while retaining multi-part values.
                    # Pass the current keyword to avoid removing its value incorrectly.
                    clean_value = self._clean_extracted_value(value, keyword)
                    result[clean_key] = clean_value
                    processed_keywords.add(keyword)
                    print(f"    [OK] {clean_key}: {clean_value[:80]}{'...' if len(clean_value) > 80 else ''}")
                else:
                    print(f"    [ERROR] {keyword}: 未找到值")
        
        # Split Owner/Operator into separate fields.
        result = self._split_combined_fields(result)
        
        return result
    
    def _split_combined_fields(self, result: Dict[str, str]) -> Dict[str, str]:
        """Split combined fields such as Owner/Operator into separate keys.
        
        Existing values for individual fields are retained.
        """
        # Map each combined label to its component fields.
        combined_fields = {
            "Owner/Operator": ["Owner", "Operator"],
        }
        
        new_result = {}
        
        for key, value in result.items():
            # Check for a combined field.
            if key in combined_fields:
                split_fields = combined_fields[key]
                print(f"    拆分组合字段 '{key}' -> {split_fields}")
                
                # Assign the combined value to each component.
                for field_name in split_fields:
                    # Keep any value already recorded for that field.
                    if field_name not in result and field_name not in new_result:
                        new_result[field_name] = value
                        print(f"       [OK] {field_name}: {value[:50]}{'...' if len(value) > 50 else ''}")
                    else:
                        print(f"       ⏭️  {field_name}: 已存在，跳过")
            else:
                # Preserve ordinary fields.
                new_result[key] = value
        
        return new_result
    
    def _clean_extracted_value(self, value: str, current_keyword: str = None) -> str:
        """Normalize an extracted value and remove unrelated field prefixes.
        
        Args:
            value: Raw extracted value.
            current_keyword: Field being extracted; its prefix is not removed.
        """
        if not value:
            return value
        
        # Trim surrounding whitespace.
        value = value.strip()
        
        # Collapse consecutive spaces.
        value = re.sub(r' {2,}', ' ', value)
        
        # Replace consecutive newlines with a separator.
        value = re.sub(r'\n{2,}', '\n', value)
        
        # Remove matching surrounding quotes.
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        
        # Remove unrelated field labels accidentally included in the value.
        # For example, "Vessel:DCA-06-MF-016" becomes "DCA-06-MF-016".
        for keyword in self.table_keywords_base:
            # Leave the current field label untouched.
            if current_keyword and keyword.lower() == current_keyword.lower().rstrip(':').strip():
                continue
            
            # Check colon-separated prefixes.
            prefix_with_colon = keyword + ":"
            if value.startswith(prefix_with_colon):
                value = value[len(prefix_with_colon):].strip()
                break
            
            # Check space-separated prefixes.
            prefix_with_space = keyword + " "
            if value.startswith(prefix_with_space) and len(value) > len(prefix_with_space):
                # Remove only when the following character is not a letter.
                rest = value[len(prefix_with_space):]
                if rest and (rest[0].isdigit() or rest[0] in ':-$'):
                    value = rest.strip()
                    break
        
        return value
    
    def parse_html_table(self, json_node: Dict) -> Dict[str, any]:
        """Parse an HTML table node into rows and structured records."""
        print("  使用HTML解析方法...")
        
        try:
            # Look for HTML under the supported field names.
            html_content = (
                json_node.get("html", "") or 
                json_node.get("table_body", "") or 
                json_node.get("table_html", "") or
                json_node.get("text", "")
            )
            
            if not html_content:
                return {"error": "No HTML content found. Available keys: " + ", ".join(json_node.keys())}
            
            # Parse the HTML with BeautifulSoup.
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')
            
            if not table:
                # Fall back to text parsing if no table element exists.
                print("  [WARNING]  未找到table标签,切换到文本解析")
                return self.parse_text_table(json_node)
            
            # Extract table data.
            result = {
                "parse_method": "html",
                "rows": []
            }
            
            # Read each row.
            all_rows = table.find_all('tr')
            raw_rows = []
            
            # Keep raw cell values and rowspan/colspan attributes.
            for tr in all_rows:
                cells = tr.find_all(['td', 'th'])
                row_data = []
                for cell in cells:
                    cell_info = {
                        "text": cell.get_text(strip=True),
                        "rowspan": int(cell.get("rowspan", 1)),
                        "colspan": int(cell.get("colspan", 1)),
                        "is_header": cell.name == 'th'
                    }
                    row_data.append(cell_info)
                if row_data:
                    raw_rows.append(row_data)
            
            # Build the simplified row representation.
            for row_info in raw_rows:
                simple_row = [cell["text"] for cell in row_info]
                if simple_row:
                    result["rows"].append(simple_row)
            
            result["raw_structure"] = raw_rows
            
            # Determine table orientation and build records.
            if len(result["rows"]) > 0:
                first_row = result["rows"][0]
                
                # Identify tables with field names in the first column.
                is_vertical = False
                if len(result["rows"]) >= 2 and len(first_row) > 1:
                    first_column_values = [row[0] for row in result["rows"]]
                    # Common field label keywords
                    field_keywords = [
                        'type', 'name', 'date', 'time', 'owner', 'operator', 
                        'damage', 'injury', 'vessel', 'flag', 'location', 
                        'complement', 'crew', 'property', 'cargo', 'port',
                        'casualties', 'no', 'number'
                    ]
                    
                    matching_count = sum(
                        1 for val in first_column_values 
                        if any(keyword in val.lower() for keyword in field_keywords)
                    )
                    
                    # Treat the table as vertical when over 30% of first-column cells match.
                    if matching_count / len(first_column_values) > 0.3:
                        is_vertical = True
                
                if is_vertical:
                    # Parse vertical tables.
                    result["table_orientation"] = "vertical"
                    result["field_names"] = [row[0] for row in result["rows"]]
                    
                    # Each remaining column describes a separate entity.
                    if len(first_row) > 1:
                        result["entity_names"] = first_row[1:]
                        
                        # Create one record per entity.
                        result["structured_data"] = []
                        
                        for col_idx in range(1, len(first_row)):
                            entity_dict = {}
                            entity_dict["__entity_name__"] = first_row[col_idx]
                            
                            for row_idx, row in enumerate(result["rows"]):
                                field_name = row[0]
                                value = row[col_idx] if col_idx < len(row) else ""
                                entity_dict[field_name] = value
                            
                            result["structured_data"].append(entity_dict)
                else:
                    # Parse horizontal tables.
                    result["table_orientation"] = "horizontal"
                    result["headers"] = first_row
                    
                    if len(result["rows"]) > 1:
                        result["structured_data"] = []
                        
                        for row in result["rows"][1:]:
                            row_dict = {}
                            
                            for i in range(max(len(first_row), len(row))):
                                header = first_row[i] if i < len(first_row) else f"Column_{i+1}"
                                value = row[i] if i < len(row) else ""
                                row_dict[header] = value
                            
                            result["structured_data"].append(row_dict)
            
            # Include table captions and notes.
            if "table_caption" in json_node and json_node.get("table_caption"):
                result["caption"] = json_node["table_caption"]
            
            if "table_footnote" in json_node and json_node.get("table_footnote"):
                result["footnote"] = json_node["table_footnote"]
            
            return result
            
        except Exception as e:
            print(f"  [ERROR] HTML解析失败: {e}")
            return {"error": str(e), "parse_method": "html"}
    
    def preprocess_table_text(self, text: str) -> str:
        """Normalize line endings and whitespace in table text."""
        # Normalize line endings.
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Collapse consecutive spaces.
        text = re.sub(r' {2,}', ' ', text)
        
        # Trim each line.
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Retain a single blank line between text blocks.
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def parse_text_table(self, json_node: Dict) -> Dict[str, str]:
        """Parse a text table with Qwen, using a prompt for two-column layouts."""
        print("  使用Qwen语义解析方法...")
        
        text = json_node.get("text", "")
        
        if not text:
            return {"error": "No text content found"}
        
        try:
            # Normalize the input text.
            text = self.preprocess_table_text(text)
            
            # Build the parsing prompt.
            prompt = self._create_parsing_prompt_v2(text)
            
            # Build the chat messages.
            messages = [
                {"role": "system", "content": "你是一个专业的表格数据解析专家,擅长从非结构化文本中提取结构化的键值对信息。"},
                {"role": "user", "content": prompt}
            ]
            
            # Apply the tokenizer's chat template.
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            model_inputs = self.tokenizer(
                [text_input],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate the response.
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=False
            )
            
            # Remove the input tokens from the generated sequence.
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    model_inputs.input_ids, 
                    generated_ids
                )
            ]
            
            # Decode the generated tokens.
            response = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            print(f"  模型原始输出:\n{response[:500]}...")
            
            # Extract JSON from the response.
            parsed_result = self._extract_json_v2(response)
            parsed_result["parse_method"] = "qwen_semantic"
            parsed_result["raw_model_output"] = response  # Keep the raw response for diagnostics.
            
            return parsed_result
            
        except Exception as e:
            print(f"  [ERROR] Qwen解析失败: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "parse_method": "qwen_semantic"}
    
    def _create_parsing_prompt_v2(self, table_text: str) -> str:
        """Build a parsing prompt for two-column text tables."""
        prompt = f"""请分析以下表格文本,这是一个两列结构的表格,左列是字段名(Field Name),右列是对应的值(Value)。

原始表格文本:
\"\"\"
{table_text}
\"\"\"

解析规则:
1. 这是一个两列表格: 左列=字段名, 右列=值
2. 字段名通常以冒号(:)结尾,如 "Date:", "Time:", "Owner:" 等
3. 字段名和值可能在同一行,也可能在相邻的行
4. 值可能跨越多行,需要合理组合
5. 常见字段包括但不限于: Date, Time, Owner, Operator, Damages, Crew Complement, Injuries, Vessel, Location, Accident Type 等

输出要求:
1. 输出标准JSON格式
2. 键(key)使用规范的字段名(去除冒号,统一格式,不要删除单词间的空格)
3. 值(value)完整准确,多行内容用空格连接
4. 不要添加任何解释性文字,只输出JSON

输出格式示例:
{{
    "Date": "December 8, 2004",
    "Time": "1705 Alaska standard time",
    "Owner": "Ayu Navigation Sdn. Bhd.",
    "Operator": "IMC Shipping Co. Pte. (private) Ltd.",
    "Damages": "$12 million vessel (total loss)",
    "Crew Complement": "26",
    "Injuries": "1 serious, 6 fatal"
}}

现在请开始解析,只输出JSON:"""
        
        return prompt
    
    def _extract_json_v2(self, text: str) -> Dict:
        """Extract a JSON dictionary from model output, repairing common errors."""
        # Look for a fenced JSON block.
        json_pattern = r'```(?:json)?\s*(.*?)\s*```'
        match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_str = match.group(1)
        else:
            # Look for the outermost braces.
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = text[start_idx:end_idx+1]
            else:
                # Try other candidate JSON objects.
                brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(brace_pattern, text, re.DOTALL)
                
                if matches:
                    # Prefer the longest match as the likely complete object.
                    json_str = max(matches, key=len)
                else:
                    json_str = text.strip()
        
        # Clean the candidate JSON string.
        json_str = self._clean_json_string(json_str)
        
        # Try parsing the candidate.
        try:
            result = json.loads(json_str)
            
            # Normalize successfully parsed data.
            result = self._postprocess_parsed_data(result)
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"  [WARNING]  JSON解析失败: {e}")
            print(f"  尝试解析的字符串: {json_str[:200]}...")
            
            # Repair common JSON syntax errors.
            fixed_json_str = self._fix_common_json_errors(json_str)
            
            try:
                result = json.loads(fixed_json_str)
                result = self._postprocess_parsed_data(result)
                return result
            except json.JSONDecodeError:
                # Fall back to extracting key-value pairs with regular expressions.
                return self._fallback_key_value_extraction(text)
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean punctuation, comments, and trailing commas in JSON text."""
        # Replace Chinese punctuation with JSON punctuation.
        json_str = json_str.replace('，', ',').replace('：', ':')
        json_str = json_str.replace('"', '"').replace('"', '"')
        json_str = json_str.replace(''', "'").replace(''', "'")
        
        # Remove comments.
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Remove trailing commas.
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str
    
    def _fix_common_json_errors(self, json_str: str) -> str:
        """Repair common syntax errors in generated JSON."""
        # Quote unquoted keys.
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
        
        # Replace single quotes.
        json_str = json_str.replace("'", '"')
        
        # Escape line breaks.
        json_str = json_str.replace('\n', ' ')
        
        # Remove redundant commas.
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def _postprocess_parsed_data(self, data: Dict) -> Dict:
        """Normalize keys and values in parsed table data."""
        if not isinstance(data, dict):
            return data
        
        cleaned_data = {}
        
        for key, value in data.items():
            # Clean field names.
            clean_key = key.strip().rstrip(':').strip()
            
            # Clean field values.
            if isinstance(value, str):
                # Collapse whitespace.
                clean_value = ' '.join(value.split())
                # Remove unwanted special characters.
                clean_value = clean_value.strip()
            else:
                clean_value = value
            
            cleaned_data[clean_key] = clean_value
        
        return cleaned_data
    
    def _fallback_key_value_extraction(self, text: str) -> Dict:
        """Extract key-value pairs with regular expressions if JSON parsing fails."""
        print("  [WARNING]  使用备用提取方案...")
        
        result = {}
        
        # Match quoted "Key": "Value" pairs.
        pattern1 = r'"([^"]+)"\s*:\s*"([^"]*)"'
        matches1 = re.findall(pattern1, text)
        
        for key, value in matches1:
            result[key.strip()] = value.strip()
        
        # Match unquoted Key: Value pairs.
        pattern2 = r'([A-Z][a-zA-Z\s]+):\s*([^\n]+)'
        matches2 = re.findall(pattern2, text)
        
        for key, value in matches2:
            key_clean = key.strip().rstrip(':')
            if key_clean not in result:  # Keep values already extracted by the more specific pattern.
                result[key_clean] = value.strip()
        
        return result
    
    def process_json_file(self, json_file_path: str) -> List[Dict]:
        """Parse tables from a JSON file and return the per-table results."""
        print(f"\n处理文件: {os.path.basename(json_file_path)}")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process list elements as document nodes.
            if isinstance(data, list):
                json_nodes = data
            # For dictionaries, look for the document node list.
            elif isinstance(data, dict):
                # Check common container keys.
                for key in ['elements', 'blocks', 'content', 'data']:
                    if key in data and isinstance(data[key], list):
                        json_nodes = data[key]
                        break
                else:
                    json_nodes = [data]
            else:
                json_nodes = [data]
            
            # Collect all page-0 nodes for irregular table parsing.
            page0_nodes = [node for node in json_nodes if node.get("page_idx") == 0]
            
            results = []
            table_count = 0
            
            # Parse the front-page irregular table only once.
            irregular_table_processed = False
            irregular_table_parsed_data = None
            irregular_table_node_indices = []  # Track all node indices belonging to that table.
            
            # Identify irregular table nodes before parsing.
            for idx, node in enumerate(json_nodes):
                is_table, table_type = self.is_table(node)
                if is_table and table_type == "irregular_table":
                    irregular_table_node_indices.append(idx)
            
            # Process the document nodes.
            for idx, node in enumerate(json_nodes):
                is_table, table_type = self.is_table(node)
                
                if is_table:
                    # Store the irregular table result at its first node only.
                    if table_type == "irregular_table":
                        if irregular_table_processed:
                            # Skip subsequent nodes belonging to the same irregular table.
                            continue
                        
                        # Parse the irregular table on its first occurrence.
                        irregular_table_processed = True
                        table_count += 1
                        print(f"\n  发现不规范表格 (涉及节点索引: {irregular_table_node_indices})")
                        
                        # Use the entire first page as extraction context.
                        irregular_table_parsed_data = self.parse_irregular_table_with_semantic(page0_nodes)
                        irregular_table_parsed_data["parse_method"] = "semantic_extraction"
                        
                        # Record the result and all related node indices.
                        result = {
                            "file": os.path.basename(json_file_path),
                            "node_index": idx,  # First detected node index
                            "related_node_indices": irregular_table_node_indices,  # All related node indices
                            "table_type": table_type,
                            "original_node": node,
                            "parsed_data": irregular_table_parsed_data
                        }
                        
                        results.append(result)
                        print(f"  [OK] 不规范表格解析完成 (合并了 {len(irregular_table_node_indices)} 个节点)")
                    else:
                        # Process HTML and text tables.
                        
                        # Check text tables for overlap with the irregular table.
                        skip_this_table = False
                        pre_parsed_data = None
                        
                        if table_type == "text_table" and irregular_table_processed and irregular_table_parsed_data:
                            # Only page-0 nodes can overlap the irregular table.
                            if node.get("page_idx") == 0:
                                # Parse the text table to compare its fields.
                                pre_parsed_data = self.parse_text_table(node)
                                
                                # Exclude parser metadata from the field set.
                                meta_fields = {"parse_method", "raw_model_output", "error"}
                                text_table_fields = set(pre_parsed_data.keys()) - meta_fields
                                irregular_table_fields = set(irregular_table_parsed_data.keys()) - meta_fields
                                
                                # Compute the fraction of overlapping fields.
                                if text_table_fields:
                                    overlap = text_table_fields & irregular_table_fields
                                    overlap_ratio = len(overlap) / len(text_table_fields)
                                    
                                    # Skip a text table when more than half its fields already exist.
                                    if overlap_ratio >= 0.5:
                                        print(f"\n  ⏭️  跳过重复表格 (索引: {idx}, 类型: {table_type})")
                                        print(f"     重叠字段: {overlap} (重叠率: {overlap_ratio:.1%})")
                                        skip_this_table = True
                        
                        if skip_this_table:
                            continue
                        
                        table_count += 1
                        print(f"\n  发现表格 #{table_count} (索引: {idx}, 类型: {table_type})")
                        
                        # Build the table result.
                        result = {
                            "file": os.path.basename(json_file_path),
                            "node_index": idx,
                            "table_type": table_type,
                            "original_node": node
                        }
                        
                        # Select the parser by table type.
                        if table_type == "html_table":
                            result["parsed_data"] = self.parse_html_table(node)
                        elif table_type == "text_table":
                            # Reuse the result from the overlap check when available.
                            if pre_parsed_data is not None:
                                result["parsed_data"] = pre_parsed_data
                            else:
                                result["parsed_data"] = self.parse_text_table(node)
                        
                        results.append(result)
                        print(f"  [OK] 解析完成")
            
            print(f"\n  文件统计: 总节点={len(json_nodes)}, 表格数={table_count}")
            return results
            
        except Exception as e:
            print(f"  [ERROR] 文件处理失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_folder(
        self, 
        folder_path: str,
        output_dir: str = None
    ) -> Dict:
        """Parse every JSON file in a directory and return batch statistics.
        
        Args:
            folder_path: Input directory.
            output_dir: Output directory; defaults to the input directory.
        """
        print(f"\n{'='*60}")
        print(f"开始批量处理: {folder_path}")
        print(f"{'='*60}")
        
        # Select the output directory.
        if output_dir is None:
            output_dir = folder_path
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"输出目录: {output_dir}\n")
        
        # Find JSON input files.
        json_files = list(Path(folder_path).glob("*.json"))
        
        if not json_files:
            print("[WARNING]  未找到JSON文件")
            return {}
        
        print(f"找到 {len(json_files)} 个JSON文件\n")
        
        # Process each file.
        stats = {
            "total_files": len(json_files),
            "processed_files": 0,
            "total_tables": 0,
            "html_tables": 0,
            "text_tables": 0,
            "irregular_tables": 0,
            "errors": 0,
            "output_files": []
        }
        
        for json_file in json_files:
            print(f"{'─'*60}")
            print(f"处理文件: {json_file.name}")

            input_filename = json_file.stem  # Filename without its extension
            output_filename = f"{input_filename}_table_parse.json"
            output_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"  ⏭️ 输出已存在，跳过: {output_filename}")
                stats["processed_files"] += 1
                stats["output_files"].append(output_filename)
                continue
            
            results = self.process_json_file(str(json_file))
            
            if results:
                # Append _table_parse.json to the input stem.
                input_filename = json_file.stem  # Filename without its extension
                output_filename = f"{input_filename}_table_parse.json"
                output_path = os.path.join(output_dir, output_filename)
                
                # Build the output record.
                file_stats = {
                    "total_tables": len(results),
                    "html_tables": sum(1 for r in results if r["table_type"] == "html_table"),
                    "text_tables": sum(1 for r in results if r["table_type"] == "text_table"),
                    "irregular_tables": sum(1 for r in results if r["table_type"] == "irregular_table")
                }
                
                output_data = {
                    "source_file": json_file.name,
                    "statistics": file_stats,
                    "results": results
                }
                
                # Save the parsed tables.
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"  已保存: {output_filename}")
                print(f"  表格数: {len(results)} (HTML: {file_stats['html_tables']}, Text: {file_stats['text_tables']}, Irregular: {file_stats['irregular_tables']})")
                
                # Update batch counts.
                stats["processed_files"] += 1
                stats["total_tables"] += len(results)
                stats["html_tables"] += file_stats["html_tables"]
                stats["text_tables"] += file_stats["text_tables"]
                stats["irregular_tables"] += file_stats["irregular_tables"]
                stats["output_files"].append(output_filename)
            else:
                print(f"  [WARNING]  未找到表格或处理失败")
                stats["errors"] += 1
        
        # Print batch totals.
        print(f"\n{'='*60}")
        print(f"批量处理完成统计")
        print(f"{'='*60}")
        print(f"总文件数: {stats['total_files']}")
        print(f"成功处理: {stats['processed_files']}")
        print(f"发现表格: {stats['total_tables']}")
        print(f"  - HTML表格: {stats['html_tables']}")
        print(f"  - 文本表格: {stats['text_tables']}")
        print(f"  - 不规范表格: {stats['irregular_tables']}")
        print(f"处理失败: {stats['errors']}")
        print(f"\n输出目录: {output_dir}")
        print(f"生成文件: {len(stats['output_files'])} 个")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """Run table parsing using the configured directories."""
    
    # Resolve paths relative to this script.
    import pathlib
    PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
    
    # Initialize the parser.
    parser = EnhancedQwenTableParser(
        model_path="/mnt/data/LLM/models/Qwen/Qwen2.5-7B-Instruct",
        device="cuda",
        newline_threshold=8  # Newline threshold for text tables
    )
    
    
    # Write a separate output for each input file.
    stats = parser.process_folder(
        folder_path=str(PROJECT_ROOT / "Document_Structural_Parsing_Output"),  # Input JSON directory
        output_dir=str(PROJECT_ROOT / "Table_Parsing_Output")  # Optional output directory
    )
    # Output filenames use the input stem followed by _table_parse.json.
    # Example: MAB0703_content_list.json -> MAB0703_content_list_table_parse.json.


if __name__ == "__main__":
    main()
