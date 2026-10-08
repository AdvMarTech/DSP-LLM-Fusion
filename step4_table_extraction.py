#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract maritime accident information from parsed tables with Qwen.

Supports HTML, text, and irregular tables using the shared accident schema.
Vessel records include personnel, casualties, and property damage.
Field validation and source-table checks resolve common extraction errors.
"""

import os
import sys
import json
import logging
import re
import html
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

# Input, output, and model configuration

# Resolve paths relative to this script.
import pathlib
_PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()

# Directory containing *_table_parse.json inputs
INPUT_DIR = str(_PROJECT_ROOT / "Table_Parsing_Output")

# Directory for *_extracted.json outputs
OUTPUT_DIR = str(_PROJECT_ROOT / "Table_Information_Extraction_Output")

# Qwen model path
DEFAULT_QWEN_MODEL_PATH = '/mnt/data/LLM/models/Qwen/Qwen2.5-7B-Instruct'

# Directory containing the shared data schema module
STRUCT_FILE_PATH = str(_PROJECT_ROOT / "maritime_accident_data_struct.py")


# Import the shared schema helpers.
# Try the configured directory, then the current module directory.
struct_dir = os.path.dirname(STRUCT_FILE_PATH)
struct_module_name = os.path.basename(STRUCT_FILE_PATH).replace('.py', '')

if struct_dir and struct_dir not in sys.path:
    sys.path.insert(0, struct_dir)

try:
    # Load the module by path.
    import importlib.util
    spec = importlib.util.spec_from_file_location(struct_module_name, STRUCT_FILE_PATH)
    if spec and spec.loader:
        struct_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(struct_module)
        
        # Import the required helpers.
        create_field_metadata = struct_module.create_field_metadata
        create_location_with_coordinates = struct_module.create_location_with_coordinates
        create_vessel_info = struct_module.create_vessel_info
        create_empty_accident_structure = struct_module.create_empty_accident_structure
        add_vessel_info = struct_module.add_vessel_info
        save_to_file = struct_module.save_to_file
        
        print(f"[OK] 成功从 {STRUCT_FILE_PATH} 导入数据结构")
    else:
        raise ImportError("无法加载模块规格")
        
except Exception as e:
    print(f"[WARNING] 无法从指定路径导入数据结构: {e}")
    print("尝试从当前目录导入 maritime_accident_data_struct...")
    try:
        from maritime_accident_data_struct import (
            create_field_metadata,
            create_location_with_coordinates,
            create_vessel_info,
            create_empty_accident_structure,
            add_vessel_info,
            save_to_file
        )
        print("[OK] 成功从当前目录导入数据结构")
    except ImportError as e2:
        print(f"[ERROR] 无法导入数据结构模块: {e2}")
        print("请确保 maritime_accident_data_struct.py 文件存在")
        sys.exit(1)

# Import model dependencies.
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("警告: transformers库未安装，将无法使用Qwen7B模型")

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Vessel type validation

# Valid vessel type keywords in lowercase
VALID_VESSEL_TYPES = [
    # Passenger vessels
    "passenger ship", "passenger vessel", "passenger ferry",
    "cruise ship", "cruise liner", "cruise vessel",
    "ferry", "ferryboat",
    # Cargo vessels
    "cargo ship", "cargo vessel", "general cargo",
    "container ship", "containership", "container vessel",
    "bulk carrier", "bulker", "bulk cargo",
    "tanker", "oil tanker", "chemical tanker", "lng tanker", "lpg tanker",
    "product tanker", "crude tanker",
    "ro-ro", "roll-on/roll-off", "roro",
    "reefer", "refrigerated cargo",
    # Work vessels
    "tug", "tugboat", "towing vessel",
    "offshore support vessel", "osv", "supply vessel", "platform supply vessel", "psv",
    "anchor handling tug", "ahts",
    "dredger", "dredging vessel",
    "pilot boat", "pilot vessel",
    "research vessel", "survey vessel",
    "icebreaker",
    # Fishing vessels
    "fishing vessel", "fishing boat", "trawler", "seiner",
    # Yachts
    "yacht", "motor yacht", "sailing yacht",
    "sailboat", "sailing vessel",
    # Barges
    "barge", "tank barge", "deck barge",
    # Other vessel types
    "workboat", "utility vessel",
    "high-speed craft", "hsc",
    "hovercraft",
    "submarine",
    "warship", "naval vessel",
]

# Construction materials that must not be treated as vessel types
VESSEL_MATERIALS = [
    "aluminum", "aluminium", "aluminum construction",
    "steel", "steel construction", "steel hull",
    "fiberglass", "fibreglass", "frp", "grp",
    "wooden", "wood", "wood construction",
    "composite", "composite construction",
    "concrete", "ferro-cement", "ferrocement",
    "iron", "iron hull",
    "plastic",
]


def is_valid_vessel_type(value: str) -> bool:
    """Return whether value describes a vessel type rather than a material."""
    if not value or not isinstance(value, str):
        return False
    
    value_lower = value.lower().strip()
    
    # Reject construction materials.
    for material in VESSEL_MATERIALS:
        if material in value_lower:
            logger.info(f"vessel_type值 '{value}' 被识别为船舶材料，将设为null")
            return False
    
    # Accept known vessel type keywords.
    for vessel_type in VALID_VESSEL_TYPES:
        if vessel_type in value_lower:
            return True
    
    # Apply additional checks to unrecognized values.
    # Short values need a type keyword such as ship, vessel, or boat.
    type_keywords = ["ship", "vessel", "boat", "craft", "carrier", "tanker", "tug", "ferry", "barge", "yacht"]
    has_type_keyword = any(kw in value_lower for kw in type_keywords)
    
    if not has_type_keyword and len(value_lower) < 20:
        # Reject short descriptions without a type keyword.
        logger.warning(f"vessel_type值 '{value}' 可能不是有效的船舶类型")
        return False
    
    return True


def get_metadata_value(field_data: Any) -> Any:
    """Unwrap a field metadata value, leaving plain values unchanged."""
    if isinstance(field_data, dict):
        return field_data.get("value")
    return field_data


def is_empty_metadata(field_data: Any) -> bool:
    value = get_metadata_value(field_data)
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, list) and not value:
        return True
    return False


def normalize_simple_name(name: Any) -> str:
    if not name:
        return ""
    text = re.sub(r"\b(?:tow|barge|vessel|ship)\b", "", str(name), flags=re.I)
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return text


class Qwen7BExtractor:
    """Extract table information with a local Qwen model."""
    
    def __init__(self, model_path: str = None):
        """Initialize the extractor with model_path or DEFAULT_QWEN_MODEL_PATH."""
        self.model_path = model_path or DEFAULT_QWEN_MODEL_PATH
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and Qwen model."""
        if not HAS_TRANSFORMERS:
            logger.error("transformers库未安装，无法加载模型")
            return
        
        if not os.path.exists(self.model_path):
            logger.error(f"模型路径不存在: {self.model_path}")
            return
        
        try:
            logger.info(f"正在加载Qwen7B模型: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Select the inference device.
            if torch.cuda.is_available():
                self.device = "cuda"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device)
            
            logger.info(f"模型加载成功，使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def is_ready(self) -> bool:
        """Return whether the model is ready."""
        return self.model is not None and self.tokenizer is not None
    
    def _generate_response(self, prompt: str, max_new_tokens: int = 2048) -> str:
        """Generate a response for prompt, limited to max_new_tokens."""
        if not self.is_ready():
            logger.error("模型未就绪")
            return ""
        
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的海事事故信息抽取助手。请严格按照要求提取信息，输出JSON格式。注意：保持原始数据格式，不要进行任何转换或规范化。"},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
            
        except Exception as e:
            logger.error(f"模型生成失败: {e}")
            return ""
    
    def detect_vessels(self, table_data: str) -> Tuple[int, List[str]]:
        """Detect vessel names in table_data and return their count and names.
        
        The input is a JSON string containing the table data.
        """
        prompt = f"""请分析以下表格数据，判断涉及多少艘船舶，并列出所有船舶的名称。

表格数据:
{table_data}

请按照以下JSON格式输出（只输出JSON，不要其他内容）:
{{
    "vessel_count": <船舶数量，整数>,
    "vessel_names": [<船舶名称列表>]
}}

【严格判断规则】船舶名称的识别标准：

✅ 真实船舶名称的特征：
1. 通常包含船舶前缀（如S/S、M/V、T/V、F/V等）+ 船名
2. 出现在以下字段中：
   - "Vessel"、"Vessel Name"、"vessel_name"
   - "Vessel, Flag"（通常包含船名和船旗国）
3. 船名通常是有意义的英文单词或专有名词
4. 示例：
   - "S/S Norway" ✅
   - "M/V Pacific Explorer" ✅
   - "Bahamas-registered passenger vessel S/S Norway" ✅

❌ 以下内容不是船舶名称，必须排除：
1. 任何包含"NTSB"的字符串：
   - "NTSB D023" ❌
   - "NTSBD024" ❌
   - "NTSB-XXX" ❌
2. 表格列名或字段名（如"Roll"、"Boiler"等）❌
3. 测试编号或代码（通常在技术表格的"Roll"列中）❌
4. 日期格式的字符串（如"2003-05-25"、"15-Dec-02"）❌
5. 单独的问号"?" ❌
6. 纯数字编号（如"1"、"2"、"3"、"22"、"23"）❌
7. 时间数据（如"0.75"、"1"、"2.5"）❌

【重要提示】：
- 在锅炉测试记录、技术数据表等表格中，"Roll"列的内容通常是测试编号，不是船舶名称
- 真正的船舶名称通常出现在事故报告的基本信息表格中
- 优先从第一个表格（基本信息表）中提取船舶名称
- 如果不确定某个名称是否为真实船舶名称，请保守判断，不要包含

只输出JSON格式，不要任何解释
"""
        
        response = self._generate_response(prompt)
        
        try:
            # Extract the JSON response.
            response = self._extract_json(response)
            result = json.loads(response)
            vessel_count = result.get("vessel_count", 1)
            vessel_names = result.get("vessel_names", [])
            
            # Filter invalid vessel names.
            filtered_names = []
            for name in vessel_names:
                name_str = str(name).strip()
                name_upper = name_str.upper()
                
                # Apply name exclusions in priority order.
                # Reject empty values and question marks.
                if not name_str or name_str == "?" or name_str == "？":
                    continue
                
                # Reject NTSB identifiers.
                if "NTSB" in name_upper:
                    continue
                
                # Reject numeric identifiers, such as boiler numbers.
                if name_str.isdigit():
                    continue
                
                # Reject decimal values, such as elapsed-time measurements.
                try:
                    float(name_str)
                    continue  # A value that parses as a float is numeric data.
                except ValueError:
                    pass  # Continue checking nonnumeric values.
                
                # Reject date formats.
                if "-" in name_str and any(char.isdigit() for char in name_str):
                    # Check date patterns such as "15-Dec-02" and "2003-05-25".
                    date_patterns = [
                        "%d-%b-%y", "%d-%B-%y",  # 15-Dec-02
                        "%Y-%m-%d", "%d-%m-%Y",  # 2003-05-25
                        "%d-%m-%y", "%y-%m-%d"   # 25-05-03
                    ]
                    is_date = False
                    for pattern in date_patterns:
                        try:
                            from datetime import datetime
                            datetime.strptime(name_str, pattern)
                            is_date = True
                            break
                        except:
                            pass
                    if is_date:
                        continue
                
                # Reject names shorter than three characters unless they have a vessel prefix.
                if len(name_str) < 3:
                    # Check common vessel prefixes.
                    vessel_prefixes = ["S/S", "M/V", "T/V", "F/V", "MT", "SS", "MV"]
                    has_prefix = any(prefix in name_upper for prefix in vessel_prefixes)
                    if not has_prefix:
                        continue
                
                # Reject column labels and technical terms.
                excluded_terms = [
                    "ROLL", "BOILER", "AVERAGE", "TOTAL", "SUM",
                    "RAMP-UP", "COOL-DOWN", "TIME", "HRS", "DATE"
                ]
                if name_upper in excluded_terms:
                    continue
                
                # Accept names with a vessel prefix.
                vessel_prefixes = ["S/S", "M/V", "T/V", "F/V", "MT", "SS", "MV"]
                has_prefix = any(prefix in name_upper for prefix in vessel_prefixes)
                
                # Check candidates without a prefix or VESSEL/SHIP keyword.
                has_vessel_keyword = "VESSEL" in name_upper or "SHIP" in name_upper
                
                # Very short candidates without these cues may not be vessel names.
                if not has_prefix and not has_vessel_keyword and len(name_str) < 5:
                    continue
                
                filtered_names.append(name)
            
            # Handle the case where filtering removes all candidates.
            if not filtered_names and vessel_names:
                logger.warning(f"船舶名称过滤后为空，原始名称: {vessel_names}")
                return vessel_count, vessel_names
            
            return vessel_count, filtered_names if filtered_names else vessel_names
        except:
            logger.warning("船舶检测解析失败，默认为单船")
            return 1, []
    
    def extract_general_fields(self, table_data: str) -> Dict:
        """Extract accident-level fields from the table_data JSON string.
        
        Personnel, casualties, and property damage are extracted per vessel.
        """
        prompt = f"""请从以下表格数据中抽取海事事故的通用信息。

表格数据:
{table_data}

请按照以下JSON格式抽取信息（只输出JSON，不要其他内容）:
{{
    "accident_no": <事故编号，保持原格式，字符串或null>,
    "accident_time": <事故日期和时间，保持原格式不转换，字符串或null>,
    "accident_location": {{
        "location": <事故地点，保持原格式，字符串或null>,
        "latitude": <纬度，保持原格式，字符串或null>,
        "longitude": <经度，保持原格式，字符串或null>
    }},
    "accident_type": <事故类型，保持原格式，字符串或null>,
    "weather_conditions": <天气条件，保持原格式，字符串或null>,
    "waterway_information": <航道/水域信息，从表格Waterway information/Waterway characteristics字段抽取，保持原格式，字符串或null>,
    "visibility": <能见度，保持原格式，字符串或null>,
    "pollution": <污染情况，保持原格式，字符串或null>,
    "economic_loss": <经济损失，保持原格式，字符串或null>,
    "Ship_loss": <船舶损失情况，保持原格式，字符串或null>
}}

重要注意事项：
1. 如果某个字段在表格中找不到对应信息，设为null
2. 【重要】保持原始数据格式，不要进行任何转换或规范化（如时间格式、数字格式等）
3. 【关键】accident_time字段必须包含完整的日期和时间信息：
   - 如果表格中有独立的"Date"和"Time"字段，需要将它们合并到一起
   - 格式示例："May 25, 2003, 0637 eastern daylight time"
   - 如果只有日期或只有时间，就填写有的那部分
4. 只输出JSON格式，不要任何解释
"""
        
        response = self._generate_response(prompt)
        
        try:
            response = self._extract_json(response)
            return json.loads(response)
        except:
            logger.warning("通用字段抽取解析失败")
            return {}
    
    def extract_vessel_info(self, table_data: str, vessel_name: str = None, vessel_index: int = None) -> Dict:
        """Extract particulars, personnel, casualties, and damage for one vessel.
        
        Args:
            table_data: JSON string containing the table data.
            vessel_name: Optional target vessel name.
            vessel_index: Optional vessel index for multi-vessel tables.
        
        Returns:
            A vessel information dictionary.
        """
        vessel_hint = ""
        if vessel_name:
            vessel_hint = f"目标船舶名称: {vessel_name}\n"
        if vessel_index is not None:
            vessel_hint += f"这是第{vessel_index + 1}艘船舶的信息\n"
        
        # Distinguish vessel function from construction material in the prompt.
        prompt = f"""请从以下表格数据中抽取船舶信息。

{vessel_hint}
表格数据:
{table_data}

请按照以下JSON格式抽取船舶信息（只输出JSON，不要其他内容）:
{{
    "Vessel Name": <船舶名称，保持原格式，字符串或null>,
    "ship_length": <船舶长度，保持原格式，字符串或null>,
    "ship_tonnage": <船舶吨位，保持原格式，字符串或null>,
    "vessel_built_year": <建造年份，保持原格式，字符串或null>,
    "flag_state": <船旗国，保持原格式，字符串或null>,
    "mmsi_or_imo": <MMSI或IMO号，保持原格式，字符串或null>,
    "vessel_type": <船舶类型，保持原格式，字符串或null>,
    "owner": <船东，保持原格式，字符串或null>,
    "operator": <运营商/操作者，保持原格式，字符串或null>,
    "crew_complement": <船员数量，保持原格式，字符串或null>,
    "passenger_count": <乘客数量，保持原格式，字符串或null>,
    "casualties": <伤亡情况，包括死亡和受伤信息，保持原格式，字符串或null>,
    "property_damage": <财产损失，保持原格式，字符串或null>
}}

【重要】字段映射规则（v3.0.9）：

1. 船舶名称映射：
   - "Vessel"、"vessel"、"Vessel Name"、"Ship Name" → 提取到 "Vessel Name"
   - 示例: "Vessel": "S/S Norway" → "Vessel Name": "S/S Norway"

2. 船旗国(flag_state)提取：
   - 从"Flag"、"flag_state"、"Flag State"字段直接提取
   - 【关键】从船舶描述中识别，如"Bahamas-registered passenger vessel" → flag_state: "Bahamas"
   - 从"_original_text"字段中查找类似"XXX-registered"的模式
   - 常见船旗国: Bahamas, Panama, Liberia, Marshall Islands, Hong Kong, Singapore等

3. 船东和运营商映射：
   - "Owner"、"owner" → 提取到 "owner"
   - "Operator"、"operator" → 提取到 "operator"
   - "Owner/Operator"字段 → 如果是同一公司，owner和operator都填相同值

4. 船员数量映射：
   - "Complement"、"Crew Complement"、"Crew"字段 → 提取船员数量到 "crew_complement"
   - 示例: "911 crew;2,135 passengers" → crew_complement: "911 crew"

5. 乘客数量映射：
   - "Complement"字段中的乘客信息 → 提取到 "passenger_count"
   - "Passengers"字段 → 提取到 "passenger_count"
   - 示例: "911 crew;2,135 passengers" → passenger_count: "2,135 passengers"

6. 伤亡信息映射：
   - "Injuries"、"Fatalities"、"Casualties"、"Fatalities/Injuries" → 全部提取到 "casualties"
   - 示例: "8 fatalities; 10 serious injuries; 7 minor injuries" → casualties: "8 fatalities; 10 serious injuries; 7 minor injuries"

7. 【v3.0.9关键】vessel_type（船舶类型）的正确识别：
   ✅ vessel_type 是船舶的功能用途类型，有效的船舶类型包括：
      - Passenger ship（客船）, Passenger vessel, Passenger ferry
      - Cargo ship（货船）, General cargo
      - Container ship（集装箱船）
      - Bulk carrier（散货船）
      - Tanker（油轮）, Oil tanker, Chemical tanker
      - Offshore support vessel（海上支援船）
      - Fishing vessel（渔船）
      - Tug（拖船）, Tugboat
      - Ferry（渡轮）
      - Ro-Ro ship（滚装船）
      - Cruise ship（游轮）
   
   ❌ vessel_type 不是船舶的建造材料，以下内容不是船舶类型：
      - "aluminum construction" ❌ (这是材料，不是类型)
      - "steel construction" ❌ (这是材料)
      - "fiberglass" ❌ (这是材料)
      - "wooden" ❌ (这是材料)
   
   如果只找到材料描述而找不到船舶类型，vessel_type应设为null

8. 【重要】如果某个字段在表格中找不到对应信息，设为null
9. 【重要】保持原始数据格式，不要进行任何转换或规范化
10. 只输出JSON格式，不要任何解释
"""
        
        response = self._generate_response(prompt)
        
        try:
            response = self._extract_json(response)
            return json.loads(response)
        except:
            logger.warning("船舶信息抽取解析失败")
            return {}
    
    def extract_multi_vessel_info(self, table_data: str, vessel_count: int, vessel_names: List[str]) -> List[Dict]:
        """Extract a list of vessel records from the table_data JSON string.
        
        Each vessel has its own personnel, casualty, and property damage fields.
        vessel_count and vessel_names identify the expected vessels.
        """
        vessels_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(vessel_names)]) if vessel_names else ""
        
        prompt = f"""请从以下表格数据中分别抽取{vessel_count}艘船舶的信息。

已识别的船舶:
{vessels_list}

表格数据:
{table_data}

请按照以下JSON格式输出所有船舶的信息（只输出JSON数组，不要其他内容）:
[
    {{
        "Vessel Name": <第1艘船的名称，保持原格式>,
        "ship_length": <船舶长度，保持原格式，或null>,
        "ship_tonnage": <船舶吨位，保持原格式，或null>,
        "vessel_built_year": <建造年份，保持原格式，或null>,
        "flag_state": <船旗国，保持原格式>,
        "mmsi_or_imo": <MMSI或IMO号，保持原格式，或null>,
        "vessel_type": <船舶类型，保持原格式，或null>,
        
        "owner": <船东，保持原格式>,
        "operator": <运营商，保持原格式>,
        "crew_complement": <该船的船员数量，保持原格式，或null>,
        "passenger_count": <该船的乘客数量，保持原格式，或null>,
        "casualties": <该船的伤亡情况，保持原格式，或null>,
        "property_damage": <该船的财产损失，保持原格式，或null>
    }},
    ... // 其他船舶
]

【重要】字段映射规则（v3.0.9）：

1. 船舶名称映射：
   - "Vessel"、"vessel"、"Vessel Name"、"Ship Name" → 提取到 "Vessel Name"
   - 示例: "Vessel": "S/S Norway" → "Vessel Name": "S/S Norway"

2. 船旗国(flag_state)提取：
   - 从"Flag"、"flag_state"、"Flag State"字段直接提取
   - 【关键】从船舶描述中识别，如"Bahamas-registered passenger vessel" → flag_state: "Bahamas"
   - 从"_original_text"字段中查找类似"XXX-registered"的模式
   - 常见船旗国: Bahamas, Panama, Liberia, Marshall Islands, Hong Kong, Singapore等

3. 船东和运营商映射：
   - "Owner"、"owner" → 提取到 "owner"
   - "Operator"、"operator" → 提取到 "operator"
   - "Owner/Operator"字段 → 如果是同一公司，owner和operator都填相同值
   - "Owner, Operator"包含复杂信息如"Owned by X, Operated by Y"需要分别提取

4. 船员数量映射：
   - "Complement"、"Crew Complement"、"Crew"字段 → 提取船员数量到 "crew_complement"
   - 示例: "911 crew;2,135 passengers" → crew_complement: "911 crew"

5. 乘客数量映射：
   - "Complement"字段中的乘客信息 → 提取到 "passenger_count"
   - "Passengers"字段 → 提取到 "passenger_count"
   - 示例: "911 crew;2,135 passengers" → passenger_count: "2,135 passengers"

6. 伤亡信息映射：
   - "Injuries"、"Fatalities"、"Casualties"、"Fatalities/Injuries" → 全部提取到 "casualties"
   - 示例: "8 fatalities; 10 serious injuries; 7 minor injuries" → casualties: "8 fatalities; 10 serious injuries; 7 minor injuries"

7. 【v3.0.9关键】vessel_type（船舶类型）的正确识别：
   ✅ vessel_type 是船舶的功能用途类型，有效的船舶类型包括：
      - Passenger ship, Cargo ship, Container ship, Bulk carrier, Tanker
      - Offshore support vessel, Fishing vessel, Tug, Ferry, Cruise ship等
   
   ❌ vessel_type 不是船舶的建造材料：
      - "aluminum construction" ❌ (这是材料，不是类型)
      - "steel construction" ❌ (这是材料)
      - "fiberglass" ❌
   
   如果只找到材料描述而找不到船舶类型，vessel_type应设为null

8. 数组中的每个元素对应一艘船舶
9. 【重要】保持原始数据格式，不要进行任何转换或规范化
10. 只输出JSON数组，不要任何解释
"""
        
        response = self._generate_response(prompt)
        
        try:
            response = self._extract_json(response)
            result = json.loads(response)
            if isinstance(result, list):
                return result
            return [result]
        except:
            logger.warning("多船舶信息抽取解析失败")
            return []
    
    def _extract_json(self, text: str) -> str:
        """Extract a JSON object or array string from model output."""
        # Remove Markdown code fences.
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Locate the JSON boundaries.
        start_idx = -1
        end_idx = -1
        
        # Find the first opening brace or bracket.
        for i, char in enumerate(text):
            if char in ['{', '[']:
                start_idx = i
                break
        
        if start_idx == -1:
            return text
        
        # Determine whether the JSON is an object or an array.
        is_array = text[start_idx] == '['
        
        # Find the matching closing delimiter.
        bracket_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == ('{' if not is_array else '['):
                bracket_count += 1
            elif text[i] == ('}' if not is_array else ']'):
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            return text[start_idx:]
        
        return text[start_idx:end_idx + 1]


class MaritimeAccidentExtractor:
    """Populate the shared accident schema from table extraction results."""
    
    def __init__(self, model_path: str = None):
        """Initialize the table extractor with the Qwen model path."""
        self.qwen_extractor = Qwen7BExtractor(model_path)
    
    def _get_table_data_string(self, table_result: Dict) -> str:
        """Serialize parsed table data for extraction.
        
        Irregular tables include original_node.text so embedded details, such as
        flag state, remain available to the model.
        """
        # Require a dictionary table result.
        if not isinstance(table_result, dict):
            logger.warning(f"table_result不是字典类型，而是: {type(table_result)}")
            return json.dumps(table_result, ensure_ascii=False, indent=2)
        
        table_type = table_result.get("table_type", "")
        parsed_data = table_result.get("parsed_data", {})
        
        # Require a dictionary of parsed data.
        if not isinstance(parsed_data, dict):
            logger.warning(f"parsed_data不是字典类型，而是: {type(parsed_data)}")
            return json.dumps(parsed_data, ensure_ascii=False, indent=2)
        
        if table_type == "html_table":
            # Use structured_data for HTML tables.
            structured_data = parsed_data.get("structured_data", [])
            if structured_data:
                return json.dumps(structured_data, ensure_ascii=False, indent=2)
            # Fall back to raw rows when structured_data is unavailable.
            rows = parsed_data.get("rows", [])
            return json.dumps(rows, ensure_ascii=False, indent=2)
        
        elif table_type == "text_table":
            # Exclude parser metadata from text table data.
            clean_data = {k: v for k, v in parsed_data.items() 
                         if k not in ["parse_method", "raw_model_output"]}
            return json.dumps(clean_data, ensure_ascii=False, indent=2)
        
        elif table_type == "irregular_table":
            # Handle irregular tables like text tables.
            # Include original text for embedded flag state and vessel type details.
            clean_data = {k: v for k, v in parsed_data.items() 
                         if k not in ["parse_method", "raw_model_output"]}
            
            # Recover the source text when available.
            original_node = table_result.get("original_node", {})
            if isinstance(original_node, dict):
                original_text = original_node.get("text", "")
                if original_text:
                    clean_data["_original_text"] = original_text
            
            return json.dumps(clean_data, ensure_ascii=False, indent=2)
        
        # Remove parser metadata for other table types.
        clean_data = {k: v for k, v in parsed_data.items() 
                     if k not in ["parse_method", "raw_model_output"]}
        return json.dumps(clean_data, ensure_ascii=False, indent=2)
    
    def _get_page_idx(self, table_result: Dict) -> List[int]:
        """Return the source page index for a table."""
        # Require a dictionary table result.
        if not isinstance(table_result, dict):
            logger.warning(f"table_result不是字典类型，而是: {type(table_result)}")
            return []
        
        original_node = table_result.get("original_node", {})
        
        # Require a dictionary source node.
        if not isinstance(original_node, dict):
            logger.warning(f"original_node不是字典类型，而是: {type(original_node)}")
            return []
        
        page_idx = original_node.get("page_idx")
        if page_idx is not None:
            return [page_idx] if isinstance(page_idx, int) else page_idx
        return []
    
    def _fill_general_fields(self, result: Dict, extracted: Dict, page_idx: List[int]):
        """Populate accident-level fields using create_field_metadata.
        
        Personnel, casualties, and property damage are handled in vessel records.
        """
        # Require a dictionary extraction result.
        if not isinstance(extracted, dict):
            logger.warning(f"extracted不是字典类型，而是: {type(extracted)}")
            return
        
        confidence = 0.98  # Default confidence
        
        # Accident-level field mapping
        simple_fields = [
            "accident_no", "accident_time", "accident_type",
            "weather_conditions", "waterway_information", 
            "pollution", "economic_loss", "ship_loss"
        ]
        
        for field in simple_fields:
            value = extracted.get(field)
            if value:
                result[field] = create_field_metadata(
                    value=value,
                    confidence=confidence,
                    page_idx=page_idx,
                    source="table"
                )
        
        # Populate the nested accident location structure.
        location_data = extracted.get("accident_location", {})
        if isinstance(location_data, dict):
            location = location_data.get("location")
            latitude = location_data.get("latitude")
            longitude = location_data.get("longitude")
            if location or latitude or longitude:
                result["accident_location"] = create_location_with_coordinates(
                    location=location,
                    latitude=latitude,
                    longitude=longitude,
                    confidence=confidence,
                    page_idx=page_idx,
                    source="table"
                )

    @staticmethod
    def _normalize_table_text(text: Any) -> str:
        """Normalize table cell whitespace without changing its meaning."""
        if text is None:
            return ""
        text = html.unescape(str(text))
        text = text.replace("\xa0", " ")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_field_label(label: Any) -> str:
        label = MaritimeAccidentExtractor._normalize_table_text(label).lower()
        label = re.sub(r"[^a-z0-9]+", " ", label)
        return re.sub(r"\s+", " ", label).strip()

    @staticmethod
    def _is_waterway_label(label: str) -> bool:
        return label in {
            "waterway",
            "waterway information",
            "waterway characteristics",
            "waterway characteristic",
            "waterway info",
            "waterway conditions",
            "route information",
        }

    @staticmethod
    def _extract_leading_weather_tail(value: str) -> Tuple[str, str]:
        """Return the weather fragment and the remaining waterway description."""
        value = MaritimeAccidentExtractor._normalize_table_text(value)
        if not value:
            return "", ""

        temp_match = re.match(
            r"^((?:air\s+|water\s+)?temperature\s+(?:about\s+)?\d+(?:\.\d+)?\s*(?:°?\s*[Ff])?)\s+(.*)$",
            value,
        )
        if temp_match:
            return temp_match.group(1).strip(), temp_match.group(2).strip()

        return "", value

    @staticmethod
    def _strip_weather_prefix_from_waterway(value: str) -> str:
        """Remove weather text displaced to the start of a waterway value.
        
        For example, "temperature 51F Carrollton Bend..." becomes
        "Carrollton Bend...".
        """
        _, value = MaritimeAccidentExtractor._extract_leading_weather_tail(value)
        if not value:
            return value

        # The end of a Weather cell may be shifted into the next Waterway row.
        value = re.sub(
            r"^(?:air\s+|water\s+)?temperature\s+about\s+\d+(?:\.\d+)?\s*(?:°?\s*[Ff])?\s+",
            "",
            value,
        ).strip()
        value = re.sub(
            r"^(?:air\s+|water\s+)?temperature\s+\d+(?:\.\d+)?\s*(?:°?\s*[Ff])?\s+",
            "",
            value,
        ).strip()

        # Longer weather fragments may end just before a named waterway.
        waterway_start = re.search(
            r"\b(?:The\s+)?[A-Z][A-Za-z.'-]*(?:\s+[A-Z0-9][A-Za-z0-9.'-]*){0,8}\s+"
            r"(?:River|Sea|Harbor|Harbour|Bay|Channel|Waterway|Gulf|Lake|Port|Marina|Bend|Point)\b",
            value,
        )
        if waterway_start and waterway_start.start() > 0:
            prefix = value[:waterway_start.start()].lower()
            weather_tail_markers = [
                "temperature", "wind", "winds", "knots", "mph", "visibility",
                "sunrise", "sunset", "twilight", "seas", "current", "ended"
            ]
            if any(marker in prefix for marker in weather_tail_markers):
                return value[waterway_start.start():].strip()

        return value

    @staticmethod
    def _rows_from_html_table(html_text: str) -> List[List[str]]:
        if not html_text:
            return []
        rows = []
        for tr_match in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", html_text, flags=re.I | re.S):
            row_html = tr_match.group(1)
            cells = []
            for cell_match in re.finditer(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.I | re.S):
                cell_text = re.sub(r"<[^>]+>", " ", cell_match.group(1))
                cells.append(MaritimeAccidentExtractor._normalize_table_text(cell_text))
            if cells:
                rows.append(cells)
        return rows

    def _collect_table_rows(self, table_result: Dict) -> List[List[str]]:
        rows: List[List[str]] = []
        parsed_data = table_result.get("parsed_data", {})
        if isinstance(parsed_data, dict):
            parsed_rows = parsed_data.get("rows", [])
            if isinstance(parsed_rows, list):
                rows.extend(row for row in parsed_rows if isinstance(row, list))

        original_node = table_result.get("original_node", {})
        if isinstance(original_node, dict):
            html_text = (
                original_node.get("table_body", "")
                or original_node.get("html", "")
                or original_node.get("table_html", "")
                or original_node.get("text", "")
            )
            rows.extend(self._rows_from_html_table(html_text))

        deduped_rows: List[List[str]] = []
        seen = set()
        for row in rows:
            normalized_row = tuple(self._normalize_table_text(cell) for cell in row)
            if normalized_row in seen:
                continue
            seen.add(normalized_row)
            deduped_rows.append(list(normalized_row))

        return deduped_rows

    def _extract_waterway_information_from_table(self, table_result: Dict) -> Optional[str]:
        """Extract Waterway information directly from table fields.
        
        Handle labels split across Waterway / information rows and weather text
        displaced to the beginning of the value.
        """
        rows = self._collect_table_rows(table_result)
        if not rows:
            return None

        collected: List[str] = []
        collecting = False

        for row in rows:
            if len(row) < 2:
                continue

            label = self._normalize_field_label(row[0])
            value = self._normalize_table_text(" ".join(str(cell) for cell in row[1:]))

            if label == "waterway":
                cleaned = self._strip_weather_prefix_from_waterway(value)
                if cleaned:
                    collected.append(cleaned)
                collecting = True
                continue

            if collecting and label in {"information", "characteristics", "characteristic", "info"}:
                if value:
                    collected.append(value)
                continue

            if self._is_waterway_label(label):
                cleaned = self._strip_weather_prefix_from_waterway(value)
                if cleaned:
                    collected.append(cleaned)
                collecting = False
                continue

            if collecting and label:
                collecting = False

        result = self._normalize_table_text(" ".join(collected))
        return result or None

    def _extract_weather_conditions_from_table(self, table_result: Dict) -> Optional[str]:
        """Extract Weather and recover temperature text displaced into the Waterway row."""
        rows = self._collect_table_rows(table_result)
        if not rows:
            return None

        weather_parts: List[str] = []
        for idx, row in enumerate(rows):
            if len(row) < 2:
                continue

            label = self._normalize_field_label(row[0])
            value = self._normalize_table_text(" ".join(str(cell) for cell in row[1:]))
            if label not in {"weather", "damage weather", "environmental damage weather"}:
                continue

            if value:
                weather_parts.append(value)

            if idx + 1 < len(rows) and rows[idx + 1]:
                next_label = self._normalize_field_label(rows[idx + 1][0])
                next_value = self._normalize_table_text(" ".join(str(cell) for cell in rows[idx + 1][1:]))
                if next_label == "waterway":
                    weather_tail, _ = self._extract_leading_weather_tail(next_value)
                    if weather_tail:
                        weather_parts.append(weather_tail)

        result = self._normalize_table_text(" ".join(weather_parts))
        return result or None

    def _extract_involved_vessels_from_table(self, table_result: Dict) -> List[str]:
        """Read Vessel name(s) directly from rows to recover the full involved-vessel list."""
        rows = self._collect_table_rows(table_result)
        if not rows:
            return []

        candidates: List[str] = []
        for idx, row in enumerate(rows):
            if len(row) < 2:
                continue
            label = self._normalize_field_label(row[0])
            if label not in {"vessel name", "vessel names", "vessel", "vessels"} and "vessel names" not in label:
                continue

            if label in {"vessel", "vessels"} and len(row) > 2:
                for cell in row[1:]:
                    cleaned = self._normalize_table_text(cell)
                    if cleaned and normalize_simple_name(cleaned):
                        candidates.append(cleaned)
                continue

            value = self._normalize_table_text(" ".join(str(cell) for cell in row[1:]))
            if "vessel names" in label and not re.search(r"\b(?:tow|barge|vessel|ship|M/V|S/S|T/V|F/V)\b", value, flags=re.I):
                if idx + 1 < len(rows):
                    next_label = self._normalize_field_label(rows[idx + 1][0])
                    next_value = self._normalize_table_text(" ".join(str(cell) for cell in rows[idx + 1][1:]))
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
                cleaned = self._normalize_table_text(part)
                cleaned = re.sub(r"^(?:and\s+)?barges?\s+", "", cleaned, flags=re.I)
                cleaned = re.sub(r"\b(?:tow|tows)\b$", "", cleaned, flags=re.I).strip(" ,")
                if cleaned and normalize_simple_name(cleaned):
                    candidates.append(cleaned)

        return self._dedupe_names(candidates)

    @staticmethod
    def _is_barge_or_auxiliary_name(name: Any) -> bool:
        text = MaritimeAccidentExtractor._normalize_table_text(name)
        if not text:
            return True
        if re.fullmatch(r"\d{4,}", text):
            return True
        return bool(re.search(r"\bbarges?\b|^AEP\s*\d+|^APEX\s*\d+|^ING\s*\d+|^IB\s*\d+", text, flags=re.I))

    def _select_core_vessel_names(self, names: List[Any]) -> List[str]:
        """Select vessels for vessel_1 through vessel_3, excluding generic or numbered barges."""
        core = [
            name for name in self._dedupe_names(names)
            if not self._is_barge_or_auxiliary_name(name)
        ]
        return core[:3]

    @staticmethod
    def _dedupe_names(names: List[Any]) -> List[str]:
        result: List[str] = []
        seen = set()
        for name in names:
            text = MaritimeAccidentExtractor._normalize_table_text(name)
            key = normalize_simple_name(text)
            if not text or not key or key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result

    def _set_field(self, result: Dict, field: str, value: Any, page_idx: List[int], confidence: float = 0.99):
        if value is None:
            return
        if isinstance(value, str):
            value = self._normalize_table_text(value)
            if not value:
                return
        result[field] = create_field_metadata(
            value=value,
            confidence=confidence,
            page_idx=page_idx,
            source="table"
        )

    @staticmethod
    def _clean_money_text(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        value = re.sub(r"^\s*(?:None|No(?:ne)? reported)\s+(?=\$)", "", value, flags=re.I).strip()
        return value

    @staticmethod
    def _split_accident_type_and_no(accident_type: Any, accident_no: Any) -> Tuple[Any, Any]:
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

    def _postprocess_general_result(self, result: Dict, table_result: Dict, page_idx: List[int]):
        """Validate and correct accident-level fields against the source table."""
        weather = self._extract_weather_conditions_from_table(table_result)
        if weather:
            self._set_field(result, "weather_conditions", weather, page_idx)

        accident_type_value, accident_no_value = self._split_accident_type_and_no(
            get_metadata_value(result.get("accident_type")),
            get_metadata_value(result.get("accident_no"))
        )
        if accident_type_value:
            self._set_field(result, "accident_type", accident_type_value, page_idx)
        if accident_no_value:
            self._set_field(result, "accident_no", accident_no_value, page_idx)

        for field in ["economic_loss", "ship_loss"]:
            cleaned = self._clean_money_text(get_metadata_value(result.get(field)))
            if cleaned:
                self._set_field(result, field, cleaned, page_idx, confidence=0.98)

    @staticmethod
    def _clean_vessel_field_value(field: str, value: Any) -> Any:
        if not isinstance(value, str):
            return value

        value = MaritimeAccidentExtractor._normalize_table_text(value)
        if field == "crew_complement":
            # Separate engine model text from crew counts, such as "GM L12-645-E2 4".
            if re.search(r"\b(?:GM|Caterpillar|Cummins|Detroit|EMD)\b", value, flags=re.I):
                numbers = re.findall(r"\b\d{1,3}\b", value)
                return numbers[-1] if numbers else None
        elif field == "flag_state":
            if re.search(r"\bUnited States\b", value, flags=re.I):
                return "United States"
        elif field == "property_damage":
            return MaritimeAccidentExtractor._clean_money_text(value)

        return value
    
    def _fill_vessel_info(self, result: Dict, vessel_data: Dict, vessel_number: int, page_idx: List[int]):
        """Populate a vessel record using the shared schema helpers.
        
        Match field names case-insensitively, accept common aliases, and validate
        vessel_type to exclude construction materials.
        """
        # Require dictionary vessel data.
        if not isinstance(vessel_data, dict):
            logger.warning(f"vessel_data不是字典类型，而是: {type(vessel_data)}")
            return
        
        # Create the vessel record using the shared schema.
        vessel_info = create_vessel_info()
        confidence = 0.98
        
        # Vessel field mapping
        vessel_fields = [
            "Vessel Name", "ship_length", "ship_tonnage", "vessel_built_year",
            "flag_state", "mmsi_or_imo", "vessel_type", 
            "owner", "operator",
            # Personnel, casualties, and damage belong to each vessel.
            "crew_complement", "passenger_count", "casualties", "property_damage"
        ]
        
        # Build a case-insensitive field lookup.
        vessel_data_lower = {k.lower(): v for k, v in vessel_data.items()}
        
        for field in vessel_fields:
            # Try the exact field name first.
            value = vessel_data.get(field)
            
            # Then try a case-insensitive match.
            if value is None:
                field_lower = field.lower()
                value = vessel_data_lower.get(field_lower)
                
                # Finally, check common field aliases.
                if value is None:
                    # Field aliases
                    field_variants = {
                        "vessel name": ["vessel_name", "vesselname", "vessel"],
                        "ship_length": ["length", "shiplength"],
                        "ship_tonnage": ["tonnage", "shiptonnage", "gross_tonnage", "grosstonnage"],
                        "vessel_built_year": ["built_year", "builtyear", "year_built", "yearbuilt"],
                        "flag_state": ["flag", "flagstate", "registry"],
                        "mmsi_or_imo": ["mmsi", "imo", "imo_number", "imonumber"],
                        "vessel_type": ["type", "vesseltype", "ship_type", "shiptype"],
                        "crew_complement": ["crew", "crewcomplement", "complement"],
                        "passenger_count": ["passengers", "passengercount", "passenger"],
                    }
                    
                    variants = field_variants.get(field_lower, [])
                    for variant in variants:
                        value = vessel_data_lower.get(variant)
                        if value is not None:
                            break
            
            # Validate the vessel type.
            if field == "vessel_type" and value:
                if not is_valid_vessel_type(value):
                    logger.info(f"vessel_type值 '{value}' 不是有效的船舶类型，设为null")
                    value = None

            value = self._clean_vessel_field_value(field, value)
            
            if value:
                vessel_info[field] = create_field_metadata(
                    value=value,
                    confidence=confidence,
                    page_idx=page_idx,
                    source="table"
                )
        
        # Add the record with the shared schema helper.
        add_vessel_info(result, vessel_number, vessel_data=vessel_info)
    
    def extract_from_table(self, table_result: Dict) -> Dict:
        """Extract an accident record from one parsed table.
        
        Use detected names when detailed extraction omits them. For a single
        vessel, fill missing property_damage from economic_loss when available.
        """
        # Create an empty accident record.
        result = create_empty_accident_structure()
        
        if not self.qwen_extractor.is_ready():
            logger.error("Qwen7B模型未就绪，无法进行抽取")
            return result
        
        # Require a dictionary table result.
        if not isinstance(table_result, dict):
            logger.error(f"table_result不是字典类型，而是: {type(table_result)}")
            return result
        
        # Serialize the table data.
        table_data_str = self._get_table_data_string(table_result)
        page_idx = self._get_page_idx(table_result)
        
        logger.info(f"开始抽取表格数据...")
        
        # Detect vessels.
        logger.info("步骤1: 检测船舶数量...")
        vessel_count, vessel_names = self.qwen_extractor.detect_vessels(table_data_str)
        table_vessel_names = self._extract_involved_vessels_from_table(table_result)
        core_vessel_names = self._select_core_vessel_names(table_vessel_names)
        if core_vessel_names:
            vessel_names = core_vessel_names
            vessel_count = len(vessel_names)
        logger.info(f"检测到 {vessel_count} 艘船舶: {vessel_names}")
        
        # Extract accident-level fields.
        logger.info("步骤2: 抽取通用字段...")
        general_fields = self.qwen_extractor.extract_general_fields(table_data_str)
        self._fill_general_fields(result, general_fields, page_idx)
        
        waterway_information = self._extract_waterway_information_from_table(table_result)
        if waterway_information:
            result["waterway_information"] = create_field_metadata(
                value=waterway_information,
                confidence=0.99,
                page_idx=page_idx,
                source="table"
            )

        if table_vessel_names:
            result["involved_vessels"] = create_field_metadata(
                value=table_vessel_names,
                confidence=0.99,
                page_idx=page_idx,
                source="table"
            )

        self._postprocess_general_result(result, table_result, page_idx)
        
        # Extract vessel particulars, personnel, casualties, and damage.
        logger.info("步骤3: 抽取船舶信息...")
        
        if vessel_count <= 1:
            # Single-vessel extraction
            vessel_info = self.qwen_extractor.extract_vessel_info(table_data_str)
            
            # Accept a list when the model returns multiple vessel records.
            if isinstance(vessel_info, list) and len(vessel_info) > 1:
                # Process a multi-vessel response as separate vessel records.
                logger.warning(f"extract_vessel_info返回了列表类型（包含{len(vessel_info)}个元素），切换到多船处理逻辑")
                vessels_info = vessel_info
                involved_vessel_names = []
                
                for i, vessel_data in enumerate(vessels_info):
                    if isinstance(vessel_data, dict):
                        self._fill_vessel_info(result, vessel_data, i + 1, page_idx)
                        v_name = vessel_data.get("Vessel Name")
                        if v_name:
                            involved_vessel_names.append(v_name)
                
                if involved_vessel_names:
                    result["involved_vessels"] = create_field_metadata(
                        value=involved_vessel_names,
                        confidence=0.98,
                        page_idx=page_idx,
                        source="table"
                    )
            else:
                # Process a single-vessel response.
                # Unwrap a one-element list.
                if isinstance(vessel_info, list):
                    vessel_info = vessel_info[0] if vessel_info else {}
                
                if vessel_info and isinstance(vessel_info, dict):
                    self._fill_vessel_info(result, vessel_info, 1, page_idx)
                    
                    # Use the detected name when detailed extraction omits it.
                    vessel_name = vessel_info.get("Vessel Name")
                    
                    if not vessel_name and vessel_names and len(vessel_names) > 0:
                        vessel_name = vessel_names[0]
                        logger.info(f"extract_vessel_info未能提取vessel_name，使用detect_vessels的结果作为fallback: {vessel_name}")
                        
                        # Update the vessel record's name.
                        if "vessel_1_info" in result and "Vessel Name" in result["vessel_1_info"]:
                            result["vessel_1_info"]["Vessel Name"] = create_field_metadata(
                                value=vessel_name,
                                confidence=0.98,
                                page_idx=page_idx,
                                source="table"
                            )
                    
                    # Update the involved-vessel summary.
                    if vessel_name:
                        result["involved_vessels"] = create_field_metadata(
                            value=[vessel_name],
                            confidence=0.98,
                            page_idx=page_idx,
                            source="table"
                        )
                    
                    # Fill missing property_damage from economic_loss for a single vessel.
                    self._sync_property_damage_for_single_vessel(result, page_idx)
        else:
            # Multi-vessel extraction
            vessels_info = self.qwen_extractor.extract_multi_vessel_info(
                table_data_str, vessel_count, vessel_names
            )
            
            involved_vessel_names = []
            for i, vessel_data in enumerate(vessels_info):
                self._fill_vessel_info(result, vessel_data, i + 1, page_idx)
                vessel_name = vessel_data.get("Vessel Name") if isinstance(vessel_data, dict) else None
                
                # Use the detected vessel name when detailed extraction omits it.
                if not vessel_name and vessel_names and i < len(vessel_names):
                    vessel_name = vessel_names[i]
                    logger.info(f"extract_multi_vessel_info未能提取船舶{i+1}的vessel_name，使用detect_vessels的结果: {vessel_name}")
                    
                    # Update the corresponding vessel record's name.
                    vessel_key = f"vessel_{i+1}_info"
                    if vessel_key in result and "Vessel Name" in result[vessel_key]:
                        result[vessel_key]["Vessel Name"] = create_field_metadata(
                            value=vessel_name,
                            confidence=0.98,
                            page_idx=page_idx,
                            source="table"
                        )
                
                if vessel_name:
                    involved_vessel_names.append(vessel_name)
            
            # Update the involved-vessel summary.
            if involved_vessel_names:
                result["involved_vessels"] = create_field_metadata(
                    value=involved_vessel_names,
                    confidence=0.98,
                    page_idx=page_idx,
                    source="table"
                )
        
        logger.info("表格抽取完成")
        return result
    
    def _sync_property_damage_for_single_vessel(self, result: Dict, page_idx: List[int]):
        """Fill missing single-vessel property_damage from economic_loss.
        
        Args:
            result: Extracted accident record.
            page_idx: Source page index.
        """
        # Require a vessel_1_info record.
        if "vessel_1_info" not in result:
            return
        
        vessel_info = result["vessel_1_info"]
        
        # Check the current property damage value.
        property_damage = vessel_info.get("property_damage", {})
        property_damage_value = None
        if isinstance(property_damage, dict):
            property_damage_value = property_damage.get("value")
        
        # Keep existing property damage.
        if property_damage_value:
            return
        
        # Check whether economic_loss is available.
        economic_loss = result.get("economic_loss", {})
        economic_loss_value = None
        if isinstance(economic_loss, dict):
            economic_loss_value = economic_loss.get("value")
        
        # Copy economic_loss into the empty property damage field.
        if economic_loss_value:
            logger.info(f"单船情况下同步property_damage: 从economic_loss复制值 '{economic_loss_value}'")
            vessel_info["property_damage"] = create_field_metadata(
                value=economic_loss_value,
                confidence=0.98,
                page_idx=page_idx,
                source="table"
            )
    
    def extract_from_file(self, input_file: str) -> Dict:
        """Extract and merge accident information from an input JSON file."""
        logger.info(f"处理文件: {input_file}")
        
        # Read the input file.
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Initialize the shared accident schema.
        result = create_empty_accident_structure()
        
        # The schema already places extraction_metadata first.
        result["extraction_metadata"]["source_files"] = [os.path.basename(input_file)]
        result["extraction_metadata"]["extraction_time"] = datetime.now().isoformat()
        result["extraction_metadata"]["extractor_version"] = "qwen7b-3.0.9"
        
        # Accept supported input container formats.
        tables = []
        
        if isinstance(data, dict):
            # Standard format: {"results": [...]}.
            tables = data.get("results", [])
        elif isinstance(data, list):
            # A list of table results is also supported.
            tables = data
        else:
            logger.warning(f"未知的数据格式: {type(data)}")
            return result
        
        # Require a list of tables.
        if not isinstance(tables, list):
            logger.warning(f"tables不是列表类型，而是: {type(tables)}")
            tables = [tables] if tables else []
        
        if not tables:
            logger.warning(f"文件中没有找到表格: {input_file}")
            return result
        
        logger.info(f"找到 {len(tables)} 个表格")
        
        # Process each table.
        for i, table in enumerate(tables):
            logger.info(f"处理第 {i + 1}/{len(tables)} 个表格...")
            
            # Require each table to be a dictionary.
            if not isinstance(table, dict):
                logger.warning(f"第 {i + 1} 个表格不是字典类型，跳过")
                continue
            
            table_result = self.extract_from_table(table)
            
            # Use the first table's core vessels; later tables fill matching records and extend involved_vessels.
            result = self._merge_results(result, table_result, allow_new_vessels=(i == 0))
        
        # Recheck single-vessel property damage after merging all tables.
        # The relevant loss fields may come from different tables.
        self._final_sync_property_damage(result)
        
        return result
    
    def _final_sync_property_damage(self, result: Dict):
        """Fill missing single-vessel property_damage after all tables are merged."""
        # Require vessel_1_info with no additional vessel records.
        vessel_count = 0
        for key in result.keys():
            if key.startswith("vessel_") and key.endswith("_info"):
                vessel_count += 1
        
        if vessel_count != 1:
            return  # Do not infer per-vessel damage from totals in multi-vessel cases.
        
        if "vessel_1_info" not in result:
            return
        
        vessel_info = result["vessel_1_info"]
        
        # Check the current property damage value.
        property_damage = vessel_info.get("property_damage", {})
        property_damage_value = None
        if isinstance(property_damage, dict):
            property_damage_value = property_damage.get("value")
        
        if property_damage_value:
            return  # Keep existing property damage.
        
        # Check economic_loss.
        economic_loss = result.get("economic_loss", {})
        economic_loss_value = None
        economic_loss_page_idx = []
        if isinstance(economic_loss, dict):
            economic_loss_value = economic_loss.get("value")
            economic_loss_page_idx = economic_loss.get("page_idx", [])
        
        if economic_loss_value:
            logger.info(f"最终同步: 将economic_loss '{economic_loss_value}' 复制到vessel_1_info.property_damage")
            vessel_info["property_damage"] = create_field_metadata(
                value=economic_loss_value,
                confidence=0.98,
                page_idx=economic_loss_page_idx,
                source="table"
            )
    
    def _get_nonempty_vessel_keys(self, data: Dict) -> List[str]:
        keys = []
        for key in data:
            if not (key.startswith("vessel_") and key.endswith("_info")):
                continue
            vessel_name = get_metadata_value(data.get(key, {}).get("Vessel Name"))
            if vessel_name:
                keys.append(key)
        return sorted(keys, key=lambda k: int(k.split("_")[1]))

    def _find_matching_vessel_key(self, base: Dict, vessel_info: Dict) -> Optional[str]:
        new_name = get_metadata_value(vessel_info.get("Vessel Name"))
        if not new_name:
            return None
        new_key = normalize_simple_name(new_name)
        if not new_key:
            return None

        for key in self._get_nonempty_vessel_keys(base):
            base_name = get_metadata_value(base.get(key, {}).get("Vessel Name"))
            base_key = normalize_simple_name(base_name)
            if not base_key:
                continue
            if new_key == base_key or new_key in base_key or base_key in new_key:
                return key
        return None

    def _merge_vessel_fields(self, base_vessel: Dict, new_vessel: Dict):
        for field_key, field_value in new_vessel.items():
            if is_empty_metadata(field_value):
                continue
            # Prefer the main accident table; later tables only fill empty fields.
            if field_key not in base_vessel or is_empty_metadata(base_vessel.get(field_key)):
                base_vessel[field_key] = field_value

    def _merge_involved_vessels(self, base: Dict, value: Dict):
        new_value = get_metadata_value(value)
        base_value = get_metadata_value(base.get("involved_vessels"))
        if not isinstance(new_value, list):
            return
        new_list = self._dedupe_names(new_value)
        base_list = self._dedupe_names(base_value if isinstance(base_value, list) else [])
        if len(new_list) > len(base_list):
            base["involved_vessels"] = value
            if "extraction_metadata" in base:
                base["extraction_metadata"]["total_vessels"] = len(new_list)

    def _merge_results(self, base: Dict, new: Dict, allow_new_vessels: bool = True) -> Dict:
        """Merge new extraction data into base and return the combined record."""
        # Require both inputs to be dictionaries.
        if not isinstance(base, dict):
            logger.warning(f"base不是字典类型，而是: {type(base)}")
            return new if isinstance(new, dict) else {}
        
        if not isinstance(new, dict):
            logger.warning(f"new不是字典类型，而是: {type(new)}")
            return base
        
        for key, value in new.items():
            if key == "extraction_metadata":
                # Merge extraction metadata.
                if isinstance(value, dict):
                    base[key]["total_vessels"] = max(
                        base[key].get("total_vessels", 0),
                        value.get("total_vessels", 0)
                    )
                continue
            
            if key == "involved_vessels" and isinstance(value, dict) and "value" in value:
                self._merge_involved_vessels(base, value)
                continue

            if key.startswith("vessel_") and key.endswith("_info"):
                vessel_idx = int(key.split("_")[1]) if key.split("_")[1].isdigit() else 999
                if vessel_idx > 3:
                    continue

                target_key = self._find_matching_vessel_key(base, value) if isinstance(value, dict) else None
                if target_key:
                    self._merge_vessel_fields(base[target_key], value)
                    continue

                if not allow_new_vessels:
                    continue

                # Merge vessel records field by field.
                # Empty fields from later tables must not overwrite existing values.
                if key not in base:
                    base[key] = value
                elif isinstance(value, dict) and isinstance(base[key], dict):
                    self._merge_vessel_fields(base[key], value)
                continue
            
            if isinstance(value, dict) and "value" in value:
                # Field metadata
                if value.get("value") is not None:
                    # Only overwrite with a nonempty value.
                    base[key] = value
        
        return base


def get_output_filename(input_file: Path) -> str:
    """Replace the _table_parse suffix with _extracted.json.
    
    Example:
        MAB1201_content_list_table_parse.json -> MAB1201_content_list_extracted.json
    """
    stem = input_file.stem
    
    # Remove the _table_parse suffix if present.
    if stem.endswith("_table_parse"):
        stem = stem[:-len("_table_parse")]
    
    # Append the _extracted suffix.
    return f"{stem}_extracted.json"


def process_directory(input_dir: str, output_dir: str, model_path: str = None):
    """Extract tables from input_dir into output_dir using model_path."""
    # Check that the input directory exists.
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the extractor.
    extractor = MaritimeAccidentExtractor(model_path)
    
    if not extractor.qwen_extractor.is_ready():
        logger.error("模型未就绪，退出处理")
        return
    
    # Only process *_table_parse.json to avoid unrelated or duplicate inputs.
    input_files = list(Path(input_dir).glob("*_table_parse.json"))
    
    if not input_files:
        logger.warning(f"输入目录中没有找到 *_table_parse.json 文件: {input_dir}")
        logger.info("请确保输入文件以 _table_parse.json 结尾")
        return
    
    logger.info(f"找到 {len(input_files)} 个 *_table_parse.json 文件待处理")
    
    # Process each input file.
    success_count = 0
    fail_count = 0
    failed_files = []
    
    for input_file in input_files:
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"处理文件: {input_file.name}")
            logger.info(f"{'=' * 60}")

            # Use the standard output naming convention.
            output_filename = get_output_filename(input_file)
            output_file = os.path.join(output_dir, output_filename)

            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(f"输出已存在，跳过: {output_file}")
                success_count += 1
                continue

            # Extract table information.
            result = extractor.extract_from_file(str(input_file))
            
            # Save with the shared schema helper.
            save_to_file(result, output_file)
            
            logger.info(f"结果已保存: {output_file}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"处理文件失败 {input_file.name}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            failed_files.append((input_file.name, str(e)))
    
    # Print extraction totals.
    logger.info(f"\n{'=' * 60}")
    logger.info("处理完成统计:")
    logger.info(f"  成功: {success_count} 个文件")
    logger.info(f"  失败: {fail_count} 个文件")
    if failed_files:
        logger.info("  失败文件列表:")
        for fname, error in failed_files:
            logger.info(f"    - {fname}: {error}")
    logger.info(f"{'=' * 60}")


def main():
    """Run batch table extraction using the configured paths."""
    print("=" * 70)
    print("海事事故表格信息抽取程序 v3.0.9")
    print("基于Qwen7B大模型语义抽取")
    print("=" * 70)
    print()
    print("配置信息:")
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  模型路径: {DEFAULT_QWEN_MODEL_PATH}")
    print(f"  数据结构文件: {STRUCT_FILE_PATH}")
    print()
    print("字段校验规则:")
    print("  1. vessel_type记录船舶类型，不记录建造材料")
    print("     - 排除'aluminum construction'等材料描述")
    print("     - 有效类型: Passenger ship, Cargo ship, Tanker, Tug等")
    print("  2. 单船情况下使用economic_loss补全缺失的property_damage")
    print("     - 当只有一条船且property_damage为空时，自动复制economic_loss")
    print()
    print("=" * 70)
    
    # Process the configured directories.
    process_directory(INPUT_DIR, OUTPUT_DIR, DEFAULT_QWEN_MODEL_PATH)


if __name__ == "__main__":
    main()
