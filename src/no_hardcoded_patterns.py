"""
NO HARDCODED PATTERNS RULE ENFORCEMENT
=====================================

This module enforces the rule: NO HARDCODED PATTERNS OR TOPIC-SPECIFIC LOGIC

FORBIDDEN PATTERNS:
==================

1. HARDCODED KEYWORD LISTS:
   ❌ car_brands = ["honda", "toyota", "ford", "bmw"]
   ❌ job_titles = ["engineer", "doctor", "teacher"]
   ❌ locations = ["city", "state", "country"]

2. HARDCODED REGEX PATTERNS:
   ❌ r'drive.*?(\d{4}.*?honda.*?crv)'
   ❌ r'work at ([a-zA-Z\s]+)'
   ❌ r'friend.*?name is ([a-zA-Z\s]+)'

3. HARDCODED TOPIC MAPPINGS:
   ❌ topic_mappings = {"car": ["drive", "vehicle"], "work": ["job", "career"]}
   ❌ company_tickers = {"nvidia": "NVDA", "apple": "AAPL"}

4. HARDCODED EXTRACTION METHODS:
   ❌ def _extract_car_info(text)
   ❌ def _extract_work_info(text)
   ❌ def _extract_friend_name(text)

5. HARDCODED ROUTING LOGIC:
   ❌ if "stock price" in query: use_financial_agent()
   ❌ if "weather" in query: use_weather_agent()

ALLOWED APPROACHES:
==================

1. SEMANTIC SIMILARITY:
   ✅ Use text similarity scoring
   ✅ Use word overlap analysis
   ✅ Use embedding-based search

2. GENERIC CONTENT EXTRACTION:
   ✅ Extract relevant sentences based on query keywords
   ✅ Use general text processing
   ✅ Let search engines handle topic-specific logic

3. DYNAMIC AGENT CONVERSATION:
   ✅ Let agents discuss and decide routing
   ✅ Use capability-based matching
   ✅ Allow agents to self-assess

4. UNIVERSAL MEMORY STORAGE:
   ✅ Store ALL user statements
   ✅ Use semantic search for retrieval
   ✅ No topic-specific storage categories

ENFORCEMENT FUNCTIONS:
=====================
"""

import re
import ast
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class HardcodedPatternDetector:
    """Detects hardcoded patterns in code that violate the generic approach."""
    
    def __init__(self):
        self.violations = []
    
    def check_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Check a Python file for hardcoded pattern violations."""
        violations = []

        # Skip checking the rule enforcement file itself
        if 'no_hardcoded_patterns.py' in file_path:
            return violations

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for hardcoded keyword lists
            violations.extend(self._check_hardcoded_lists(content, file_path))

            # Check for hardcoded regex patterns
            violations.extend(self._check_hardcoded_regex(content, file_path))

            # Check for hardcoded extraction methods
            violations.extend(self._check_extraction_methods(content, file_path))

            # Check for hardcoded topic mappings
            violations.extend(self._check_topic_mappings(content, file_path))

        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")

        return violations
    
    def _check_hardcoded_lists(self, content: str, file_path: str) -> List[Dict]:
        """Check for hardcoded keyword lists."""
        violations = []
        
        # Look for suspicious list patterns (hardcoded topic-specific lists)
        suspicious_patterns = [
            r'car_brands\s*=\s*\[',
            r'job_titles\s*=\s*\[',
            r'locations\s*=\s*\[',
            r'company_names\s*=\s*\[',
            r'stock_tickers\s*=\s*\{',
            r'factual_patterns\s*=\s*\{',
            r'company_tickers\s*=\s*\{',
            # Only flag hardcoded lists of topic-specific terms, not individual mentions
            r'\[\s*["\'](honda|toyota|ford|bmw|audi)["\'].*?\]',
            r'\[\s*["\'](engineer|doctor|teacher|lawyer)["\'].*?\]',
            r'\{\s*["\'](nvidia|apple|microsoft)["\']:\s*["\'](NVDA|AAPL|MSFT)["\']'
        ]
        
        for pattern in suspicious_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                violations.append({
                    'type': 'hardcoded_list',
                    'file': file_path,
                    'line': line_num,
                    'pattern': match.group(),
                    'message': 'Hardcoded keyword list detected - use semantic approach instead'
                })
        
        return violations
    
    def _check_hardcoded_regex(self, content: str, file_path: str) -> List[Dict]:
        """Check for hardcoded regex patterns."""
        violations = []
        
        # Look for topic-specific regex patterns
        suspicious_regex = [
            r'r["\'].*?(honda|toyota|ford|bmw).*?["\']',
            r'r["\'].*?(work at|job at|employed).*?["\']',
            r'r["\'].*?(friend.*?name|best friend).*?["\']',
            r'r["\'].*?(drive.*?car|car.*?drive).*?["\']'
        ]
        
        for pattern in suspicious_regex:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                violations.append({
                    'type': 'hardcoded_regex',
                    'file': file_path,
                    'line': line_num,
                    'pattern': match.group(),
                    'message': 'Hardcoded regex pattern detected - use generic text processing instead'
                })
        
        return violations
    
    def _check_extraction_methods(self, content: str, file_path: str) -> List[Dict]:
        """Check for hardcoded extraction methods."""
        violations = []

        # Look for topic-specific extraction methods (but exclude legitimate generic ones)
        extraction_patterns = [
            r'def _extract_car_info',
            r'def _extract_work_info',
            r'def _extract_friend_name',
            r'def _extract_price_info',
            r'def _extract_ticker',
            r'def _extract_company_info',
            r'def _extract_stock_info'
        ]

        # Exclude legitimate generic extraction methods
        allowed_patterns = [
            r'def _extract_image_info',  # Generic image extraction
            r'def _extract_relevant_content',  # Generic content extraction
            r'def _extract_and_store_user_info'  # Generic user info storage
        ]

        for pattern in extraction_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                # Check if this is an allowed pattern
                is_allowed = any(re.search(allowed, match.group()) for allowed in allowed_patterns)
                if not is_allowed:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        'type': 'hardcoded_extraction',
                        'file': file_path,
                        'line': line_num,
                        'pattern': match.group(),
                        'message': 'Hardcoded extraction method detected - use generic content extraction instead'
                    })

        return violations
    
    def _check_topic_mappings(self, content: str, file_path: str) -> List[Dict]:
        """Check for hardcoded topic mappings."""
        violations = []
        
        # Look for topic mapping dictionaries
        mapping_patterns = [
            r'company_tickers\s*=\s*\{',
            r'topic_mappings\s*=\s*\{',
            r'category_keywords\s*=\s*\{',
            r'["\'](nvidia|apple|microsoft)["\']:\s*["\'](NVDA|AAPL|MSFT)["\']'
        ]
        
        for pattern in mapping_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                violations.append({
                    'type': 'hardcoded_mapping',
                    'file': file_path,
                    'line': line_num,
                    'pattern': match.group(),
                    'message': 'Hardcoded topic mapping detected - use dynamic/semantic approach instead'
                })
        
        return violations

def check_codebase_for_violations(src_dir: str = "src") -> List[Dict[str, Any]]:
    """Check the entire codebase for hardcoded pattern violations."""
    import os
    
    detector = HardcodedPatternDetector()
    all_violations = []
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                violations = detector.check_file(file_path)
                all_violations.extend(violations)
    
    return all_violations

def print_violation_report(violations: List[Dict[str, Any]]):
    """Print a formatted report of violations."""
    if not violations:
        print("✅ NO HARDCODED PATTERNS FOUND - CODEBASE IS CLEAN!")
        return
    
    print(f"❌ FOUND {len(violations)} HARDCODED PATTERN VIOLATIONS:")
    print("=" * 60)
    
    for violation in violations:
        print(f"File: {violation['file']}")
        print(f"Line: {violation['line']}")
        print(f"Type: {violation['type']}")
        print(f"Pattern: {violation['pattern']}")
        print(f"Message: {violation['message']}")
        print("-" * 40)

if __name__ == "__main__":
    # Run the check
    violations = check_codebase_for_violations()
    print_violation_report(violations)
