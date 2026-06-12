"""
guideline_chunker.py
====================
Clean production-ready chunker for GRPO + RAG pipeline

Features:
- One rule = one chunk
- Correct / Wrong code extracted cleanly
- Description contains ONLY natural-language rule text
- Metadata for retrieval
"""

import re
from dataclasses import dataclass, asdict
from typing import List


# =====================================================
# DATA CLASS
# =====================================================
@dataclass
class RuleChunk:
    rule_number: str
    rule_title: str
    description: str
    correct_code: str
    wrong_code: str
    keywords: str = ""
    ast_pattern: str = ""
    bug_type: str = ""

    def as_flat_text(self) -> str:
        return f"""
Rule {self.rule_number}: {self.rule_title}

Bug Type:
{self.bug_type}

AST Pattern:
{self.ast_pattern}

Keywords:
{self.keywords}

Description:
{self.description}

Correct Example:
{self.correct_code}

Wrong Example:
{self.wrong_code}
""".strip()

    def to_dict(self):
        return asdict(self)


# =====================================================
# CHUNKER
# =====================================================
class GuidelineChunker:

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.text = f.read()

    # -------------------------------------------------
    def parse(self) -> List[RuleChunk]:

        raw_blocks = re.split(r"\n---\n", self.text)

        chunks = []

        for block in raw_blocks:
            block = block.strip()

            if not block:
                continue

            chunk = self._parse_block(block)

            if chunk:
                chunks.append(chunk)

        print(f"✅ Parsed {len(chunks)} guideline chunks.")
        return chunks

    # -------------------------------------------------
    def _parse_block(self, block: str):

        # ---------------------------------------------
        # Header
        # ---------------------------------------------
        header = re.search(r"##\s*(\d+)\.\s*(.+)", block)

        if not header:
            return None

        rule_number = header.group(1).strip()
        rule_title = header.group(2).strip()

        # ---------------------------------------------
        # Correct code
        # ---------------------------------------------
        correct_match = re.search(
            r"###\s*Correct:\s*\n+(.*?)(?=\n###\s*Wrong:)",
            block,
            re.S
        )

        correct_code = (
            correct_match.group(1).strip()
            if correct_match else ""
        )

        # ---------------------------------------------
        # Wrong code
        # ---------------------------------------------
        wrong_match = re.search(
            r"###\s*Wrong:\s*\n+(.*)$",
            block,
            re.S
        )

        wrong_code = (
            wrong_match.group(1).strip()
            if wrong_match else ""
        )

        # ---------------------------------------------
        # Description ONLY
        # ---------------------------------------------
        desc = block

        # remove header
        desc = re.sub(
            r"##\s*\d+\.\s*.+",
            "",
            desc,
            count=1
        )

        # remove correct section
        desc = re.sub(
            r"###\s*Correct:\s*\n+.*?(?=\n###\s*Wrong:)",
            "",
            desc,
            flags=re.S
        )

        # remove wrong section
        desc = re.sub(
            r"\n###\s*Wrong:\s*\n+.*$",
            "",
            desc,
            flags=re.S
        )

        description = desc.strip()

        # ---------------------------------------------
        # Metadata
        # ---------------------------------------------
        keywords = self._make_keywords(rule_title)
        ast_pattern = self._make_ast_pattern(rule_title)
        bug_type = self._make_bug_type(rule_title)

        return RuleChunk(
            rule_number=rule_number,
            rule_title=rule_title,
            description=description,
            correct_code=correct_code,
            wrong_code=wrong_code,
            keywords=keywords,
            ast_pattern=ast_pattern,
            bug_type=bug_type
        )

    # -------------------------------------------------
    def _make_keywords(self, title: str) -> str:

        t = title.lower()

        words = re.sub(r"[^a-z0-9 ]", " ", t).split()
        words = [w for w in words if len(w) > 2]

        extra = []

        if "mutable" in t:
            extra += ["list default", "dict default", "shared state"]

        if "exception" in t:
            extra += ["try except", "bare except"]

        if "naming" in t:
            extra += ["snake_case", "PascalCase"]

        if "whitespace" in t:
            extra += ["pep8", "spacing"]

        if "lambda" in t:
            extra += ["anonymous function"]

        if "global" in t:
            extra += ["global variable"]

        if "type hints" in t:
            extra += ["typing", "annotations"]

        if "imports" in t:
            extra += ["import order", "stdlib", "third party"]

        if "line length" in t:
            extra += ["79 chars", "pep8 line"]

        if "comparison" in t:
            extra += ["conditional check"]

        seen = set()
        final = []

        for item in words + extra:
            if item not in seen:
                seen.add(item)
                final.append(item)

        return ", ".join(final)

    # -------------------------------------------------
    def _make_ast_pattern(self, title: str) -> str:

        t = title.lower()

        mapping = {
            "mutable default arguments":
                "FunctionDef(default=List|Dict|Set)",

            "exception handling":
                "Try(ExceptHandler(type=None))",

            "exception chaining":
                "Raise(without cause inside except)",

            "comparison to none":
                "Compare(== None / != None)",

            "comparison to booleans":
                "Compare(True / False)",

            "lambda assignments":
                "Assign(value=Lambda)",

            "global variables":
                "Global(name)",

            "context managers":
                "open() without with",

            "type hints":
                "FunctionDef missing annotations",

            "imports":
                "Import(multiple names same line)",

            "semicolons":
                "Multiple statements one line",

            "list comprehensions":
                "map/filter(lambda)",

            "return statements":
                "Mixed return value / bare return",
        }

        for key, value in mapping.items():
            if key in t:
                return value

        return "Style / Formatting Rule"

    # -------------------------------------------------
    def _make_bug_type(self, title: str) -> str:

        t = title.lower()

        if "mutable" in t:
            return "Runtime Bug"

        if "exception" in t:
            return "Error Handling"

        if "comparison" in t:
            return "Logic Bug"

        if "global" in t:
            return "Maintainability"

        if "lambda" in t:
            return "Readability"

        if "type hints" in t:
            return "Static Analysis"

        if "imports" in t:
            return "Style"

        if "whitespace" in t or "line length" in t:
            return "Formatting"

        if "naming" in t:
            return "Readability"

        return "General Quality"