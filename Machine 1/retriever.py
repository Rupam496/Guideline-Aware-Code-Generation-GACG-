"""
retriever.py
============
Smart retrieval logic for GRPO + RAG pipeline.

Strategy:
- Code <= 40 lines : embed prompt + full code directly
- Code >  40 lines : hybrid (AST signals + first 30 lines for formatting)

This ensures embedding quality stays sharp regardless of code length,
and ALL rules (HIGH/MEDIUM/LOW) are covered.
"""

import ast
import re
from typing import List

from vector_store import VectorStore
from guideline_chunker import RuleChunk


# =====================================================
# AST Signal Extractor
# Captures HIGH + MEDIUM priority rule violations
# from long code without diluting the embedding
# =====================================================
def extract_ast_signals(code: str) -> str:
    """
    Reduce long code to a compact set of violation-carrying signals.
    Covers: mutable defaults, bare except, missing type hints,
            lambda assignments, globals, None comparison,
            naming conventions, context manager misuse.
    Falls back to first 20 lines on SyntaxError.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "\n".join(code.splitlines()[:20])

    signals = []
    seen = set()

    def add(s: str):
        if s not in seen:
            seen.add(s)
            signals.append(s)

    for node in ast.walk(tree):

        # Rule 29: mutable default arguments
        # Rule 23: missing type hints
        # Rule 3:  naming convention (snake_case)
        if isinstance(node, ast.FunctionDef):

            args = node.args
            arg_names = [a.arg for a in args.args]

            mutable = [
                ast.dump(d)[:40]
                for d in args.defaults
                if isinstance(d, (ast.List, ast.Dict, ast.Set))
            ]

            no_hints = [
                a.arg for a in args.args
                if a.annotation is None
            ]

            has_return_hint = node.returns is not None

            sig = f"def {node.name}({', '.join(arg_names)})"

            if mutable:
                add(f"{sig}  # mutable default: {mutable}")
            else:
                add(sig)

            if no_hints:
                add(f"  # missing type hints on: {no_hints}, return_hint={has_return_hint}")

            # snake_case check
            if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                add(f"  # non-snake_case function name: {node.name}")

        # Rule 3: class naming (PascalCase)
        if isinstance(node, ast.ClassDef):
            if not node.name[0].isupper():
                add(f"# non-PascalCase class: {node.name}")

        # Rule 19: bare except
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                add("except:  # bare except — catches everything")
            else:
                try:
                    add(f"except {ast.unparse(node.type)}:")
                except Exception:
                    add("except <specific>:")

        # Rule 20: exception chaining
        # Raise inside except without 'from e'
        if isinstance(node, ast.Raise):
            if node.exc is not None and node.cause is None:
                add("raise <Exception>()  # missing 'from e' chain")

        # Rule 21: open() without with
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "open":
                add("open() call — verify context manager usage")

        # Rule 27: lambda assignment
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Lambda):
                try:
                    add(f"lambda assignment: {ast.unparse(node)[:60]}")
                except Exception:
                    add("lambda assignment detected")

        # Rule 28: global statement
        if isinstance(node, ast.Global):
            add(f"global {', '.join(node.names)}  # avoid globals")

        # Rule 17: comparison to None with == or !=
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    try:
                        left = ast.unparse(node.left)
                        if "None" in left or any(
                            "None" in ast.unparse(c)
                            for c in node.comparators
                        ):
                            add("== None or != None comparison detected")
                    except Exception:
                        pass

        # Rule 18: comparison to True/False
        if isinstance(node, ast.Compare):
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and comp.value in (True, False):
                    add(f"comparison to boolean literal: == {comp.value}")

        # Rule 16: inconsistent return
        if isinstance(node, ast.FunctionDef):
            returns = [
                n for n in ast.walk(node)
                if isinstance(n, ast.Return)
            ]
            bare = [r for r in returns if r.value is None]
            valued = [r for r in returns if r.value is not None]
            if bare and valued:
                add(f"# mixed return in {node.name}: some bare, some valued")

        # Rule 22: map/filter with lambda
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("map", "filter"):
                if node.args and isinstance(node.args[0], ast.Lambda):
                    add("map/filter with lambda — prefer list comprehension")

    return "\n".join(signals) if signals else code[:300]


# =====================================================
# Hybrid Retrieval
# For code > 40 lines:
#   Stream 1 — AST signals    (HIGH/MEDIUM rules)
#   Stream 2 — first 30 lines (LOW formatting rules)
# =====================================================
def hybrid_retrieve(
    store: VectorStore,
    prompt: str,
    code: str,
    top_k: int = 5,
) -> List[RuleChunk]:

    # --- Stream 1: AST signals for structural violations ---
    ast_signals = extract_ast_signals(code)
    ast_query   = f"Task: {prompt}\n\nCode signals:\n{ast_signals}"
    ast_hits    = store.retrieve(ast_query, top_k=top_k)

    # --- Stream 2: first 30 lines for LOW formatting rules ---
    # Indentation, imports, whitespace, blank lines, line length,
    # string quotes, trailing whitespace — all visible in top 30 lines
    first_30    = "\n".join(code.splitlines()[:30])
    text_query  = f"Task: {prompt}\n\nCode:\n{first_30}"
    text_hits   = store.retrieve(text_query, top_k=max(top_k - 2, 3))

    # --- Merge, deduplicate, preserve AST-hit priority ---
    seen    = set()
    results = []

    for chunk in ast_hits + text_hits:
        if chunk.rule_number not in seen:
            seen.add(chunk.rule_number)
            results.append(chunk)

    return results


# =====================================================
# Smart Retrieve — public API
# Called once per generated code candidate
# =====================================================
def smart_retrieve(
    store: VectorStore,
    prompt: str,
    code: str,
    top_k: int = 5,
) -> List[RuleChunk]:
    """
    Automatically selects retrieval strategy based on code length.

    <= 40 lines : prompt + full code  (sharp, all rules, direct)
    >  40 lines : hybrid              (AST + first-30-lines)

    Returns:
        List of RuleChunk — relevant guideline chunks for this code
    """

    lines = code.splitlines()

    if len(lines) <= 40:
        # Direct embedding — quality is fine at this length
        query = f"Task: {prompt}\n\nCode:\n{code}"
        return store.retrieve(query, top_k=top_k)

    else:
        # Hybrid — maintains quality for long code
        return hybrid_retrieve(store, prompt, code, top_k=top_k)
