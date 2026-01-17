#!/usr/bin/env python3
"""
Comprehensive Test Suite for RAG Explanation Modules
Tests all 21 Python modules to ensure they work correctly
"""

import sys
import os
import subprocess
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_module_test(module_path, description):
    """Run a single module and return success status"""
    print(f"\n🧪 Testing: {description}")
    print(f"📁 File: {module_path}")

    try:
        # Run the module
        result = subprocess.run([sys.executable, module_path],
                              capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ PASSED")
            return True
        else:
            print("❌ FAILED")
            print(f"Error output: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (60s)")
        return False
    except Exception as e:
        print(f"💥 ERROR: {str(e)}")
        return False

def main():
    """Run all module tests"""
    print("🚀 RAG Explanation System - Comprehensive Test Suite")
    print("=" * 60)

    # Define all modules to test
    modules_to_test = [
        # Module 01: Prerequisites
        ("module01_prerequisites/bash.py", "Bash Scripting Basics"),
        ("module01_prerequisites/sql.py", "SQL Database Operations"),
        ("module01_prerequisites/numpy_basics.py", "NumPy Array Operations"),
        ("module01_prerequisites/scikit_learn.py", "Scikit-learn ML Basics"),

        # Module 02: Vector Mathematics
        ("module02_vector_mathematics/vectors.py", "Vector Operations"),
        ("module02_vector_mathematics/dot_product.py", "Dot Product Calculations"),
        ("module02_vector_mathematics/cosine_similarity.py", "Cosine Similarity"),

        # Module 03: Embeddings and Search
        ("module03_embeddings_search/embeddings.py", "Text Embeddings"),
        ("module03_embeddings_search/semantic_search.py", "Semantic Search"),
        ("module03_embeddings_search/vector_keyword_search.py", "Vector Keyword Search"),

        # Module 04: Vector Databases
        ("module04_vector_databases/vector_databases.py", "Simple Vector Database"),
        ("module04_vector_databases/vector_db_implementations.py", "Vector DB Implementations"),

        # Module 05: Indexing
        ("module05_indexing/indexing.py", "Indexing Concepts"),
        ("module05_indexing/indexing_algorithms.py", "Indexing Algorithms"),
        ("module05_indexing/indexing_approaches.py", "Indexing Approaches"),
        ("module05_indexing/precision_problem.py", "Precision Problem"),

        # Module 06: Text Processing
        ("module06_text_processing/chunking.py", "Text Chunking"),

        # Module 07: LLM Integration
        ("module07_llm_prompting/prompt_engineering.py", "Prompt Engineering"),
        ("module07_llm_prompting/fine_tuning.py", "Fine-tuning Concepts"),
        ("module07_llm_prompting/llm.py", "LLM Text Generation"),

        # Module 08: Complete RAG
        ("module08_rag/rag.py", "Complete RAG Pipeline"),
    ]

    total_modules = len(modules_to_test)
    passed_tests = 0
    failed_tests = []

    print(f"📊 Testing {total_modules} modules...\n")

    # Test each module
    for module_path, description in modules_to_test:
        if run_module_test(module_path, description):
            passed_tests += 1
        else:
            failed_tests.append((module_path, description))

    # Print summary
    print("\n" + "=" * 60)
    print("📈 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed_tests}/{total_modules}")
    print(f"❌ Failed: {len(failed_tests)}/{total_modules}")

    if failed_tests:
        print("\n💥 FAILED MODULES:")
        for path, desc in failed_tests:
            print(f"  - {desc} ({path})")
    else:
        print("\n🎉 ALL TESTS PASSED! The RAG system is fully operational.")

    print("\n📝 Note: Some modules may show warnings for missing optional dependencies,")
    print("       but they should still run with fallback implementations.")

    # Return appropriate exit code
    return 0 if len(failed_tests) == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)