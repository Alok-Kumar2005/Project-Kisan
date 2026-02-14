import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
import json

load_dotenv()

def test_environment():
    required_vars = {
        "LANGSMITH_API_KEY": "LangSmith API key",
        "LANGSMITH_TRACING": "LangSmith tracing enabled",
        "LANGSMITH_PROJECT": "LangSmith project name",
        "GOOGLE_API_KEY": "Google API key for Gemini"
    }
    
    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask the key for security
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"  ✓ {var}: {masked}")
        else:
            print(f"  ✗ {var}: Not set")
            missing.append(f"{var} ({description})")
    
    if missing:
        print(f"Missing variables: {', '.join(missing)}")
        return False
    else:
        print(f"All environment variables configured!")
        return True


def test_dataset():
    dataset_path = "evaluation/datasets/test_cases.json"
    
    if not os.path.exists(dataset_path):
        print(f"Test cases file not found at {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = data.get('test_cases', [])
        print(f"  ✓ Found {len(test_cases)} test cases")
        
        # Show categories
        categories = set(tc.get('category', 'Unknown') for tc in test_cases)
        print(f"  ✓ Categories: {', '.join(sorted(categories))}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading test cases: {e}")
        return False


def test_evaluators():
    try:
        from evaluation.evaluators import EVALUATORS
        
        print(f"Found {len(EVALUATORS)} evaluators:")
        for evaluator in EVALUATORS:
            print(f"     - {evaluator.__name__}")
        
        return True
    except Exception as e:
        print(f"Error importing evaluators: {e}")
        return False


def test_llm_connection():
    try:
        from src.ai_component.llm import LLMChainFactory
        
        factory = LLMChainFactory(model_type="gemini")
        print(f"LLM factory initialized")
        print(f"Model: {factory.gemini_model_name}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error initializing LLM: {e}")
        return False


def main():
    print("=" * 60)
    print("LangSmith Evaluation System - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Environment", test_environment),
        ("Dataset", test_dataset),
        ("Evaluators", test_evaluators),
        ("LLM Connection", test_llm_connection)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"Unexpected error in {name} test: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:20s} {status}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\nAll tests passed! You're ready to run evaluations!")
        print("\nNext step:")
        print("  python evaluation/run_eval.py")
    else:
        print("\nSome tests failed. Please fix the issues above.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
