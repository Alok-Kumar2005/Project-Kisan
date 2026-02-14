import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langsmith.evaluation import evaluator
from langsmith.schemas import Run, Example
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


# Initialize LLM for LLM-as-judge evaluation
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


@evaluator
def agricultural_relevance_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if the response is relevant to agriculture and farming.
    Score: 1.0 if highly relevant, 0.5 if partially relevant, 0.0 if not relevant.
    """
    question = example.inputs.get("question", "")
    answer = run.outputs.get("output", "") if run.outputs else ""
    
    if not answer:
        return {"key": "agricultural_relevance", "score": 0.0}
    
    # Simple keyword-based check
    agricultural_keywords = [
        "crop", "farm", "soil", "seed", "harvest", "yield", "plant", "agriculture",
        "fertilizer", "pest", "disease", "irrigation", "water", "market", "mandi",
        "wheat", "rice", "maize", "cotton", "vegetable", "organic", "kharif", "rabi"
    ]
    
    answer_lower = answer.lower()
    keyword_count = sum(1 for keyword in agricultural_keywords if keyword in answer_lower)
    
    if keyword_count >= 3:
        score = 1.0
    elif keyword_count >= 1:
        score = 0.5
    else:
        score = 0.0
    
    return {
        "key": "agricultural_relevance",
        "score": score,
        "comment": f"Found {keyword_count} agricultural keywords"
    }


@evaluator
def helpfulness_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if the response provides actionable, helpful advice.
    Uses LLM-as-judge for assessment.
    """
    question = example.inputs.get("question", "")
    answer = run.outputs.get("output", "") if run.outputs else ""
    
    if not answer:
        return {"key": "helpfulness", "score": 0.0}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert evaluator of agricultural advice. 
Your task is to rate how helpful and actionable the answer is for a farmer.

Rate on a scale:
- 1.0: Highly helpful with specific, actionable advice
- 0.7: Helpful with some actionable points
- 0.4: Somewhat helpful but vague
- 0.0: Not helpful or incorrect

Respond with ONLY a number between 0.0 and 1.0."""),
        ("user", """Question: {question}

Answer: {answer}

Score (0.0-1.0):""")
    ])
    
    try:
        chain = prompt | judge_llm
        result = chain.invoke({"question": question, "answer": answer})
        score_text = result.content.strip()
        
        # Extract first number from response
        import re
        match = re.search(r'0\.\d+|1\.0', score_text)
        score = float(match.group()) if match else 0.5
        
    except Exception as e:
        print(f"Error in helpfulness evaluation: {e}")
        score = 0.5  # Default to neutral score on error
    
    return {
        "key": "helpfulness",
        "score": score
    }


@evaluator
def correctness_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates factual correctness using reference answer if available.
    Uses LLM-as-judge for semantic comparison.
    """
    question = example.inputs.get("question", "")
    answer = run.outputs.get("output", "") if run.outputs else ""
    reference = example.outputs.get("reference_answer", "") if example.outputs else ""
    expected_topics = example.outputs.get("expected_topics", []) if example.outputs else []
    
    if not answer:
        return {"key": "correctness", "score": 0.0}
    
    # If we have a reference answer, use LLM to compare
    if reference:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert agricultural advisor evaluating answer correctness.
Compare the given answer with the reference answer.

Rate on a scale:
- 1.0: Factually correct and complete
- 0.7: Mostly correct with minor issues
- 0.4: Partially correct
- 0.0: Incorrect or misleading

Respond with ONLY a number between 0.0 and 1.0."""),
            ("user", """Question: {question}

Reference Answer: {reference}

Student Answer: {answer}

Score (0.0-1.0):""")
        ])
        
        try:
            chain = prompt | judge_llm
            result = chain.invoke({
                "question": question,
                "reference": reference,
                "answer": answer
            })
            score_text = result.content.strip()
            
            import re
            match = re.search(r'0\.\d+|1\.0', score_text)
            score = float(match.group()) if match else 0.5
            
        except Exception as e:
            print(f"Error in correctness evaluation: {e}")
            score = 0.5
    
    # Otherwise, check if expected topics are covered
    elif expected_topics:
        answer_lower = answer.lower()
        covered_topics = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
        score = covered_topics / len(expected_topics) if expected_topics else 0.5
    
    else:
        score = 0.5  # Neutral if no reference
    
    return {
        "key": "correctness",
        "score": score
    }


@evaluator
def conciseness_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if the answer is concise and to-the-point.
    Penalizes overly long or overly short answers.
    """
    answer = run.outputs.get("output", "") if run.outputs else ""
    
    if not answer:
        return {"key": "conciseness", "score": 0.0}
    
    word_count = len(answer.split())
    
    # Ideal range: 50-200 words
    if 50 <= word_count <= 200:
        score = 1.0
    elif 30 <= word_count < 50 or 200 < word_count <= 300:
        score = 0.7
    elif 15 <= word_count < 30 or 300 < word_count <= 400:
        score = 0.4
    else:
        score = 0.2
    
    return {
        "key": "conciseness",
        "score": score,
        "comment": f"Word count: {word_count}"
    }


@evaluator
def language_quality_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates language quality: grammar, clarity, professional tone.
    Uses simple heuristics.
    """
    answer = run.outputs.get("output", "") if run.outputs else ""
    
    if not answer:
        return {"key": "language_quality", "score": 0.0}
    
    score = 1.0
    issues = []
    
    # Check for complete sentences
    if not answer.strip().endswith(('.', '!', '?')):
        score -= 0.2
        issues.append("incomplete_ending")
    
    # Check for reasonable sentence structure
    sentences = answer.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    
    if avg_sentence_length < 5:
        score -= 0.2
        issues.append("too_short_sentences")
    elif avg_sentence_length > 40:
        score -= 0.2
        issues.append("too_long_sentences")
    
    # Check for proper capitalization
    if answer and answer[0].islower():
        score -= 0.1
        issues.append("capitalization")
    
    score = max(0.0, min(1.0, score))
    
    return {
        "key": "language_quality",
        "score": score,
        "comment": f"Issues: {', '.join(issues)}" if issues else "Good quality"
    }


# Export all evaluators
EVALUATORS = [
    agricultural_relevance_evaluator,
    helpfulness_evaluator,
    correctness_evaluator,
    conciseness_evaluator,
    language_quality_evaluator
]


if __name__ == "__main__":
    # Test evaluators
    from langsmith.schemas import Run, Example
    
    test_run = Run(
        id="test",
        name="test",
        run_type="chain",
        inputs={},
        outputs={"output": "Plant rice during monsoon season in Bihar. Use high-yielding varieties and ensure proper water management."}
    )
    
    test_example = Example(
        id="test",
        created_at="2024-01-01",
        inputs={"question": "What crops grow best in Bihar during monsoon?"},
        outputs={
            "reference_answer": "Rice is the main monsoon crop in Bihar.",
            "expected_topics": ["rice", "monsoon"]
        }
    )
    
    print("Testing evaluators:")
    for evaluator_func in EVALUATORS:
        result = evaluator_func(test_run, test_example)
        print(f"{result['key']}: {result['score']}")
