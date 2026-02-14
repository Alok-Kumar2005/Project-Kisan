import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import asyncio
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Import your LLM and evaluators
from src.ai_component.llm import LLMChainFactory
from evaluation.evaluators import EVALUATORS

load_dotenv()


# Initialize LangSmith client
client = Client()


def load_test_cases(filepath="evaluation/datasets/test_cases.json"):
    """Load test cases from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['test_cases']


def create_dataset(dataset_name="agricultural_qa_v1", test_cases=None):
    if test_cases is None:
        test_cases = load_test_cases()
    
    # Check if dataset exists
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"✓ Using existing dataset: {dataset_name}")
        return dataset
    except:
        pass
    
    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Test cases for agricultural AI assistant quality evaluation"
    )
    
    print(f"✓ Created new dataset: {dataset_name}")
    
    # Add examples to dataset
    for test_case in test_cases:
        client.create_example(
            dataset_id=dataset.id,
            inputs={"question": test_case["question"]},
            outputs={
                "reference_answer": test_case.get("reference_answer", ""),
                "expected_topics": test_case.get("expected_topics", []),
                "category": test_case.get("category", "")
            },
            metadata={
                "id": test_case.get("id", ""),
                "category": test_case.get("category", "")
            }
        )
    
    print(f"✓ Added {len(test_cases)} test cases to dataset")
    return dataset


async def predict_response(inputs: dict, llm_factory: LLMChainFactory):
    question = inputs.get("question", "")
    
    # Create a simple prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Krishi Shayak, an expert agricultural AI assistant helping farmers in India.
Provide clear, accurate, and actionable advice on farming, crops, diseases, market information, and agricultural practices.
Keep responses concise and farmer-friendly."""),
        ("user", "{question}")
    ])
    
    try:
        # Get LLM chain
        chain = await llm_factory.get_llm_chain_async(prompt)
        
        # Get response
        response = await chain.ainvoke({"question": question})
        
        # Extract content
        if hasattr(response, 'content'):
            output = response.content
        else:
            output = str(response)
        
        return {"output": output}
    
    except Exception as e:
        print(f"Error generating prediction: {e}")
        return {"output": f"Error: {str(e)}"}


def run_evaluation(
    dataset_name="agricultural_qa_v1",
    model_type="gemini",
    experiment_name=None
):
    print(f"\n{'='*60}")
    print(f"🚀 Starting LLM Evaluation")
    print(f"{'='*60}\n")
    
    # Load test cases
    test_cases = load_test_cases()
    print(f"✓ Loaded {len(test_cases)} test cases")
    
    # Create/get dataset
    dataset = create_dataset(dataset_name, test_cases)
    
    # Initialize LLM
    llm_factory = LLMChainFactory(model_type=model_type)
    print(f"✓ Initialized {model_type.upper()} model")
    
    # Create prediction function
    async def predict_wrapper(inputs: dict):
        return await predict_response(inputs, llm_factory)
    
    # Set experiment name
    if experiment_name is None:
        from datetime import datetime
        experiment_name = f"{model_type}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"✓ Experiment name: {experiment_name}")
    print(f"\nRunning predictions and evaluations...")
    print(f"This may take a few minutes...\n")
    
    # Run evaluation
    results = evaluate(
        predict_wrapper,
        data=dataset_name,
        evaluators=EVALUATORS,
        experiment_prefix=experiment_name,
        max_concurrency=2  # Limit concurrent requests
    )
    
    return results


def display_results(results):
    metrics = {}
    
    for result in results:
        for eval_result in result.get('evaluation_results', {}).get('results', []):
            key = eval_result.get('key', 'Unknown')
            score = eval_result.get('score', 0.0)
            
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(score)
    
    for metric, scores in sorted(metrics.items()):
        avg_score = sum(scores) / len(scores) if scores else 0.0
        percentage = avg_score * 100
        
        bar_length = int(percentage / 5)  
        bar = "=" * bar_length + " " * (20 - bar_length)
        
        print(f"{metric:25s} {bar} {percentage:5.1f}%")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete!")
    print(f"{'='*60}\n")
    
    project_name = os.getenv("LANGSMITH_PROJECT", "project-kisan")
    print(f"View detailed results in LangSmith:")
    print(f"https://smith.langchain.com/projects/p/{project_name}/datasets\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM evaluation using LangSmith")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini",
        choices=["gemini", "groq"],
        help="LLM model to evaluate (default: gemini)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="agricultural_qa_v1",
        help="Dataset name (default: agricultural_qa_v1)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    try:
        required_env_vars = ["LANGSMITH_API_KEY", "GOOGLE_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Missing required environment variables: {', '.join(missing_vars)}")
            print(f"Please set them in your .env file")
            return
        
        results = run_evaluation(
            dataset_name=args.dataset,
            model_type=args.model,
            experiment_name=args.experiment
        )
        
        display_results(results)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
