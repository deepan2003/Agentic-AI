import os
import re
import json  # Added missing import to prevent crashing!
import uuid
from dotenv import load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI
from main import agent_system  # Imports your compiled LangGraph workflow

load_dotenv()

# 1. Initialize the Judge LLM
judge_llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)

# HELPER FUNCTION: Safely extracts the first number (1-5) found in the LLM text
def extract_score(response_text: str) -> int:
    match = re.search(r'[1-5]', response_text)
    if match:
        return int(match.group())
    return 3  # Neutral fallback score if the model fails to provide a number

# 2. Define the Target Wrapper Function for LangGraph
def predict_rag_system(inputs: dict) -> dict:
    unique_thread = f"eval_thread_{uuid.uuid4()}"
    config = {"configurable": {"thread_id": unique_thread}}
    response = agent_system.invoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]}, 
        config=config
    )
    final_answer = response["messages"][-1].content
    return {"output": final_answer}

# ====================================================================
# 3. DEFINE THE 4 CORE METRICS WITH SAFE PARSING
# ====================================================================

def evaluate_correctness(run, example) -> dict:
    user_question = example.inputs["question"]
    ground_truth = example.outputs["ground_truth"]
    agent_output = run.outputs["output"]
    
    prompt = (
        f"Question: {user_question}\n"
        f"Ground Truth Reference: {ground_truth}\n"
        f"Agent Output: {agent_output}\n\n"
        "Grade the Agent Output against the Ground Truth. Does it capture the core factual truth?\n"
        "Provide your explanation, but ensure you include a final score from 1 (completely wrong) to 5 (completely correct)."
    )
    raw_response = judge_llm.invoke(prompt).content
    score = extract_score(raw_response)
    return {"key": "correctness", "score": score / 5.0}

def evaluate_concision(run, example) -> dict:
    agent_output = run.outputs["output"]
    
    prompt = (
        f"Agent Output: {agent_output}\n\n"
        "Analyze the response length and clarity. Is it appropriately concise, or is it bloated with filler words?\n"
        "Provide your explanation, but ensure you include a final score from 1 (very wordy) to 5 (perfectly direct)."
    )
    raw_response = judge_llm.invoke(prompt).content
    score = extract_score(raw_response)
    return {"key": "concision", "score": score / 5.0}

def evaluate_groundedness(run, example) -> dict:
    agent_output = run.outputs["output"]
    
    prompt = (
        f"Agent Output: {agent_output}\n\n"
        "Does the agent state assumptions, extrapolate blindly, or apologize unnecessarily without data?\n"
        "Provide your explanation, but ensure you include a final score from 1 (hallucinated) to 5 (completely solid)."
    )
    raw_response = judge_llm.invoke(prompt).content
    score = extract_score(raw_response)
    return {"key": "groundedness", "score": score / 5.0}

def evaluate_retrieval_relevance(run, example) -> dict:
    user_question = example.inputs["question"]
    agent_output = run.outputs["output"]
    
    prompt = (
        f"User Initial Request: {user_question}\n"
        f"Final Provided Context/Answer: {agent_output}\n\n"
        "Did the system successfully address the core intent of the query, or did it fail to get useful data?\n"
        "Provide your explanation, but ensure you include a final score from 1 (irrelevant) to 5 (highly specific)."
    )
    raw_response = judge_llm.invoke(prompt).content
    score = extract_score(raw_response)
    return {"key": "retrieval_relevance", "score": score / 5.0}

# ====================================================================
# 4. EXECUTE EVALUATION RUN
# ====================================================================

if __name__ == "__main__":
    client = Client()
    
    # Directly naming it for your fresh synthetic dataset
    dataset_name = "Agentic_RAG_Synthetic_Bench"
    
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        
        # Open and load your clean generated file
        with open("synthetic_data.json", "r", encoding="utf-8") as f:
            saved_examples = json.load(f)
        
        # Separate them easily into inputs and outputs for LangSmith
        inputs_batch = [{"question": item["question"]} for item in saved_examples]
        outputs_batch = [{"ground_truth": item["ground_truth"]} for item in saved_examples]
        
        # Upload cleanly to the cloud
        client.create_examples(
            inputs=inputs_batch,
            outputs=outputs_batch,
            dataset_id=dataset.id
        )
        print(f"🎉 Successfully pushed AI-generated dataset to LangSmith!")

    print("🚀 Running automated evaluation via LangSmith Judges...")
    
    eval_results = client.evaluate(
        predict_rag_system,
        data=dataset_name,
        evaluators=[
            evaluate_correctness,
            evaluate_concision,
            evaluate_groundedness,
            evaluate_retrieval_relevance
        ],
        experiment_prefix="llama3-agent-rag-bench"
    )
    print("✅ Evaluation complete! Check your LangSmith Web UI dashboard to view live charts and score metrics.")