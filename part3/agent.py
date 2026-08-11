# part3/agent.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from part3.tools import check_return_risk, classify_product_image

# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    current_query: str
    intent: str
    retrieved_chunks: List[dict]
    grounding_score: float
    tool_output: dict
    final_response: dict
    prompt_injection_flag: bool

# --- 2. SETUP RETRIEVAL ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("vector_index/faiss.index")
with open("vector_index/chunk_metadata.json", "r") as f:
    chunk_metadata = json.load(f)

# --- 3. PROMPTS & GUARDRAILS ---
SYSTEM_PROMPT = """You are Flipkart's support assistant. 
Follow these rules (Specific, Short, Surround, Single):
- Answer only using the provided sources.
- Keep responses concise.
- Wrap the final output in the requested JSON format.
- Your single job is to output the final answer JSON.
"""

def check_prompt_injection(query: str) -> bool:
    """Guardrail: Block prompt injections."""
    malicious_phrases = ["ignore previous", "ignore all rules", "pretend you are", "system prompt"]
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in malicious_phrases)

# --- 4. NODES ---
def intent_node(state: AgentState):
    """
    Classifies intent using Few-Shot Examples (Mock Implementation).
    Example 1: "Can I return this pair of shoes?" -> policy
    Example 2: "Is order 1234 likely to be returned?" -> return_risk
    """
    query = state["current_query"].lower()
    
    if check_prompt_injection(query):
        return {"prompt_injection_flag": True, "intent": "blocked"}
        
    # Mock LLM Intent Classification
    if "risk" in query or "order" in query:
        intent = "return_risk"
    elif "image" in query or "classify" in query or ".png" in query:
        intent = "image_classification"
    else:
        intent = "policy"
        
    return {"intent": intent, "prompt_injection_flag": False}

def retrieval_node(state: AgentState):
    """RAG Retrieval with Groundedness Guardrail."""
    query = state["current_query"]
    query_embedding = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, 3) # Top 3
    
    retrieved = []
    best_score = float('inf') # FAISS L2 distance: lower is better
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            retrieved.append(chunk_metadata[idx])
            best_score = min(best_score, dist)
            
    # L2 distance threshold for groundedness (approx cosine sim mapping)
    grounding_score = round(float(best_score), 2)
    return {"retrieved_chunks": retrieved, "grounding_score": grounding_score}

def tool_node(state: AgentState):
    """Executes the correct tool based on intent."""
    intent = state.get("intent")
    query = state["current_query"]
    
    if intent == "return_risk":
        # Mock feature extraction from query for offline deterministic run
        # Mock feature extraction from query for offline deterministic run
        features = {
            "price_inr": 1500, "discount_pct": 10.0, "customer_tenure_days": 300,
            "num_previous_orders": 5, "num_previous_returns": 0, "delivery_distance_km": 15.0,
            "delivery_days": 3, "is_weekend_order": 0, "rating_given": 5.0,
            "product_category": "Apparel",
            "payment_method": "Prepaid_Card"
        }
        output = check_return_risk(features)
        
    elif intent == "image_classification":
        # Extract filename if present, else use default test image
        img_path = "data/sample_images/00_tshirt_top.png"
        for word in query.split():
            if ".png" in word:
                img_path = f"data/sample_images/{word}"
        output = classify_product_image(img_path)
    else:
        output = {}
        
    return {"tool_output": output}

def response_node(state: AgentState):
    """Deterministic MOCK_LLM response generation."""
    if state.get("prompt_injection_flag"):
        return {"final_response": {
            "answer": "I can't follow instructions that attempt to override the support-agent rules.",
            "source": "system_guardrail",
            "confidence": 1.0
        }}
        
    intent = state.get("intent")
    
    if intent == "policy":
        score = state.get("grounding_score", 99.0)
        threshold = 1.35 # L2 threshold
        
        if score > threshold: # If distance is too high, it's ungrounded
            return {"final_response": {
                "answer": f"REFUSE. Top retrieved distance: {score}. Grounding threshold: {threshold}. I cannot answer ungrounded policy questions.",
                "source": "policy_kb",
                "confidence": 0.0
            }}
        else:
            chunks = state.get("retrieved_chunks", [])
            answer = " ".join([c["text"] for c in chunks[:1]]) if chunks else "Policy found."
            return {"final_response": {
                "answer": answer,
                "source": "policy_kb",
                "confidence": 0.95
            }}
            
    elif intent == "return_risk":
        out = state.get("tool_output", {})
        return {"final_response": {
            "answer": f"The return probability is {out.get('return_probability')} (Bucket: {out.get('risk_bucket')}).",
            "source": "return_risk_tool",
            "confidence": 0.99
        }}
        
    elif intent == "image_classification":
        out = state.get("tool_output", {})
        return {"final_response": {
            "answer": f"The image is classified as {out.get('predicted_category')}.",
            "source": "image_classifier_tool",
            "confidence": out.get("confidence", 0.0)
        }}

# --- 5. BUILD GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("intent_node", intent_node)
workflow.add_node("retrieval_node", retrieval_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("response_node", response_node)

workflow.set_entry_point("intent_node")

def route_intent(state: AgentState):
    if state.get("prompt_injection_flag"):
        return "response_node"
    if state["intent"] == "policy":
        return "retrieval_node"
    return "tool_node"

workflow.add_conditional_edges(
    "intent_node",
    route_intent,
    {
        "retrieval_node": "retrieval_node",
        "tool_node": "tool_node",
        "response_node": "response_node"
    }
)

workflow.add_edge("retrieval_node", "response_node")
workflow.add_edge("tool_node", "response_node")
workflow.add_edge("response_node", END)

app = workflow.compile()