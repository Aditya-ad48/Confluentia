from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer
import chromadb
from outputs import _call_llm 
import json
from config import EMBED_MODEL_NAME, CHROMA_DB_PATH, COLLECTION_NAME


model = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Context-Aware Router Function
def choose_tool(query: str, available_tools: List[Dict], conversation_profile: Dict) -> str:
    """
    Uses an LLM to act as a router, selecting the best tool for a given user query
    by considering the overall conversation profile.
    """
    tool_descriptions = "\n".join(
        [f"- Tool Name: {tool['name']}\n  Description: {tool['description']}" for tool in available_tools]
    )
    
    # The conversation profile is now included in the prompt for better decision
    profile_str = json.dumps(conversation_profile, indent=2)
    prompt = f"""
    You are an intelligent routing agent. Your task is to select the single best tool to answer the user's query.
    You MUST consider the overall Conversation Profile to make an informed decision. For example, sentiment analysis is more relevant for a 'Customer Support' call than an 'Informational Inquiry'.

    Conversation Profile:
    {profile_str}

    Available Tools:
    {tool_descriptions}

    User Query: "{query}"

    Respond with ONLY the name of the tool. Do not add any explanation or conversation.
    """
    
    # Using LLM to decide which tool to use
    chosen_tool_name = _call_llm(prompt, context="") 
    
    # Cleaning the response to ensure we only get the tool name
    for tool in available_tools:
        if tool['name'] in chosen_tool_name:
            print(f"Router selected tool: {tool['name']} based on profile: {conversation_profile.get('conversation_type')}")
            return tool['name']
            
    # If the LLM fails to choose a valid tool default to a text summary
    print("Router defaulted to: summarize_text")
    return "summarize_text"

def query_chroma(query: str, top_k: int = 5, metadata_filters: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Queries the ChromaDB collection for relevant text segments.
    """
    q_emb = model.encode([query]).tolist()

    query_args = {
        'query_embeddings': q_emb,
        'n_results': top_k
    }
    if metadata_filters:
        query_args['where'] = metadata_filters

    res = collection.query(**query_args)

    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    distances = res.get("distances", [[]])[0]

    return {"documents": docs, "metadatas": metadatas, "ids": ids, "distances": distances}