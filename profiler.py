import json
from outputs import _call_llm # We use the same LLM call function from outputs.py

def create_conversation_profile(full_transcript: str) -> dict:
    """
    Analyzes a full transcript to classify the conversation type, extract key
    entities, and generate a summary, returning it all as a structured dictionary.
    """
    print("--- Creating conversation profile ---")
    
    # This prompt asks the LLM to act as an analyst and return a structured JSON.
    prompt = """
    You are a professional business analyst. Your task is to read the following conversation transcript and provide a structured analysis.
    You MUST respond with ONLY a single, valid JSON object with the following schema:
    - "conversation_type": (String) Classify this conversation. Valid options are only: 'Sales', 'Customer Support', 'Informational Inquiry', 'Interview', 'Internal Meeting', or 'Other'.
    - "key_entities": (Object) Extract the most important named entities. Examples: "customer_name", "product_inquired", "support_ticket_id", "university_program", "company_discussed".
    - "overall_summary": (String) A brief, one-sentence summary of the conversation's purpose and outcome.

    Do not add any conversational text or explanations before or after the JSON object.
    """
    
    # We call the LLM with the prompt and the full transcript 
    response_text = _call_llm(prompt, context=full_transcript)
    
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        
        if json_start != -1 and json_end != -1:
            json_string = response_text[json_start:json_end + 1]
            return json.loads(json_string)
        else:
            raise ValueError("No JSON object found in the LLM response.")

    except Exception as e:
        print(f"Failed to create a valid conversation profile. Error: {e}")

        return {
            "conversation_type": "Unknown",
            "key_entities": {},
            "overall_summary": "Analysis failed."
        }