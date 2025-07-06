
import re
import matplotlib.pyplot as plt
from openai import OpenAI
#from together import Together
from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd
import multiprocessing

# --- Configuration ---
OPENAI_API_KEY = "xxxxxxx"  #add API key
TOGETHER_API_KEY = "yyyyyy" #add API key

# Initialize API clients

openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("Warning: a key not found")

# TOGETHER AI 
#TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
#together.api_key = TOGETHER_API_KEY
#together_client = Together()


def call_openai_api(model_name, messages, max_tokens_turn):
    """
    Calls the OpenAI API.
    """
    if not openai_client:
        return "Error: OpenAI client not initialized. Please set OPENAI_API_KEY."
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens_turn
            #temperature=temperature (uncomment temperature for Together Qwen/QwQ-32B)
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

#def call_together_ai_api(model_name, messages, max_tokens_turn=3000, temperature=0.7):



def extract_final_answer_gsm8k_ai(text):
    """
    Extracts the final numerical answer from a GSM8K-style string, strictly from \boxed{}.
    If the content inside \boxed{} contains letters (like '18th'), they are removed.
    Only digits, periods, and commas are retained before final cleanup.
    """
    match_boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if match_boxed:
        raw = match_boxed.group(1)
        # Keep only digits, dots, and commas
        cleaned = re.sub(r"[^\d\.,]", "", raw)
        cleaned = cleaned.replace(",", "").strip()
        try:
            float(cleaned)
            return cleaned
        except ValueError:
            pass

    print(f"Warning: Could not extract a valid numerical answer from boxed content in: '{text}'")
    return None
