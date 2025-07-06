import re
import matplotlib.pyplot as plt
from openai import OpenAI 
#from together import Together
from datasets import load_dataset
from dotenv import load_dotenv



# --- 1. Load Benchmark ---
def load_benchmark_data(benchmark_name="gsm8k", split="test", num_samples=1):
    """
    Loads a specified number of samples from a benchmark.
    For gsm8k, 'main' config is used.
    """
    try:
        if benchmark_name == "gsm8k": # change name of benchmark
            dataset = load_dataset(benchmark_name, "main", split=split, streaming=True)
        elif benchmark_name == "math_dataset": # Example for MATH dataset (might need specific config)
            dataset = load_dataset("competition_math", split=split, streaming=True) # Check actual name/config for MATH
        else:
            # Fallback or specific loader for FrontierMath if available
            # For now, this is a placeholder
            print(f"Attempting to load {benchmark_name}. This might require specific handling.")
            dataset = load_dataset(benchmark_name, split=split, streaming=True)

        samples = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            samples.append(example)
        if not samples:
            raise ValueError(f"No samples loaded. Check benchmark name '{benchmark_name}', split '{split}', or num_samples.")
        print(f"Loaded {len(samples)} samples from {benchmark_name} ({split} split).")
        return samples
    except Exception as e:
        print(f"Error loading benchmark {benchmark_name}: {e}")
        print("Please ensure the dataset is available on Hugging Face Hub or provide a custom loader.")
        print("Example: For GSM8K, use benchmark_name='gsm8k', config_name='main'.")
        return []


def extract_final_answer_gsm8k_bm(text): #change name of function "extract_final_answer_NAME_bm"
    """
    Extracts the final numerical answer from a string for GSM8K.
    GSM8K answers are typically the final number in the 'answer' field, often after "#### ".
    For LLM output, we look for a number, possibly in \boxed{}.
    """
    # Look for \boxed{answer}
    match_boxed = re.search(r"\\boxed\{([\d\.\,\s]+)\}", text)
    if match_boxed:
        return match_boxed.group(1).replace(",", "").strip()

    # Look for numbers, prioritizing those at the end or after specific markers
    # This regex finds sequences of digits, possibly with commas or a decimal point.
    # It tries to find the last such number.
    numbers = re.findall(r"[\d\.\,]+", text)
    if numbers:
        # Try to return the last number found, cleaning it up
        potential_answer = numbers[-1].replace(",", "").strip()
        # Basic validation if it's a number
        try:
            float(potential_answer)
            return potential_answer
        except ValueError:
            pass # Not a simple number

    # Fallback if no clear number is found
    print(f"Warning: Could not extract a clear numerical answer from: '{text}'")
    return None 
