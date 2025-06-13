import os
import re
from openai import OpenAI
from together import Together
import together
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize API clients
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

if TOGETHER_API_KEY:
    together.api_key = TOGETHER_API_KEY
else:
    print("Warning: TOGETHER_API_KEY not found. Together.ai models will not be available.")

together_client = Together()
# --- 1. Load Benchmark ---
def load_benchmark_data(benchmark_name="gsm8k", split="test", num_samples=1):
    """
    Loads a specified number of samples from a benchmark.
    For gsm8k, 'main' config is used.
    """
    try:
        if benchmark_name == "gsm8k":
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

# --- 2. Implement API Call Functions ---
def call_openai_api(model_name, messages, max_tokens_turn, model_type="reasoning"):
    """
    Calls the OpenAI API.
    """
    if not openai_client:
        return "Error: OpenAI client not initialized. Please set OPENAI_API_KEY."
    try:
        if model_type=="reasoning":
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                # reasoning={"effort": "low"},
                max_completion_tokens=max_tokens_turn,
                # max_tokens=max_tokens_turn,
                # temperature=0.7, # Adjust temperature as needed
            )
            return response.choices[0].message.content.strip()
        else:

            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens_turn,
                temperature=0.7, # Adjust temperature as needed
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

def call_together_ai_api(model_name, messages, max_tokens_turn):
    """
    Calls the Together.ai API.
    Formats messages for Together AI if needed (often similar to OpenAI).
    """
    if not TOGETHER_API_KEY:
        return "Error: Together.ai API key not set."
    try:
        # Together.ai's Chat completion expects a list of messages similar to OpenAI
        response = together_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens_turn,
            temperature=0.7, # Adjust temperature as needed
        )
        # Accessing the response content might differ slightly based on the exact SDK version
        # This is a common way, check `response` object structure if it fails.
        if response is not None:
            # if response and response.get('choices'):
            return response.choices[0].message.content.strip()
    
        else:
            return f"Error: Unexpected response structure from Together.ai: {response}"

    except Exception as e:
        return f"Error calling Together.ai API: {e}"

# --- 3. Implement Main Chat Logic and Evaluation ---

def extract_final_answer_gsm8k(text):
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
    return None # Or some indicator that extraction failed


def parse_ground_truth_answer_gsm8k(answer_str):
    """
    Parses the ground truth answer for GSM8K.
    The format is usually "The final answer is \n#### <number>\n"
    """
    match = re.search(r"####\s*([\d\.\,]+)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_str.strip() # Fallback if "####" is not present


def run_collaborative_solving(
    model1_config,
    model2_config,
    benchmark_samples,
    num_chat_turns=3,
    max_tokens_per_turn=512,
    ):
    """
    Runs the collaborative solving process.

    model_config: {"name": "model_identifier", "api": "openai" or "together"}
    """
    correct_answers = 0
    total_samples = len(benchmark_samples)
    results = []

    for i, sample in enumerate(benchmark_samples):
        print(f"\n--- Solving Sample {i+1}/{total_samples} ---")
        problem_statement = sample.get('question', sample.get('problem')) # Adapt based on benchmark
        if not problem_statement:
            print(f"Warning: Could not find problem statement in sample: {sample}")
            continue

        ground_truth_answer_raw = sample.get('answer', sample.get('solution')) # Adapt
        if not ground_truth_answer_raw:
            print(f"Warning: Could not find ground truth answer in sample: {sample}")
            continue

        ground_truth_answer = parse_ground_truth_answer_gsm8k(str(ground_truth_answer_raw))
        print(f"Problem: {problem_statement}")
        print(f"Ground Truth (parsed): {ground_truth_answer}")

        # Initialize models and conversation history
        models = [model1_config, model2_config]
        model_apis = {
            "openai": call_openai_api,
            "together": call_together_ai_api,
        }

        # Initial prompt for collaboration
        # The initial prompt is given to the first model.
        # It includes the problem and the instruction to collaborate.
        initial_prompt_text = (
            f"""You are a helpful AI assistant. You will collaborate with another AI assistant
            to solve the following math problem. Please show your reasoning step by step.
            Take turns to contribute to the solution. When you think you have the final answer, 
            state it clearly, for example, by enclosing it in \\boxed{{number}}.\n\n
            Problem: {problem_statement}\n\nLet's start. What is your first step or thought?"""
        )

        conversation_history = [{"role": "user", "content": initial_prompt_text}]
        current_solution_text = ""

        # Chat loop
        for turn in range(num_chat_turns * 2): # Each model gets `num_chat_turns`
            current_model_idx = turn % 2
            current_model_config = models[current_model_idx]
            model_api_func = model_apis[current_model_config["api"]]

            print(f"\nTurn {turn // 2 + 1} - Model {current_model_idx + 1} ({current_model_config['api']}: {current_model_config['name']}) thinking...")

            # Add instruction for the current model
            if turn > 0: # For subsequent turns, prompt refers to previous model's output
                # The 'user' message here simulates the instruction to the current LLM
                # The actual content of the "chat" is already in conversation_history
                prompt_for_current_turn = (
                    f"""The other model said: '{conversation_history[-1]['content']}'.
                    Please continue the collaboration. Show your reasoning. If you are refining,
                    be clear about what you are changing or adding. If you think you have the final answer,
                    state it clearly, for example, by enclosing it in \\boxed{{number}}."""
                )
                # We add this as a "user" message to guide the assistant, but the history already contains the flow.
                # For some APIs, the last message must be 'user'.
                # Let's structure it so history is built correctly for the API call.
                # The API will get the full history. The prompt above is more for "meta-guidance".

            # Call the API
            # The history contains the problem, and all previous turns.
            model_response = model_api_func(
                current_model_config["name"],
                conversation_history, # Pass the entire conversation history
                max_tokens_per_turn
            )

            if "Error:" in model_response:
                print(f"API Error: {model_response}")
                current_solution_text += f"\nModel {current_model_idx + 1} Error: {model_response}"
                break # Stop this sample on API error

            print(f"Model {current_model_idx + 1} response: {model_response}")

            # Add model's response to history for the next turn
            # The role should be 'assistant' for the model's actual output
            conversation_history.append({"role": "assistant", "content": model_response})

            # The next 'user' message (if any more turns) will be the implicit prompt to continue.
            # Or, if we want to be explicit:
            if turn < (num_chat_turns * 2) -1:
                 conversation_history.append({"role": "user", "content": "What is your next step or thought based on the above?"})


            current_solution_text += f"\nModel {current_model_idx + 1} ({current_model_config['name']}): {model_response}"

            # Optional: Check for early stopping if a model confidently gives a final answer
            # This is complex, as "confidence" is hard to gauge.
            # For now, let it run all turns.

        # After all turns, extract the final answer from the last model's response or the whole conversation
        # We'll take the last assistant's response.
        final_llm_output = ""
        for msg in reversed(conversation_history):
            if msg["role"] == "assistant":
                final_llm_output = msg["content"]
                break

        extracted_answer = extract_final_answer_gsm8k(final_llm_output)
        print(f"Final collaborative text: ... {final_llm_output[-300:]}") # Print last part
        print(f"Extracted Answer: {extracted_answer}")
        print(f"Ground Truth Answer: {ground_truth_answer}")

        is_correct = False
        if extracted_answer is not None and ground_truth_answer is not None:
            # Comparing numbers, might need tolerance for floats
            try:
                if abs(float(extracted_answer) - float(ground_truth_answer)) < 1e-3: # Tolerance for float comparison
                    is_correct = True
            except ValueError: # If conversion to float fails
                if extracted_answer.strip() == ground_truth_answer.strip():
                    is_correct = True

        if is_correct:
            correct_answers += 1
            print("Result: CORRECT")
        else:
            print("Result: INCORRECT")

        results.append({
            "problem": problem_statement,
            "ground_truth": ground_truth_answer,
            "collaborative_solution": current_solution_text,
            "final_llm_output": final_llm_output,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
        })

    performance = (correct_answers / total_samples) * 100 if total_samples > 0 else 0
    print(f"\n--- Benchmark Performance ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Accuracy: {performance:.2f}%")

    return performance, results

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration for Models ---
    # Together.ai models: https://docs.together.ai/docs/inference-models
    # OpenAI models: https://platform.openai.com/docs/models

    # Example 1: OpenAI (GPT-4o mini) + Together.ai (Mixtral)
    model1_config = {"name": "o3-mini-2025-01-31", "api": "openai"}
    model2_config = {"name": "Qwen/QwQ-32B", "api": "together"}

    # Example 2: Two OpenAI models
    # model1_config = {"name": "gpt-4o-mini", "api": "openai"}
    # model2_config = {"name": "gpt-3.5-turbo", "api": "openai"}

    # Example 3: Two Together.ai models
    # model1_config = {"name": "mistralai/Mixtral-8x7B-Instruct-v0.1", "api": "together"}
    # model2_config = {"name": "Qwen/Qwen1.5-72B-Chat", "api": "together"} # Example, choose a suitable one


    # --- Benchmark and Run Parameters ---
    # Using gsm8k as FrontierMath might not be directly available via `load_dataset` without specifics
    # If you have FrontierMath files, you'll need a custom function for `load_benchmark_data`
    benchmark_name_to_run = "gsm8k" # or "math_dataset", or your custom "FrontierMath"
    num_benchmark_samples = 1      # Number of problems to solve
    chat_turns_per_model = 2       # Each model gets to "speak" this many times
    tokens_per_turn = 1028          # Max tokens for each model's response in a turn

    if not OPENAI_API_KEY and (model1_config["api"] == "openai" or model2_config["api"] == "openai"):
        print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable to use OpenAI models.")
    elif not TOGETHER_API_KEY and (model1_config["api"] == "together" or model2_config["api"] == "together"):
        print("Together API key not found. Please set the TOGETHER_API_KEY environment variable to use Together.ai models.")
    else:
        print(f"Running collaborative solving on {benchmark_name_to_run}...")
        print(f"Model 1: {model1_config['api']} - {model1_config['name']}")
        print(f"Model 2: {model2_config['api']} - {model2_config['name']}")
        print(f"Number of samples: {num_benchmark_samples}, Chat turns per model: {chat_turns_per_model}, Tokens per turn: {tokens_per_turn}")

        benchmark_data = load_benchmark_data(
            benchmark_name=benchmark_name_to_run,
            num_samples=num_benchmark_samples
        )

        if benchmark_data:
            performance, detailed_results = run_collaborative_solving(
                model1_config,
                model2_config,
                benchmark_data,
                num_chat_turns=chat_turns_per_model,
                max_tokens_per_turn=tokens_per_turn,
            )
            # You can save `detailed_results` to a file if needed
            # for res in detailed_results:
            # print(f"Problem: {res['problem']}\nExtracted: {res['extracted_answer']}, GT: {res['ground_truth']}, Correct: {res['is_correct']}\n---\n")
        else:
            print("Could not load benchmark data. Exiting.")