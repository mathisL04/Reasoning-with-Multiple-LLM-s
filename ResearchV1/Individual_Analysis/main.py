import pandas as pd
import logging
import multiprocessing
from datetime import datetime
from benchmark import load_benchmark_data
from ai_benchmark_tester import ai_benchmark_tester 
import random


# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
)

# --- Worker function ---
def worker(args):
    idx, question = args
    try:
        logging.info(f"Processing question {idx}")
        # Create a minimal single-row dataframe with the result
        result_df = ai_benchmark_tester(question=question)
        return result_df
    except Exception as e:
        logging.error(f"Error on question {idx}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    benchmark_name = "gsm8k" # change with benchmark name
    num_samples = 500        # Total size of sample to test on (n = 200 recommended for satturated benchmarks)

    logging.info("Loading benchmark questions...")
    questions = load_benchmark_data(benchmark_name=benchmark_name, split="test", num_samples=1319)

    random.seed(42)
    selected_questions = random.sample(questions, num_samples) # list of dic: question & answer

    args = [(i, q) for i, q in enumerate(selected_questions)]

    logging.info("Starting multiprocessing pool...")
    with multiprocessing.Pool(processes=6) as pool:
        results = pool.map(worker, args)

    logging.info("Combining results into dataframe...")
    all_results = pd.concat([r for r in results if r is not None], ignore_index=True)

    logging.info("Saving to CSV...")
    all_results.to_csv("mydata.csv", index=False)

    logging.info("Done.")
