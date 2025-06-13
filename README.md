
# LLM Collaborative Solver

This project provides a Python script to orchestrate a "conversation" between two Language Models (LLMs), enabling them to collaboratively solve reasoning problems. The framework is designed to use models from different providers (e.g., OpenAI and Together.ai) and evaluates their combined performance on benchmarks like GSM8K.

## Features

-   **Multi-Provider Support:** Seamlessly integrates with both OpenAI and Together.ai APIs.
-   **Collaborative Chat:** Models take turns building on each other's responses to solve complex problems.
-   **Benchmark Evaluation:** Loads problems from Hugging Face Datasets (e.g., `gsm8k`) and calculates the accuracy of the final answers.
-   **Configurable:** Easily change the models, number of problems, chat length, and tokens per turn directly in the script.
-   **Secure:** Uses a `.env` file to manage API keys securely, keeping them out of your source code.

## Project Structure

```
.
├── main.py  # The main script
├── requirements.txt         # Project dependencies
├── .env                     # API keys (local, not committed to git)
└── README.md                # This file
```

## Setup Instructions

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate the environment
# On macOS or Linux:
source venv/bin/activate

# On Windows (Command Prompt or PowerShell):
.\venv\Scripts\activate
```

Once activated, your terminal prompt will be prefixed with `(venv)`.

### 3. Install Dependencies

The required Python packages are listed in `requirements.txt`. Install them using pip.

```bash
pip install -r requirements.txt
```
*(If you don't have a `requirements.txt` file yet, create one after installing the packages with the command: `pip freeze > requirements.txt`)*

### 4. Set Up API Keys

This project requires API keys for the LLM providers you intend to use.

1.  Create a file named `.env` in the root of the project directory.
2.  Add your API keys to this file. The script will load them automatically.

**`.env` file template:**

```env
OPENAI_API_KEY="sk-..."
TOGETHER_API_KEY="..."
```

**Important:** Add `.env` to your `.gitignore` file to ensure you never accidentally commit your secret keys to the repository.

## Running the Code

### 1. Configure the Run

Open the `main.py` file and scroll to the bottom `if __name__ == "__main__":` block. Here you can configure the run parameters:

-   **`model1_config` & `model2_config`**: Define which models to use and from which API.
-   **`benchmark_name_to_run`**: Specify the Hugging Face benchmark (e.g., "gsm8k").
-   **`num_benchmark_samples`**: Set the number of problems to test.
-   **`chat_turns_per_model`**: Define how many times each model gets to speak.

**Example Configuration:**

Check main.py for an example configuration

Then run the script


```bash
python main.py
```
