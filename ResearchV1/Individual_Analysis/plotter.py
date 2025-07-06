import pandas as pd
import matplotlib.pyplot as plt
import itertools

def sample_rate_accuracy(ai_files_dict):
    """
    ai_files_dict: dict
        A dictionary mapping ai model names to CSV filenames.
        Example:
        {
            'gpt_4o': 'gpt_4o_gsm8k.csv',
            'together': 'together_gsm8k.csv',
            'claude': 'claude_gsm8k.csv'
        }
    """

    sample_sizes = [1] + list(range(10, 501, 10))
    results = []

    # Generate distinct markers for each model
    markers = itertools.cycle(['o', 's', '^', 'D', '*', 'v', 'x', 'p', 'h', '+'])

    for ai_name, filename in ai_files_dict.items():
        df = pd.read_csv(filename)

        for size in sample_sizes:
            sample = df.sample(n=size, random_state=42)
            accuracy = sample['correct'].mean()
            results.append({'accuracy': accuracy, 'sample_size': size, 'ai': ai_name})

    data = pd.DataFrame(results)

    # Plotting
    plt.figure(figsize=(10, 6))

    for ai_name, marker in zip(ai_files_dict.keys(), markers):
        model_data = data[data['ai'] == ai_name]
        plt.plot(model_data['sample_size'], model_data['accuracy'] * 100,
                 marker=marker, label=ai_name.upper())

    plt.xlim(0, 500)
    plt.ylim(0, 105)
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Sample Size by AI Model', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()



dico = {
        'gpt_4o': 'gpt_4o_gsm8k.csv',
        'together': 'together_gsm8k.csv',
        }

sample_rate_accuracy(dico)
