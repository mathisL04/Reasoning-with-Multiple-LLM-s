import pandas as pd
import matplotlib.pyplot as plt
from benchmark import extract_final_answer_gsm8k_bm
from api import call_openai_api, extract_final_answer_gsm8k_ai
#from api import call_together_ai_api, extract_final_answer_gsm8k_ai
from datetime import datetime



def ai_benchmark_tester(question):

    bm_question = question['question']
    bm_answer = question['answer']
    # get the ith question and answer of the sample

    prompt = [
    {"role": "system", "content": "You are a helpful assistant. You will be tasked to solve a math problem. When you think you have the final answer, state it clearly, for example, by enclosing it in \\boxed{{number}}. Do not add the unit value in the answer, just the number"},
    {"role": "user", "content": bm_question}]
    # create the prompt for ai
    
    ai_answer = call_openai_api(model_name = "gpt-4o", messages = prompt, max_tokens_turn = 1028)
    # call the ai api
      #ai_answer = call_together_ai_api(
        #model_name="Qwen/QwQ-32B",
        #messages=prompt,
        #max_tokens_turn=1028,
        #temperature=0.7

    clean_ai_answer = extract_final_answer_gsm8k_ai(ai_answer)
    clean_bm_answer = extract_final_answer_gsm8k_bm(bm_answer)
    # properly format the ai and benchmark answers

    try:
        new_row = pd.DataFrame([{
        'date': datetime.now(),
        'question': bm_question,
        'ai_answer': ai_answer,
        'ai_answer_clean': float(clean_ai_answer),
        'bm_answer': bm_answer,
        'bm_answer_clean': float(clean_bm_answer),
        'correct': float(clean_bm_answer) == float(clean_ai_answer)
        }])
    except:
        print('######## ai: ' + str(ai_answer))
        print('######## bm: ' + str(bm_answer) )
        print('######### quest: ' + str(bm_question))

    return new_row

