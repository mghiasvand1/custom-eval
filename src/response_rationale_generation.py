from tqdm import tqdm
from groq import Groq
import pandas as pd
import random
import json
import ast


api_key = ""
client = Groq(api_key=api_key)

def get_model_response(prompt):
    user_message = {
        "role": "user",
        "content": prompt
    }
    try:
        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[user_message],
            temperature=0,
            seed=42,
            max_tokens=4096,
        )
        response = completion.choices[0].message.content
        start_idx = str(response).find('{')
        end_idx = str(response).rfind('}') + 1
        response = ast.literal_eval(response[start_idx:end_idx])
        return response
    except:
        return None


areas = ["MENTALCHAT", "DEPTHQA"]
domains = ["psychology", "mathematics"]

area = areas[0]
domain = domains[0]
curr_inputs = []
new_inputs, outputs = [], []

with open("data/synthetic/instructions.json", "r", encoding="utf-8") as f:
    instructions_json = json.load(f)

with open("data/synthetic/criteria.json", "r", encoding="utf-8") as f:
    criteria_json = json.load(f)

_type = "train"

instructions = instructions_json.get(area, {}).get(_type, [])
criteria_train = criteria_json.get(area, {}).get("train", [])
criteria_test = criteria_json.get(area, {}).get("test", [])
criteria_list = criteria_json.get(area, {}).get(_type, [])

criteria_list = [
    f"{c['name']}: {c['description']}" for c in criteria_list
]
criteria_train = [
    f"{c['name']}: {c['description']}" for c in criteria_train
]
criteria_test = [
    f"{c['name']}: {c['description']}" for c in criteria_test
]
whole_criteria = criteria_train + criteria_test


with open("src/prompt_templates.json", "r", encoding="utf-8") as f:
    prompt_templates = json.load(f)

prompt_template = prompt_templates.get("RESPONSE_RATIONALE_PROMPT", "")
fine_tune_template = prompt_templates.get("FINE_TUNE_PROMPT", "")

for instruction in instructions:
    for criterion in criteria_list:
        temp_prompt = prompt_template.replace("|$DOMAIN$|", domain)
        temp_prompt = temp_prompt.replace("|$CRITERIA$|", criterion)
        temp_prompt = temp_prompt.replace(
            "|$CRITERIA_1$|", f"|{criterion.lower().replace('.', '')}|")
        temp_prompt = temp_prompt.replace(
            "|$CRITERIA_2$|", f"|{criterion.lower().split(': ')[0]}|")
        temp_prompt = temp_prompt.replace("|$INPUT$|", instruction)

        curr_inputs.append(temp_prompt)

        fine_tune_prompt = fine_tune_template.replace("|$DOMAIN$|", domain)
        fine_tune_prompt = fine_tune_prompt.replace("|$CRITERIA$|", criterion)
        fine_tune_prompt = fine_tune_prompt.replace("|$INPUT$|", instruction)

        new_inputs.extend([fine_tune_prompt] * 5)

inputs = []

for _input in tqdm(curr_inputs):
    response = get_model_response(_input)

    score_outputs = []
    answer_outputs = []
    if response:
        for i in range(1, 6):
            rationale = response.get(f"score_{i}", {}).get(
                "rationale", "No rationale provided")
            answer = response.get(f"score_{i}", {}).get(
                "answer", "No answer provided")
            answer_outputs.append(answer)
            if _type == "train":
                score_outputs.append(
                    f"{rationale}\n\nThe final score is {i} out of 5.")
            elif _type == "test":
                score_outputs.append(f"{i}")

        outputs.extend(score_outputs)

        for idx in range(5):
            if idx < len(new_inputs):
                new_inputs[idx] = new_inputs[idx].replace(
                    "|$ANSWER$|", answer_outputs[idx])

        inputs.extend(new_inputs[:5])
    new_inputs = new_inputs[5:]

df = pd.DataFrame({"input": inputs, "output": outputs})
df.to_csv(f"data/synthetic/{_type}_data_{area}.csv", index=False)
