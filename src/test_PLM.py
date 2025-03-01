from sklearn.metrics import cohen_kappa_score
from groq import Groq
from tqdm import tqdm
import pandas as pd
import json


api_key = ""
client = Groq(api_key=api_key)
areas = ["MENTALCHAT", "DEPTHQA"]
area = areas[1]
DATA_PATH = f"data/synthetic/test_data_{area}.csv"

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
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        response = json.loads(completion.choices[0].message.content)
        return response.get("score")
    except (json.JSONDecodeError, AttributeError):
        return None

df = pd.read_csv(DATA_PATH)

df["gold_score"] = df["output"].astype(int)

llm_scores = []
for prompt in tqdm(df["input"]):
    prompt = prompt.replace("""Solely consider the provided criteria, and return a rationale followed by a score within the five-point Likert scale for evaluating the answer to the given input.""",
                            """Solely consider the provided criteria and return a rationale followed by a score on a five-point Likert scale for evaluating the answer to the given input in the following JSON format, without any additional feedback: {"rationale": "", "score": ""}""")
    llm_response = get_model_response(prompt)
    try:
        llm_scores.append(int(llm_response))
    except (ValueError, TypeError):
        llm_scores.append(None)

df["llm_score"] = llm_scores
df = df.dropna(subset=["llm_score"]).astype({"llm_score": "int"})

qwk = cohen_kappa_score(df["gold_score"], df["llm_score"], weights="quadratic")

print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
