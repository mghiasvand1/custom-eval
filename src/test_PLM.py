from sklearn.metrics import cohen_kappa_score
from groq import Groq
from tqdm import tqdm
import pandas as pd
import ast


api_key = ""
client = Groq(api_key=api_key)
areas = ["MENTALCHAT", "DEPTHQA"]
area = areas[1]
DATA_PATH = f"data/synthetic/test_data_{area}.csv"
KEYWORDS = {
    "MENTALCHAT": {"ood": [":\nemotional supportiveness", ":\npracticality of advicey"], "id": [":\nclarity and comprehensibility", ":\nbalance between validation and encouragement"]},
    "DEPTHQA": {"ood": [":\nstep-by-step explanation", ":\nterminology accuracy"], "id": [":\nuse of definitions", ":\nclarity"]},
}


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
            max_tokens=2048,
        )
        response = completion.choices[0].message.content
        start_idx = str(response).find('{')
        end_idx = str(response).rfind('}') + 1
        response = ast.literal_eval(response[start_idx:end_idx])
        return response["score"]
    except:
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

pattern_ood = "|".join(KEYWORDS[area]["ood"])
df1 = df[df["input"].str.lower().str.contains(pattern_ood, na=False)]
qwk_1 = cohen_kappa_score(
    df1["gold_score"], df1["llm_score"], weights="quadratic") if not df1.empty else None

pattern_id = "|".join(KEYWORDS[area]["id"])
df2 = df[df["input"].str.lower().str.contains(pattern_id, na=False)]
qwk_2 = cohen_kappa_score(
    df2["gold_score"], df2["llm_score"], weights="quadratic") if not df2.empty else None

qwk_3 = cohen_kappa_score(
    df["gold_score"], df["llm_score"], weights="quadratic")

print("Out-of-Domain QWK:", qwk_1)
print("In-Domain QWK:", qwk_2)
print("Overall QWK:", qwk_3)
