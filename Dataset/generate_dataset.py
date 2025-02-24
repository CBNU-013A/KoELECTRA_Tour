import os
import re
import time
import json
import numpy as np
import unicodedata
from tqdm import tqdm
import openai
import json

API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=API_KEY)

with open("synthetic.json", "r", encoding="utf-8") as f:
    all_entities = json.load(f)

def generate(prompts, model="gpt-4o", max_tokens=512):
    """
    OpenAI API를 사용
    """
    responses = []
    for prompt in prompts:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_tokens,
            temperature=0.7,
        )
        generated_text = response.choices[0].message.content.strip()
        responses.append(generated_text)
    return responses

def sample_entities(all_entities, min_k=1, max_k=2):
    """
    전체 개체명 데이터셋에서 램덤하게 min_k 이상 max_k 이하의 개체명을 샘플링함
    """
    k = np.random.randint(min_k, max_k+1)
    idxs = np.random.choice(range(len(all_entities)), size=k, replace=False)
    entities = []
    for i in idxs:
        ents = all_entities[i]
        name = np.random.choice(ents["entity_name"])
        entities.append({"class_name": ents["class_name"], "entity_name": name})
    return entities


def construct_sentence_prompt(entities, style="dialog"):
    """
    프롬포트 구성
    """
    prompt = f"Generate a {style} sentence that includes the following entities in Korean.\n\n"
    entities_string = ", ".join([f"{e['entity_name']}({e['class_name']})" for e in entities])
    prompt += f"Entities: {entities_string}\n"
    prompt += "Sentence:"
    return prompt

def clean_sentence(sentence):
    """
    문장 '삭제
    """
    sentence = sentence.strip('"')
    return sentence


num_iterations = 100    # 예시로 2번만
batch_size = 5        # 한번에 3개씩

generated_sentences = []

for _ in tqdm(range(num_iterations)):
    batch_entities = [sample_entities(all_entities) for _ in range(batch_size)]
    batch_prompts = [construct_sentence_prompt(ents) for ents in batch_entities]
    batch_generated = generate(batch_prompts, model="gpt-4o", max_tokens=256)
    for generated, entities in zip(batch_generated, batch_entities):
        cleaned_sentence = clean_sentence(generated)
        generated_sentences.append({"entities": entities, "sentence": cleaned_sentence})
    time.sleep(1)  # rate limit 방지를 위한 딜레이


with open("ner_results.json", "w", encoding="utf-8") as f:
    json.dump(generated_sentences, f, ensure_ascii=False, indent=4)
#print(json.dumps(data, indent=4, ensure_ascii=False))