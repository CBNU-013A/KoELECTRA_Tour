import openai
import json
import os
from tqdm import tqdm
import time
import re

API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=API_KEY)

def generate(prompts, model="gpt-4o", max_tokens=512):
    """
    OpenAI API를 사용
    """
    dev_msg = (
        "You are a helpful assistant.\n"
        "Next is th example of the BIO tagging for the given sentence in Korean.\n"
        "Sentence: '이 카페는 공간이 작지만, 가족친화적인 분위기가 정말 좋아요'\n"
        "Example of the BIO tagging:\n"
        "이/O 카페는/O 공간이/B-ASP 작지만,/B-OPI 가족친화적인/B-ASP 분위기가/B-ASP 정말/I-OPI 좋아요/B-OPI\n"
        "Please keep the output format as 'O O B-ASP B-OPI O B-ASP I-OPI B-OPI'.\n"
        "DO NOT INCLUDE OTHER COMMNENTS IN THE OUTPUT."
        )
    responses = []
    for prompt in prompts:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": dev_msg},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_tokens,
            temperature=0.7,
        )
        generated_text = response.choices[0].message.content.strip()
        responses.append(generated_text)
    return responses

def construct_bio_prompt(text, entities, tokens):
    """
    문장과 엔티티 정보를 바탕으로 BIO 태깅을 요청하는 프롬프트를 생성합니다.
    Aspect와 Opinion term을 모두 태깅합니다.
    """
    prompt = f"다음 문장을 단어단위로 토큰으로 나눴음. 각 토큰에 대해 BIO 태깅을 해줘.\n"
    prompt += f"문장: {text}\n"
    prompt += "토큰: " + " ".join(tokens) + "\n\n"
    prompt += "BIO 스킴 규칙:\n"
    prompt += "- B-ASP: Aspect term의 시작 부분\n"
    prompt += "- I-ASP: Aspect term의 내부 부분\n"
    prompt += "- B-OPI: Opinion term의 시작 부분\n"
    prompt += "- I-OPI: Opinion term의 내부 부분\n"
    prompt += "- O: Aspect와 Opinion에 해당하지 않는 부분\n\n"
    prompt += "Aspect는 문장에서 어떤 대상에 대한 속성을 나타내는 단어이며, Opinion은 해당 속성에 대한 주관적인 평가를 나타내는 단어입니다.\n"
    prompt += "이 문장에 포함된 엔티티:\n"
    for ent in entities:
        prompt += f"- {ent['entity_name']} ({ent['class_name']})\n"
    prompt += "문장에 해당 엔티티는 없지만, Aspect와 Opinion term으로 태깅할 단어도 포함될 수 있습니다.\n"
    prompt += "태깅 결과가 다음과 같을 때\n"
    prompt += "예시 - '이 카페는 공간이 작지만, 가족친화적인 분위기가 정말 좋아요' / 이/O 카페는/O 공간이/B-ASP 작지만/B-OPI ,/O 가족친화적인/B-ASP 분위기가/B-ASP 정말/I-OPI 좋아요/B-OPI\n"
    prompt += "\n출력 형식은 다음과 같아야 합니다:\n"
    prompt += "O O B-ASP B-OPI O B-ASP I-OPI B-OPI\n"
    prompt += "위 형식에 맞추어 BIO 태깅 결과를 작성해주세요."
    return prompt

def tokenize_sentence(sentence):
    """
    문장을 단어로 토큰화합니다.
    """
    tokens = re.findall(r'\w+|[^\w\s]', sentence)
    return tokens

with open("ner_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

bio_data = []

batch_size = 5

for item in tqdm(range(0, len(data), batch_size)):
    batch = data[item:item+batch_size]
    batch_prompts = []
    batch_tokens = []

    for i in batch:
        tokens = tokenize_sentence(i["sentence"])
        batch_tokens.append(tokens)

        prompt = construct_bio_prompt(i["sentence"], i["entities"], tokens)
        batch_prompts.append(prompt)

    batch_results = generate(batch_prompts, model="gpt-4o", max_tokens=1024)

    for i, tokens, bio_result in zip(batch, batch_tokens, batch_results):
        bio_data.append({
            "sentence": i["sentence"],
            "tokens": tokens,
            "bio_tagging": bio_result
        })
    time.sleep(1)

with open("bio_tagged_data.json", "w", encoding="utf-8") as f:
    json.dump(bio_data, f, indent=4, ensure_ascii=False)