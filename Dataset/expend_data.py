import os
import json
import openai
from tqdm import tqdm
import unicodedata
import re

# OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def generate(prompts, model="gpt-4o", max_tokens=1024):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompts}
            ],
            temperature=0.7,
            max_completion_tokens=max_tokens,
        )
        texts = [choice.message.content.strip() for choice in response.choices]
        return texts
    except Exception as e:
        print(f"OpenAI API 호출 중 오류 발생: {e}")
        return []

def construct_entity_prompt(class_name, entity_names, k=100):
    prompt = f"Below is a list of <{class_name}> entity names in Korean. Please list exactly {k} new <{class_name}> entity names in Korean that are similar.\n\n"
    prompt += "Existing entity names:\n"
    for e in entity_names:
        prompt += f"- {e}\n"
    prompt += "\nNew entity names:\n"
    return prompt

def clean_text(text):
    """문자열에서 유니코드 특수 문자 및 불필요한 공백 제거"""
    return unicodedata.normalize("NFKC", text).encode("utf-8", "ignore").decode("utf-8").strip()

def postprocess_entities(synthetic_entities):
    """
    API 응답에서 개체명만 추출합니다.
    숫자로 시작하는 항목(예: "1. ..." 또는 "1) ...")만 추출하고,
    그렇지 않은 줄은 무시합니다.
    """
    processed = []
    for ents in synthetic_entities:
        # 응답 전체에서 줄 단위로 분할
        lines = ents.split("\n")
        new_entities = []
        for line in lines:
            line = line.strip()
            # 숫자로 시작하는 항목만 처리 (예: "1. 서대문" 또는 "2) 경복궁")
            if re.match(r'^\d+[\.\)]', line):
                # 숫자와 구분 기호 제거
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = line.replace("-", "").strip()
                line = unicodedata.normalize("NFKC", line).encode("utf-8", "ignore").decode("utf-8").strip()
                if line:
                    new_entities.append(line)
        processed += new_entities
    # 중복 제거 후 반환
    return list(set(processed))

with open("fewshot.json", "r", encoding="utf-8") as f:
    few_entities = json.load(f)

synthetic_entities = []

for real_ent in tqdm(few_entities):
    class_name, entity_names = real_ent['class_name'], real_ent['entity_name']
    prompt = construct_entity_prompt(class_name, entity_names)
    syn_entities = generate(prompt)
    syn_entities = postprocess_entities(syn_entities)
    synthetic_entities.append({'class_name': class_name, 'entity_name': syn_entities})


with open("synthetic.json", "w", encoding="utf-8") as f:
    json.dump(synthetic_entities, f, ensure_ascii=False, indent=4)
# print(json.dumps(synthetic_entities, indent=4, ensure_ascii=False))