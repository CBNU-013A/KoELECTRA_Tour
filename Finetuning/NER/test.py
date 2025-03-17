from transformers import ElectraTokenizer, ElectraForTokenClassification
import torch
import sys
sys.stdout.reconfigure(encoding='utf-8')

model_path = "./ckpt/output/checkpoint-100/"

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = ElectraForTokenClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()    

from konlpy.tag import Mecab

mecab = Mecab()

def merge_tokens(tokens, labels):
    merged_tokens = []
    merged_labels = []

    current_word = ""
    current_label = None

    for token, label in zip(tokens, labels):
        if token.startswith("##"):  # 서브워드이면 앞 단어에 붙이기
            current_word += token[2:]
            if current_label == "B-LOC" and label == "B-PLC":
                current_label = "B-PLC"  # LOC + PLC 조합이면 PLC 유지
        else:
            if current_word:
                merged_tokens.append(current_word)
                merged_labels.append(current_label)
            current_word = token
            current_label = label

    if current_word:
        merged_tokens.append(current_word)
        merged_labels.append(current_label)

    return merged_tokens, merged_labels


def to_noun_form(word):
    """ Mecab을 활용하여 단어를 명사형으로 변환 """
    parsed = mecab.pos(word)
    noun_form = ""

    for morph, tag in parsed:
        if tag.startswith("VA"):  # 형용사 처리 (예쁘 → 예쁨, 빠르 → 빠름)
            if morph.endswith("다"):  
                base_morph = morph[:-1]  # "예쁘다" → "예쁘"
            else:
                base_morph = morph

            # "으" 탈락 규칙 적용
            if base_morph.endswith("으"):
                noun_form += base_morph[:-1] + "ㅁ"  # "빠르" → "빠름"
            else:
                noun_form += base_morph + "ㅁ"  # "예쁘" → "예쁨"

        elif tag.startswith("VV"):  # 동사 처리 (좋 → 좋음)
            noun_form += morph + "음"

        elif tag.startswith("NNG") or tag.startswith("NNP"):  # 명사는 유지
            noun_form += morph

        elif tag in ["JOSA", "EC", "EP", "EF"]:  # 조사 및 어미 제거
            continue

        else:
            noun_form += morph  # 나머지는 유지

    return noun_form if noun_form else word  # 변환된 값이 없으면 원래 단어 유지


def Test(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]

    # Special Token 제거
    filtered_tokens, filtered_labels = zip(*[(t, l) for t, l in zip(tokens, predicted_labels) if t not in ["[CLS]", "[SEP]"]])

    # 서브워드 합치기
    merged_tokens, merged_labels = merge_tokens(filtered_tokens, filtered_labels)

    # 형태소 분석을 통해 명사형 변환
    transformed_tokens = [to_noun_form(token) for token in merged_tokens]

    print("\n[NER 결과]:")
    for token, transformed, label in zip(merged_tokens, transformed_tokens, merged_labels):
        if label != "O":  # "O" 태그는 출력하지 않음
            print(f"{token} ({transformed}): {label}")

Test("5층에 챔피온1250이 있다 아이와 오기 좋다 옷도 팔고 도서관도 있다 식당도 있다 할게 많다")