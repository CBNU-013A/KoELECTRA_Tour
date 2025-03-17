import pandas as pd
import torch
import sys
from collections import Counter
import os

sys.stdout.reconfigure(encoding='utf-8')

from transformers import ElectraTokenizer, ElectraForTokenClassification

model_path = "/srv/013a/Build_02/Models/NER"
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = ElectraForTokenClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def merge_tokens(tokens, labels):
    merged_tokens = []
    merged_labels = []

    current_word = ""
    current_label = None

    for token, label in zip(tokens, labels):
        # O 라벨이면 스킵
        if label == "O" or label is None:
            # 만약 현재 누적 중인 토큰이 있으면 우선 저장
            if current_word:
                merged_tokens.append(current_word)
                merged_labels.append(current_label)
                current_word = ""
                current_label = None
            continue

        # 현재 라벨이 O가 아닐 때만 병합 로직 수행
        if token.startswith("##"):
            # 서브워드이면 앞 토큰에 붙이기
            current_word += token[2:]
            # 필요하다면 LOC->PLC 전환 처리 로직을 추가
            if current_label == "B-LOC" and label == "B-PLC":
                current_label = "B-PLC"
        else:
            # 기존에 누적된 토큰이 있으면 저장
            if current_word:
                merged_tokens.append(current_word)
                merged_labels.append(current_label)
            current_word = token
            current_label = label

    # 마지막 남은 토큰 처리
    if current_word:
        merged_tokens.append(current_word)
        merged_labels.append(current_label)

    return merged_tokens, merged_labels

def predict_NER(sentence):
    # 3-1. 토크나이저 입력
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # 3-2. 모델 추론
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]

    # 3-3. Special Token 제거 ([CLS], [SEP], [PAD])
    filtered_tokens, filtered_labels = zip(*[
        (t, l) for t, l in zip(tokens, predicted_labels)
        if t not in ["[CLS]", "[SEP]", "[PAD]"]
    ])

    # 3-4. 서브워드 병합
    merged_tokens, merged_labels = merge_tokens(filtered_tokens, filtered_labels)

    # 3-5. (token, label) 형태로 결과 반환
    results = []
    for token, label in zip(merged_tokens, merged_labels):
        results.append((token, label if label else "O"))
    return results

def extract_ASP_PLC(sentence):
    ner_results = predict_NER(sentence)
    # 관심 라벨: B-ASP, I-ASP, B-PLC, I-PLC
    filtered = [
        (tok, lab)
        for (tok, lab) in ner_results
        if lab and (lab.startswith("B-ASP") or lab.startswith("I-ASP")
                    or lab.startswith("B-PLC") or lab.startswith("I-PLC"))
    ]
    return filtered

def process_reviews(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    # 리뷰 텍스트가 담긴 컬럼명 (필요에 따라 수정)
    review_col = "Review"

    asp_counter = Counter()
    plc_counter = Counter()

    for idx, row in df.iterrows():
        review_text = str(row[review_col]) if review_col in row else ""
        extracted_tokens = extract_ASP_PLC(review_text)  # [(token, label), ...]

        for token, label in extracted_tokens:
            if label.startswith("B-ASP") or label.startswith("I-ASP"):
                asp_counter[token] += 1
            elif label.startswith("B-PLC") or label.startswith("I-PLC"):
                plc_counter[token] += 1

    # 결과를 데이터프레임으로 변환하고 빈도순 정렬
    asp_df = pd.DataFrame(asp_counter.items(), columns=["ASP_Token", "ASP_Count"]).sort_values(by="ASP_Count", ascending=False)
    plc_df = pd.DataFrame(plc_counter.items(), columns=["PLC_Token", "PLC_Count"]).sort_values(by="PLC_Count", ascending=False)

    # CSV로 저장
    asp_df.to_csv(output_csv.replace(".csv", "_asp_counts.csv"), index=False)
    plc_df.to_csv(output_csv.replace(".csv", "_plc_counts.csv"), index=False)

    print(f"[완료] '{output_csv.replace('.csv', '_asp_counts.csv')}' 및 '{output_csv.replace('.csv', '_plc_counts.csv')}' 파일로 저장했습니다.")

if __name__ == "__main__":
    import argparse
    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description="Process reviews and extract ASP/PLC labels.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file containing reviews.")
    args = parser.parse_args()

    input_csv = args.input_csv

    # 결과 저장 경로 설정
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    # 입력 파일명을 기반으로 출력 파일명 생성
    base_filename = os.path.basename(input_csv).replace(".csv", "")
    output_csv = os.path.join(result_dir, f"{base_filename}_asp_plc.csv")

    process_reviews(input_csv, output_csv)