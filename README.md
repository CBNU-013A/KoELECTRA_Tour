# KoELECTRA_tour

*(가칭)*


충북대학교 013A팀의 사용자 리뷰 기반 관광지 추천 시스템에 사용되는 KoELECTRA 기반 NLP 모델입니다. 사용자의 리뷰에서 관심사를 추출(NER)하고, 해당 관심사를 감성분석(ABSA)을 진행합니다.

## Fine-tuning
### [NER](Finetuning/NER/finetuning_ner.ipynb)

- [DataSet](Finetuning/NER/data/make_dataset.ipynb)
    - OpenAI의 GPT-4o를 활용한 few-shot 기법을 통해 학습데이터를 생성.

|F1|loss|precision|
|--|--|--|
|0.76|0.34|0.76|
> 2025/02/23 기준. 학습 문장수 500개. 보강 예정

### ABSA