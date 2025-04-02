import onnxruntime as ort
import numpy as np
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

# KoBERT 토크나이저 및 모델 로드
tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

onnx_path = "E:/star/BabyLION/code/emotion_ai_model_final.onnx"  # 내가 변환한 모델 경로
onnx_session = ort.InferenceSession(onnx_path)


def predict(sentence):
    inputs = tokenizer.encode_plus(
        sentence,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    segment_ids = inputs[
        "token_type_ids"
    ]  # KoBERT에서는 token_type_ids를 segment_ids로 사용

    # KoBERT의 [PAD] 토큰 ID 가져오기
    pad_token_id = tokenizer.pad_token_id
    valid_length = (input_ids != pad_token_id).sum(axis=1).tolist()[0]
    # print("segment_ids:", segment_ids)

    # ONNX 모델 입력값 준비
    ort_inputs = {
        "token_ids": input_ids.astype(np.int64),  # ✅ 이름 변경
        "valid_length": np.array([valid_length], dtype=np.int64),  # ✅ 추가
        "segment_ids": segment_ids.astype(np.int64),  # ✅ 이름 변경
    }

    # ONNX 실행
    ort_outs = onnx_session.run(None, ort_inputs)
    logits = ort_outs[0]

    # 소프트맥스 적용 후 감정 예측
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    predicted_class = np.argmax(probs, axis=1)[0]
    emotions = ["공포", "놀람", "분노", "슬픔", "중립", "행복", "혐오"]
    predicted_emotion = emotions[predicted_class]
    print(f">> 입력하신 내용에서 {predicted_emotion}이 느껴집니다.")


end = 1
while end == 1:
    sentence = input("하고싶은 말을 입력해주세요 : ")
    if sentence == "0":
        break
    predict(sentence)
    print("\n")
