from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from kobert_tokenizer import KoBERTTokenizer

tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")


app = Flask(__name__)

# 모델 경로는 실제 경로에 맞게
session = ort.InferenceSession("model/emotion_ai_model_final.onnx")

# 클래스 리스트
emotions = ["공포", "놀람", "분노", "슬픔", "중립", "행복", "혐오"]


# 소프트맥스=
def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)  # Overflow 방지
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


@app.route("/")
def home():
    return "Hello I5E2! Please send POST request to /emotion."


@app.route("/emotion", methods=["POST"])
def emotion():
    data = request.get_json()
    text = data["text"]
    print(f"[입력 텍스트] {text}")

    inputs = tokenizer.encode_plus(
        text,
        max_length=64,
        truncation=True,
        padding="max_length",
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors=None,  # tensor가 아닌 리스트 형태 반환
    )

    token_ids = np.array([inputs["input_ids"]], dtype=np.int64)
    segment_ids = np.array([inputs["token_type_ids"]], dtype=np.int64)
    valid_length = np.array(
        [sum(np.array(inputs["attention_mask"]) > 0)], dtype=np.int64
    )

    # # 디버깅
    # print(f"[Token IDs] {token_ids}")
    # print(f"[Valid Length] {valid_length}")
    # print(f"[Segment IDs] {segment_ids}")
    # decoded = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
    # print(f"[Decoded Tokens] {decoded}")

    ort_inputs = {
        "token_ids": token_ids,
        "segment_ids": segment_ids,
        "valid_length": valid_length,
    }

    ort_outs = session.run(None, ort_inputs)
    logits = ort_outs[0]

    probs = softmax(logits)

    predicted_class = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs, axis=1)[0])

    return jsonify(
        {
            "emotion_class": predicted_class,  # 숫자 라벨
            "emotion_name": emotions[predicted_class],  # 라벨 이름
            "confidence": round(confidence, 2),  # 신뢰도 (소수점 2자리)
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
