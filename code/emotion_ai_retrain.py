# fmt:off
## 1. 초기 설정

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
import pandas as pd

# koBERT, transformers lib import
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

## CPU
#device = torch.device("cpu")

## GPU
device = torch.device("cuda:0")

# BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

# [AI Hub] 감정 분류를 위한 대화 음성 데이터셋
data = pd.read_csv("emotion_ai/self_dataset.csv", encoding="EUC-KR")


## 2. 데이터 전처리

# 7개의 감정 class → 숫자
data.loc[(data["감정"] == "공포"), "감정"] = 0  # fear → 0
data.loc[(data["감정"] == "놀람"), "감정"] = 1  # surprise → 1
data.loc[(data["감정"] == "분노"), "감정"] = 2  # angry → 2
data.loc[(data["감정"] == "슬픔"), "감정"] = 3  # sadness → 3
data.loc[(data["감정"] == "중립"), "감정"] = 4  # neutral → 4
data.loc[(data["감정"] == "행복"), "감정"] = 5  # happiness → 5
data.loc[(data["감정"] == "혐오"), "감정"] = 6  # disgust → 6

# [발화문, 상황] data_list 생성
data_list = []
for ques, label in zip(data["로그"], data["감정"]):
    data = []
    data.append(ques)
    data.append(str(label))

    data_list.append(data)

## data_list 검토
#print(data)
#print(data_list[:10])

# data_list train : test = 8 : 2로 나누기
from sklearn.model_selection import train_test_split

dataset_train, dataset_test = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=32)

# 데이터셋 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# BERTDataset : 각 데이터가 BERT 모델의 입력으로 들어갈 수 있도록 tokenization, int encoding, padding하는 함수
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab = vocab, pad = pad, pair = pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         

    def __len__(self):
        return (len(self.labels))


# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 7
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, max_len, True, False)

# torch 형식으로 변환
#train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
#test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)


## 3. 모델 구현

class BERTClassifier(nn.Module):
    def __init__(
        self,
        bert,
        hidden_size=768,
        num_classes=7,  # 클래스 수로 조정
        dr_rate=None,
        params=None,
    ):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device),
            return_dict=False,
        )
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


# BERT  모델 불러오기 (dropout_rate: Overfiitingg 방지)
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# optimizer와 schedule 설정
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()  # 다중분류를 위한 loss function

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


# calc_accuracy : 정확도 측정
def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc

# calc_f1score : f1 score 측정
from sklearn.metrics import f1_score

def calc_f1score(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    # max_indices와 Y를 numpy 배열로 변환
    f1score = f1_score(Y.data.cpu().numpy(), max_indices.data.cpu().numpy(), average="macro")
    return f1score


## 4. Train

train_history = []
test_history = []
loss_history = []

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    train_f1score = 0.0
    test_f1score = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        # print(label.shape, out.shape)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        train_f1score += calc_f1score(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} | batch id {} | loss {} | train acc {} | f1 score {}".format(
                    e + 1,
                    batch_id + 1,
                    loss.data.cpu().numpy(),
                    train_acc / (batch_id + 1),
                    train_f1score / (batch_id + 1),
                ))
            train_history.append(train_acc / (batch_id + 1))
            loss_history.append(loss.data.cpu().numpy())
    print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
    # train_history.append(train_acc / (batch_id+1))

    # .eval() : nn.Module에서 train time과 eval time에서 수행하는 다른 작업을 수행할 수 있도록 switching 하는 함수
    # 즉, model이 Dropout이나 BatNorm2d를 사용하는 경우, train 시에는 사용하지만 evaluation을 할 때에는 사용하지 않도록 설정해주는 함수
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
        test_f1score += calc_f1score(out, label)
    print("epoch {} | test acc {} | f1 score {}".format(e + 1, test_acc / (batch_id + 1), test_f1score / (batch_id + 1)))
    test_history.append(test_acc / (batch_id + 1))


## 5. TEST

# predict : 학습 모델을 활용하여 다중 분류된 클래스를 출력해주는 함수

def predict(predict_sentence): # input = 감정분류하고자 하는 sentence

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False) # 토큰화한 문장
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = batch_size, num_workers = 5) # torch 형식 변환
    
    model.eval() 

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval = []
        for i in out: # out = model(token_ids, valid_length, segment_ids)
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("공포가")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람이")
            elif np.argmax(logits) == 2:
                test_eval.append("분노가")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 4:
                test_eval.append("중립이")
            elif np.argmax(logits) == 5:
                test_eval.append("행복이")
            elif np.argmax(logits) == 6:
                test_eval.append("혐오가")

        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")



# 질문에 0 입력 시 종료
end = 1
while end == 1:
    sentence = input("하고싶은 말을 입력해주세요 : ")
    if sentence == "0":
        break
    predict(sentence)
    print("\n")


## 6. onnx 변환

# 더미 입력 (token_ids, valid_length, segment_ids)
dummy_input = (
    torch.randint(0, len(vocab), (1, max_len)).to(device),
    torch.tensor([max_len]).to(device),
    torch.zeros(1, max_len).long().to(device), 
)

torch.onnx.export(
    model,
    dummy_input,  # 입력 튜플
    "emotion_ai_model_final.onnx",  # 저장 경로
    export_params=True,
    input_names=["token_ids", "valid_length", "segment_ids"],
    output_names=["output"], 
    dynamic_axes={
        "token_ids": {0: "batch_size"}, # max_len이 고정 -> seq_length도 고정
        "valid_length": {0: "batch_size"},
        "segment_ids": {0: "batch_size"},
        "output": {0: "batch_size"},
    }, 
    opset_version=11,  # opset_version 설정
)

print(f"ONNX 변환이 완료되었습니다!")
# fmt:off
