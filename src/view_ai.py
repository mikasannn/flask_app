#必要なライブラリをインポート
from transformers import BertJapaneseTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
import torch
import numpy as np
import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

#データの読み込み
df = pd.read_csv('chatbot.csv',header=0,names=['No','Category', 'Title', 'question', 'answer'])
df.head(3)

#データの抽出
for row in df.itertuples():
    print(row[4])

def sentence_to_vector(model, tokenizer, sentence):

  # 文を単語（=トークン）に区切って数字にラベル化　⇒　PCが処理できる入力形式に変換（それを実行するのがトークナイザ）形態素解析
  tokens = tokenizer(sentence)["input_ids"]

  # BERTモデルの処理のためtensor型に変換
  input = torch.tensor(tokens).reshape(1,-1)

  # BERTモデルに入力し文のベクトルを取得
  with torch.no_grad():
    outputs = model(input, output_hidden_states=True)
    last_hidden_state = outputs.last_hidden_state[0]
    averaged_hidden_state = last_hidden_state.sum(dim=0) / len(last_hidden_state) 

  return averaged_hidden_state

#for(ループ)でdfの中を1行ずつ取り出してrow(変数)へ入れて表示する。
sentences = []
for row in df.itertuples():
    sentences.append(row[4])

answers = []
for row in df.itertuples():
    answers.append(row[5])

from scipy.special import eval_sh_legendre
#文字の類似度計算（cos類似度）
def calc_similarity(sentences, sentence2):
  sentence_vector2 = sentence_to_vector(model, tokenizer, sentence2)
  scores = []
 #複数(15個の質問文)のsentenceを計算し、scores = []の中に追加していく
  for sentence in sentences:
        sentence_vector1 = sentence_to_vector(model, tokenizer, sentence)
        scores.append(torch.nn.functional.cosine_similarity(sentence_vector1, sentence_vector2, dim=0).detach().numpy().copy())

  # print(scores) 
  return scores

input_sentence="iphoneの設定方法がわかりません"

scores = calc_similarity(sentences,input_sentence)


for index, score in enumerate(scores):
    if score > 0.8:
        print(str(index) + ":" + str(score) + ":" + sentences[index])
        #str関数は文字列

# 最もscoreが高いものを取得
index = scores.index(max(scores))
print(sentences[index])
print(scores[index])

import numpy as np
# scoreが高い順に表示
# print(scores)
#argsort関数は並び替えする関数。(scores)が並び替えたい引数。[::-1]
#sorted_scoreはコサイン類似度ではなくインデックス数がはいってる
sorted_score=np.argsort(scores)[::-1]
# print(sorted_score)
for i in sorted_score:
    print(str(i) + ":" + str(scores[i]) + ":" + sentences[i])

#if文
if scores[sorted_score[0]] >=0.95:
     print(sentences[sorted_score[0]])
     print(answers[sorted_score[0]])    
elif 0.85 <= scores[sorted_score[0]] < 0.95:
     print('解答候補を３つ提示します。')
     print('１．'+ sentences[sorted_score[0]])
     print(answers[sorted_score[0]])
     print('２．'+ sentences[sorted_score[1]])
     print(answers[sorted_score[1]])
     print('３．'+ sentences[sorted_score[2]])
     print(answers[sorted_score[2]])
else:
    print('よくある質問ではないようです。担当者へ問い合わせください。')
