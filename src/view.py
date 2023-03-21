#必要なライブラリをインポート
import os
import slack
from flask import Flask
from slackeventsapi import SlackEventAdapter
from transformers import BertJapaneseTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
import torch
import numpy as np
import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from scipy.special import eval_sh_legendre
from transformers import AutoTokenizer, AutoModel


SLACK_SIGNING_SECRET = '5b67a7ce8fd3c9eb8f39a9941da7a86c'
SLACK_BOT_TOKEN = ''
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')

import logging
logging.basicConfig(
    level=logging.DEBUG, # ログの出力レベルを指定します。DEBUG, INFO, WARNING, ERROR, CRITICALから選択できます。
    format='%(asctime)s %(levelname)s %(message)s', # ログのフォーマットを指定します。
    datefmt='%Y-%m-%d %H:%M:%S' # ログの日付時刻フォーマットを指定します。
)

app = Flask(__name__)

client = slack.WebClient(token=SLACK_BOT_TOKEN)
BOT_USER_ID = client.api_call("auth.test")['user_id']
slack_event_adapter = SlackEventAdapter(SLACK_SIGNING_SECRET,'/',app)


#文字の類似度計算（cos類似度）★コラボコードの関数は外にだした
def calc_similarity(model, tokenizer,sentences, sentence2):
  sentence_vector2 = sentence_to_vector(model, tokenizer, sentence2)
  scores = []
 #複数(15個の質問文)のsentenceを計算し、scores = []の中に追加していく
  for sentence in sentences:
        sentence_vector1 = sentence_to_vector(model, tokenizer, sentence)
        scores.append(torch.nn.functional.cosine_similarity(sentence_vector1, sentence_vector2, dim=0).detach().numpy().copy())

  # print(scores) 
  return scores

#★コラボコードの関数は外にだした
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


@slack_event_adapter.on('message')
def respond_message(payload):
    # payloadの中の'event'に関する情報を取得し、もし空なら空のディクショナリ{}をあてがう
    event = payload.get('event', {})
    # 投稿のチャンネルID、ユーザーID、投稿内容を取得
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')

    #~~~~~~【ここから】コラボ貼り付け~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)

    #データの読み込み
    df = pd.read_csv('chatbot.csv',header=0,names=['No','Category', 'Title', 'question', 'answer'])
    #df.head(3)

    #データの抽出
    #for row in df.itertuples():
    #   print(row[4])

    #for(ループ)でdfの中を1行ずつ取り出してrow(変数)へ入れて表示する。
    sentences = []
    for row in df.itertuples():
        sentences.append(row[4])

    answers = []
    for row in df.itertuples():
        answers.append(row[5])

    input_sentence=text

    scores = calc_similarity(model, tokenizer,sentences,input_sentence)
    logging.debug(str(scores))
    
    for index, score in enumerate(scores):
        if score > 0.8:
            print(str(index) + ":" + str(score) + ":" + sentences[index])
            #str関数は文字列

    # 最もscoreが高いものを取得
    index = scores.index(max(scores))
    print(sentences[index])
    print(scores[index])

    # scoreが高い順に表示
    # print(scores)
    #argsort関数は並び替えする関数。(scores)が並び替えたい引数。[::-1]
    #sorted_scoreはコサイン類似度ではなくインデックス数がはいってる
    sorted_score=np.argsort(scores)[::-1]
    # print(sorted_score)
    for i in sorted_score:
        print(str(i) + ":" + str(scores[i]) + ":" + sentences[i])

    #if文
    if scores[sorted_score[0]] >= 0.95:
        result = f"{answers[sorted_score[0]]}"
    elif 0.85 <= scores[sorted_score[0]] < 0.95:
        result = f"解答候補を３つ提示します。\n1.{answers[sorted_score[0]]}\n\n2.{answers[sorted_score[1]]}\n\n3.{answers[sorted_score[2]]}"
    else:
        result = 'よくある質問ではないようです。担当者へ問い合わせください。'
    print(result)
    '''
    if scores[sorted_score[0]] >=0.95:
        print(sentences[sorted_score[0]])
        print(answers[sorted_score[0]])    
    elif 0.85 <= scores[sorted_score[0]] < 0.95:
        print('解答候補を３つ提示します。')
        print('1'+ sentences[sorted_score[0]])
        print(answers[sorted_score[0]])
        print('2'+ sentences[sorted_score[1]])
        print(answers[sorted_score[1]])
        print('3'+ sentences[sorted_score[2]])
        print(answers[sorted_score[2]])
    else:
        print('よくある質問ではないようです。担当者へ問い合わせください。')
    '''
    #~~~~~~【ここまで】コラボ貼り付け~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # もしボット以外の人からの投稿だった場合
    if BOT_USER_ID != user_id:
        # chat_postMessageメソッドでオウム返しを実行
        client.chat_postMessage(channel=channel_id, text=result)

if __name__ == "__main__":
    app.run(debug=True)