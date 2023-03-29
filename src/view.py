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
from flask import request
import torch.nn.functional as F


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

class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)
    

@slack_event_adapter.on('message')
def respond_message(payload):
    if request.headers.get('X-Slack-Retry-Num') is None:
    # payloadの中の'event'に関する情報を取得し、もし空なら空のディクショナリ{}をあてがう
        event = payload.get('event', {})
     # 投稿のチャンネルID、ユーザーID、投稿内容を取得
        channel_id = event.get('channel')
        user_id = event.get('user')
        text = event.get('text')

    #~~~~~~【ここから】コラボ貼り付け~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens-v2")

    #データの読み込み
    df = pd.read_csv('chatbot.csv',header=0,names=['No','Category', 'Title', 'question', 'answer'])
    #df.head()

    no = []
    sentences = []
    answers = []
    for row in df.itertuples():
        no.append(row[1])
        sentences.append(row[4])
        answers.append(row[5])

    print(no)
    print(sentences)
    print(answers)


    #変数sentencesに格納された文章の数をカウント（質問数のこと15×10）
    sentences_size = len(sentences)
    print(sentences_size)
    
    #vector_text.ptファイルが存在していて、かつ、行数が現在のchatobot.csvと一致していたら、
    #ベクトル化せずに、すでに前にベクトル化したものを使う。
    #※ファイルがなかったり、chatbot.csvの行数が増えてたりしたら、普通にベクトル化する。
    '''
    vecs = model.encode(sentences, batch_size=sentences_size)
    logging.debug('■len(vecs)='+str(len(vecs)))
    torch.save(vecs,'vector_text.pt')
    logging.debug('■torch.save end')
    
    '''
    chatbotcsv_encode_skip = False
    if  os.path.exists('vector_text.pt'):
        vecs = torch.load('vector_text.pt')
        if len(vecs) == sentences_size:
            chatbotcsv_encode_skip = True

    logging.debug('■chatbotcsv_encode_skip='+str(chatbotcsv_encode_skip))

    if chatbotcsv_encode_skip == False:
        vecs = model.encode(sentences, batch_size=sentences_size)
        torch.save(vecs,'vector_text.pt')
    
    '''
    logging.debug('■len(vecs)='+str(len(vecs)))
    
    input_text = ['MerQNetへ接続できないんですが対処法はありますか。']
    vecs2 = model.encode(input_text, batch_size=1)
    
    input_text=text
    vecs2 = model.encode(input_text, batch_size=1)
    
    logging.debug('■input_text='+str(input_text))
    logging.debug('■len(vecs2)='+str(len(vecs2)))
    

    scores = F.cosine_similarity(vecs2, vecs).tolist()
    logging.debug('■len(scores)='+str(len(scores)))
    scores
    '''
    input_text = []
    input_text.append(text)
    vecs2 = model.encode(input_text, batch_size=1)

    scores = F.cosine_similarity(vecs2, vecs).tolist()
    scores

    # 最もscoreが高いものを取得
    index = scores.index(max(scores))
    print(sentences[index])
    print(scores[index])
    
    sorted_score=np.argsort(scores)[::-1]
    sorted_score

    if scores[sorted_score[0]] >= 0.65:
        result = f"{answers[sorted_score[0]]}"
    elif 0.56 <= scores[sorted_score[0]] < 0.65:
        result = "解答候補を３つ提示します。\n"

        used_no = []
        max_no = 3
        for i in range(len(scores)):
            if no[sorted_score[i]] not in used_no:
                #print(str(i)+', score='+str(scores[sorted_score[i]])+', no='+str(no[sorted_score[i]])+', sentences='+str(sentences[sorted_score[i]])+', answesr='+str(answers[sorted_score[i]]))
                result += f"{len(used_no)+1}. {answers[sorted_score[i]]}\n\n"
                used_no.append(no[sorted_score[i]])

            if len(used_no)==max_no:
                break

    else:
        result = "よくある質問ではないようです。担当者へ問い合わせください。"
    print(result)

    #~~~~~~【ここまで】コラボ貼り付け~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # もしボット以外の人からの投稿だった場合
    if BOT_USER_ID != user_id:
        # chat_postMessageメソッドでオウム返しを実行
        client.chat_postMessage(channel=channel_id, text=result)

if __name__ == "__main__":
    app.run(debug=True)