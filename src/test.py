import slack
from flask import Flask
from slackeventsapi import SlackEventAdapter
SLACK_SIGNING_SECRET = '5b67a7ce8fd3c9eb8f39a9941da7a86c'
SLACK_BOT_TOKEN = ''
import os
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
#OSの環境変数に設定
app = Flask(__name__)
 
# トークンを指定してWebClientのインスタンスを生成
client = slack.WebClient(token=SLACK_BOT_TOKEN)
# ボットのユーザーIDを取得
BOT_USER_ID = client.api_call("auth.test")['user_id']
 
# Slackから来たイベントを受け付けるためのインスタンスを生成
# 第1引数：Singing Secretキーを指定
# 第2引数：イベントの送り先（ルートURL以降の部分）
# 第3引数：イベントの送り先のWebサーバ（今回はFlaskのインスタンス）
slack_event_adapter = SlackEventAdapter(
    SLACK_SIGNING_SECRET,'/',app)
 
# メッセージが投稿された事を検知してイベントを発行
@slack_event_adapter.on('message')
 
# 誰かの投稿に対してオウム返しする関数
# payloadはSlack APIが送ってくるデータで、投稿に関する情報を保持している
def respond_message(payload):
    # payloadの中の'event'に関する情報を取得し、もし空なら空のディクショナリ{}をあてがう
    event = payload.get('event', {})
    # 投稿のチャンネルID、ユーザーID、投稿内容を取得
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
 
    # もしボット以外の人からの投稿だった場合
    if BOT_USER_ID != user_id:               
        # chat_postMessageメソッドでオウム返しを実行
        client.chat_postMessage(channel=channel_id, text=text)
 
if __name__ == "__main__":
    app.run(debug=True)
