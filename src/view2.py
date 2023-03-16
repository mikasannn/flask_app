'''
# coding: utf-8

#　必要なモジュールのインポート
# from flask import Flask, render_template, make_response, request, jsonify
from flask import Flask, request, jsonify

# app という変数でFlaskオブジェクトをインスタンス化
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    # Slackから送られてくるPOSTリクエストのBodyの内容を取得
    json =  request.json
    print(json)
    # レスポンス用のJSONデータを作成
    # 受け取ったchallengeのKey/Valueをそのまま返却する
    d = {'challenge' : json["challenge"]}
    # レスポンスとしてJSON化して返却
    return jsonify(d)

if __name__ == '__main__':
    app.run(debug=True)
'''
