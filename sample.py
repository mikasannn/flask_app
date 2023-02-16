#必要なモジュールのインポート
from flask import Flask

#　Flask をインスタンス化
app = Flask(__name__)

# ルートディレクトリ(URLに一番最初にaccessがあったときに表示する場所のこと)にアクセスがあった時の処理
@app.route('/')
def hello():
    return'Hello world!'

# エントリーポイント
if __name__=='__main__':
   app.run()

