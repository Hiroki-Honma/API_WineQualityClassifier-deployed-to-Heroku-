from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

#訓練済みモデルの読み込み
clf = joblib.load('clf_wine_quality.pkl')

@app.route('/', methods=['POST'])
def predict():
    x = request.json['x']
    y = clf.predict([x])[0]   #[0]はリストの"中身"を取り出している
    ret = {"y" : int(y)}
    return jsonify(ret)

if __name__ == '__main__':
    app.run(debug=True)