
from flask import Flask, request, jsonify, render_template
from models.ngram_predictor import NGramPredictor
from models.lstm_predictor import train_model, load_model, predict_next_words
import threading
import os

app = Flask(__name__)

# load data
DATAFILE = "data/sample_corpus.txt"
with open(DATAFILE, "r", encoding="utf8") as f:
    LINES = [l.strip() for l in f if l.strip()]

# build ngram model
ngram = NGramPredictor(n=3)
ngram.train(LINES)

# prepare or load LSTM model
LSTM_PATH = "models/lstm_demo.pth"
if not os.path.exists(LSTM_PATH):
    print("Training small LSTM (this may take ~1-2 minutes)...")
    # train in same process (blocking) — small dataset so OK
    train_model(LINES, seq_len=4, epochs=50, batch_size=8, lr=0.01, save_path=LSTM_PATH)
model, vocab, inv_vocab, seq_len = load_model(LSTM_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/ngram", methods=["POST"])
def predict_ngram():
    data = request.json
    context = data.get("context", "")
    words = context.strip().split()
    preds = ngram.predict_next(words, top_k=5)
    return jsonify([{"word": w, "prob": float(p)} for w,p in preds])

@app.route("/predict/lstm", methods=["POST"])
def predict_lstm():
    data = request.json
    context = data.get("context", "")
    words = context.strip().split()
    preds = predict_next_words(model, vocab, inv_vocab, seq_len, words, top_k=5)
    return jsonify([{"word": w, "prob": float(p)} for w,p in preds])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
