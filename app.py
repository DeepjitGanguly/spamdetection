from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    msg_vec = vectorizer.transform([message])
    prediction = model.predict(msg_vec)
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template('index.html', result=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
