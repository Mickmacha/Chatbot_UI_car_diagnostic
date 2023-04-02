from flask import Flask, jsonify, render_template, request,url_for
from flask_cors import CORS
from chat import car_diagnostic

app = Flask(__name__)
CORS(app)
# @app.get('/')
# def index_get():
#     return render_template('base.html')

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = car_diagnostic(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
