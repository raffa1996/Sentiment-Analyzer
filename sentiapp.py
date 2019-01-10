from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# load the built-in model
gbr = joblib.load('/Users/rahulsharma/model2.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
# @app.route('/api', methods=['POST'])
# def make_prediction():
#     data = request.get_json(force=True)
#     #convert our json to a numpy array
#     one_hot_data = input_to_one_hot(data)
#     predict_request = gbr.predict([one_hot_data])
#     output = [predict_request[0]]
#     print(data)
#     return jsonify(results=output)
def post_data(text):
    df_temp = pd.DataFrame()
    df_temp = df_temp.append({'Text': text},ignore_index=True)
    return df_temp

@app.route('/api',methods=['POST'])
def get_delay():
    result=request.form
    t = result['Phrase']
    user_input = t
    print(user_input)
    temp_df = post_data(user_input)
    vectorizer = CountVectorizer()
    senti_pred = gbr.predict(vectorizer.transform(temp_df['Text']))
    senti_num = senti_pred[0]
    return json.dumps({'Senti':senti_num});
    # return render_template('result.html',prediction=price_pred)

if __name__ == '__main__':
    app.run(port=8080, debug=True)
