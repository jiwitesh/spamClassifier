from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle
from wsgiref import simple_server


# Loading Models
modelLoc = './models/model.pkl'
classifier = pickle.load(open(modelLoc, 'rb'))
modifierModel=pickle.load(open('./models/modifier.pkl','rb'))

#initialising Flask instance
app = Flask(__name__)

# Display the home page 
@app.route('/')
def index():
	return render_template('index.html')

# get the prediction
@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		inputText = request.form['message']
		inputData = [inputText]
		wordVectors = modifierModel.transform(inputData).toarray()
		predictedOutput = classifier.predict(wordVectors)
	return render_template('prediction.html',prediction = predictedOutput)



if __name__ == "__main__":
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()