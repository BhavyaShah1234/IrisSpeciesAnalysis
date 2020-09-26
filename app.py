import flask
import numpy as np
import pickle

app = flask.Flask(__name__, template_folder=r'C:\Users\user\Desktop\Machine Learning\Iris\template')
model = pickle.load(open(r'C:\Users\user\Desktop\Machine Learning\Iris\model.pkl', 'rb'))

@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    species = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    features = [x for x in flask.request.form.values()]
    features = [np.array(features)]
    output = model.predict(features)
    return flask.render_template('index.html', prediction_text=f'Species={species[output[0]]}')

if __name__ == "__main__":
    app.run(debug=True)
