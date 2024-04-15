from flask import Flask, render_template, request
#from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.externals import joblib
import numpy as np
import pickle


with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    # Get input values from the form
    square_feet = float(request.form['square_feet'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    neighborhood = request.form['neighborhood']
    year_built = int(request.form['year_built'])
    
    if neighborhood == 'Rural':
        neighborhood_val = 0
    elif neighborhood == 'Suburb':
        neighborhood_val = 1
    elif neighborhood == 'Urban':
        neighborhood_val = 2

    # Scale the input data
    scaled_features = scaler.transform([[square_feet, bedrooms, bathrooms, year_built, neighborhood_val]])

    # Make prediction
    prediction = model.predict(scaled_features)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)