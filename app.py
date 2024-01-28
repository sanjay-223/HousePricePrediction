from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

def preprocess(X:list, scaler):
    columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    encode_col = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    ocean_proximity = X.pop()
    X = pd.DataFrame([X], columns=columns)
    temp = pd.DataFrame(np.zeros((1, 5)), columns=encode_col)
    temp[ocean_proximity] = 1
    X = X.join(temp)
    X['total_rooms'] = np.log(X['total_rooms'])
    X['total_bedrooms'] = np.log(X['total_bedrooms'])
    X['population'] = np.log(X['population'])
    X['households'] = np.log(X['households'])
    X = scaler.transform(X)
    
    return X


model = joblib.load('random_forest_model.joblib')  
scaler = joblib.load('scaler.joblib')  

@app.route('/')
def input_page():
    return render_template('input_page.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        housing_median_age = float(request.form['housing_median_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bedrooms = float(request.form['total_bedrooms'])
        population = float(request.form['population'])
        households = float(request.form['households'])
        median_income = float(request.form['median_income'])
        ocean_proximity = request.form['ocean_proximity']

        # Preprocess the input data
        input_data = preprocess([longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity], scaler)
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_data)
        
        # Render the output page with the prediction
        return render_template('output_page.html', prediction=prediction[0])

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
