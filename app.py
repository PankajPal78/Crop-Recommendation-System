from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the pre-trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('StandardScaler.pkl', 'rb'))
ms = pickle.load(open('minmaxScaler.pkl', 'rb'))

# Initialize the Flask application
app = Flask(__name__)

# Define the routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Retrieve form data
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Prepare feature array for prediction
        feature_list = [N, P, K, temp, humidity, pH, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply scalers
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Make prediction
        prediction = model.predict(final_features)

        # Define crop dictionary
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
            12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
            21: "Chickpea", 22: "Coffee"
        }

        # Get crop name from prediction
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there."
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

        # Render the result
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"An error occurred: {str(e)}")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
