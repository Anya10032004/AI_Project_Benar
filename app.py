from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
 # Create the Flask app

app = Flask(__name__)

def clip_outliers(data, lower_percentile=0.25, upper_percentile=0.75, factor=1.5):
    df_clipped = data.copy()
    for column in df_clipped.columns:
        Q1 = df_clipped[column].quantile(lower_percentile)
        Q3 = df_clipped[column].quantile(upper_percentile)

        IQR = Q3 - Q1 

        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)

        df_clipped[column] = np.clip(df_clipped[column], lower_bound, upper_bound)
    
    return df_clipped

# Load the model and scaler
model = joblib.load('models/final_rff_model(2).pkl')
process_data = joblib.load('models/preprocessing_pipeline2.pkl')



# Define a home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the user (JSON format)
        # data = request.join

        # Extract features in the correct order

        # Before: 
        # input_data = [data['Pregnancies'], data['Glucose'], data['BloodPressure'], 
        #               data['SkinThickness'], data['Insulin'], data['BMI'], 
        #               data['DiabetesPedigreeFunction'], data['Age']]

        # After 
        name = request.form.get('name') 
        glucose = float(request.form.get('glucose'))
        bmi = float(request.form.get('bmi'))
        blood_pressure = float(request.form.get('bloodPressure'))
        insulin = float(request.form.get('insulin'))
        pregnancy = int(request.form.get('pregnancy'))
        skin_thickness = float(request.form.get('skinThickness'))
        age = int(request.form.get('age'))

        
        # Convert to NumPy array and reshape for the model
        # Before(We convert into numpy array):
        # input_array = np.array(input_data).reshape(1, -1)

        # After(We convert into pandas dataframe): 
        input_data = pd.DataFrame([{
            'Pregnancies': pregnancy,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'Age': age
        }])


        # Process the data(check apakah ada yang negative atau 0, buang outlier, etc)
        after_the_data_got_procesed = process_data.transform(input_data)

        # Make a prediction 
        prediction = model.predict(after_the_data_got_procesed)

        # return the prediction result
        return jsonify({'diabetes': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug= True)


# To do list 03/12/24:
#  1. So the main problem is that we need a sckit-learn with a version 
#     of 1.0.2, and we can do that by "pip install scikit-learn==1.0.2"
#     but in order to excecuteit  sucessfully, we need the distutils Module, but that module  is not exist and cannot 
#     be installed by "pip install distutils" because it was a built-in package at the beggining.
#     The error we get when  "pip install distutils" 
#     is 
#     "
#      ERROR: Could not find a version that satisfies the requirement distutils (from versions: none)
#      ERROR: No matching distribution found for distutils
#     "
#    So the main problem is "How to get the distutils module"
#    But unfortunetly, the distutils module only provided in the older python version(which is bellow 3.12)
#    And our is 3.13, we already download the python that is bellow 3.13, which is 3.7, but in order
#    to use that python version(the 3.7 one), we have to make a virtual enviroment

#    So the main problem is to create a virtual enviroment and we will put the older python version it 

#   We have made the virtual enviroment and put the proper python for our case in it, i thought the problem will be over but turns out when i tried to 
#   to install the sklearn module with version 1.0.2 by "pip install sckit-learn==1.0.2" turns out i encaunter a new problem, which is 
#   the sckit-learn needs to be compiled during installitation and to do that i need a C++ compiler.

#   So know we have to get the  Microsoft Visual C++ version 14.0 or newer

#   The steps we will do in detail is in this link: https://chatgpt.com/c/673fe8a1-cd10-8000-8cde-7928b24e8446

#   Okay turns out that my friend success to to make the backend using Flask. I check the version of the sklearn she use and it's the same.
#   She said that maybe because she remade the "final_rff_model.pkl" and the "preprocessing_pipeline1.pkl" in her new laptop that has the same version of sklearn as i do 
#   because she make the old "final_rff_model.pkl" and "preprocessing_pipeline1.pkl" in here old laptop that use old python, which mean an old version of sklearn 
#   And when re-make the files on her new laptop, it will automatically follow the sklearn version on her new laptop

#   So what we really going to do is to use the new "finale_rff_model.pkl" and "preprocessing_pipeline1.pkl", so we will not use the old one 
#   And maybe after that let's try re-make the application but using our backend 
#   If we sucess, let's try to understand the flow program of o

#   So we will do the next task not with this laptop because the command prompt on this laptop run really slow, but the other laptop 
#   have the old version of python, so before we do the next task(use the new "finale_rff_model.pkl" and "preprocessing_pipeline1.pkl")
#   01/01/25:
#    1. Update to new python(our current one is 3.7.5, we will update it to the new one)
#    2. Udate some libraries that we need

#   Hey, turns out my laptop is back to normal, so we already install the things we need, but we forgot to put the file of the website(html css and js)
#   SO we will put here after that, those files(the html, css, abd JS) is located on another file call AI

#   Yey it work, but it is working on the app2.py, but your file is in the app.py, now i guess the next step is to learn in the alur of the code from
#   the app2.py so that what should we do that and the same time also write the right code to the app.py



