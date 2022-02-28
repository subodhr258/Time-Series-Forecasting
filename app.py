from flask import Flask, request, jsonify, render_template
import pandas as pd
from datetime import datetime

app = Flask(__name__)
df = pd.read_csv('Monthly_Predictions.csv')
df.set_index(df['Months'],inplace=True)
df.drop('Months',axis=1,inplace=True)
df.index = pd.to_datetime(df.index)

months = {1:'January',2:'February',3:'March',4:'April',5:'May',
6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    req = [ x for x in request.form.values()][0]
    print(req)
    input_date = datetime.strptime(req,'%Y-%m-%d')
    month_input = datetime(input_date.year,input_date.month,1,0,0,0)

    if month_input > datetime(2020,12,1,0,0,0) or month_input < datetime(2016,6,1,0,0,0):
        return render_template('index.html',prediction_text="Please enter valid date.")

    output = round(df.loc[month_input]['forecast'],1)
    return render_template('index.html', 
    prediction_text=f'The average temperature for {months[input_date.month]} {input_date.year} is about {output}°C')

if __name__ == "__main__":
    app.run(debug=True)
