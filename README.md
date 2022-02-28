# Time-Series-Weather-Forecasting
Live Link:https://temp-forecast.herokuapp.com/  
Weather Data: https://drive.google.com/drive/folders/1bxSdbVdfV-p6z6gUgHz_k5dppj0pO5Zh  

##Directory Layout  

    .
    ├── static
        └── css
            └── style.css
    ├── templates
        └── index.html
    ├── EDA_Report.html                 Auto EDA Report generated by Dataprep
    ├── Monthly_Predictions.csv         Output of SARIMAX model
    ├── Procfile                        Procfile for Heroku
    ├── README.md                       
    ├── SARIMAX_Model.py                Main model that generates the predictions
    ├── Time Series Weather Forecast... Jupyter Notebook containing the whole process of analysis, building models and hyperparamter tuning
    ├── Time_series_monthly.csv         Preprocessed input series data for SARIMAX model
    ├── app.py                          The main Flask app.
    ├── requirements.txt                Contains all required packages
    └── runtime.txt                     To specify the version of python to Heroku 

## Local Setup:
```
1) git clone <link>
2) pip install requirements.txt
3) python app.py
4) Hosted URL: http://127.0.0.1:5000/ 
```
