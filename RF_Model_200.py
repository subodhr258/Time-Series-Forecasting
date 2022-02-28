import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import joblib

def get_model(sequences):
    
    X = sequences.iloc[:,:-1]
    y = sequences.iloc[:,-1]
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=10)
    
    rf = RandomForestRegressor(n_estimators=125,max_features='sqrt',max_depth=15,
                               min_samples_split=10,min_samples_leaf=1,bootstrap=False)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    print("r2_score:",r2_score(y_test,y_pred))
    print("mse:",mean_squared_error(y_test,y_pred))
    print("mae:",mean_absolute_error(y_test,y_pred))
    
    return rf

sequence_data = pd.read_csv('./Weather_data_Sequences_200.csv')
rf_model = get_model(sequence_data)
joblib.dump(rf_model,'rf_model_200.h5')
print('\nModel Saved!')

