import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import datetime

app = Flask(__name__)
model_cls = pickle.load(open('new_model_RF.pkl', 'rb'))
model_reg = pickle.load(open('new_model_gb_3sets.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    #For rendering results on HTML GUI
    l=list(request.form.values())
    dat=l[0]
    d=pd.to_datetime(dat)
    year=d.year
    day=d.day
    l.pop(0)
    l.insert(0,day)
    l.insert(0,year)
    print(l)
    int_features = [float(x) for x in l]
    #final_features = [np.array(int_features)]
    final_features = [int_features]
    final_features_clasi=final_features
    prediction = model_cls.predict(final_features_clasi)
    col=['year', 'month', 'pressure', 'humidity', 'temperature', 'wind_speed','wind_direction', 'dew_point']
    final_features_reg=pd.DataFrame(final_features,columns=col)
    pred_value=model_reg.predict(final_features_reg)

    #output = round(prediction[0], 2)
    #flag= prediction[0]
    flag= prediction[0]
    abc=pred_value[0]
    val=np.round(abc,2)
    print('val ##########################################################################',val)
    print('flag',flag)
    if flag==0:
        return render_template('index.html', prediction_text='There is no Rainfall on this day')
    elif val>15:
        #return  render_template('index.html', prediction_text='There is high chance of Rainfall on this day with intensity of {}'.format(pred_value))
        return  render_template('index.html', prediction_text='high chances : {} mm/day'.format(val))
    else:
        #return render_template('index.html', prediction_text='There is less chance of Rainfall on this day with intensity of {}'.format(pred_value))
        return  render_template('index.html', prediction_text='less chances : {} mm/day'.format(val))

if __name__ == "__main__":
    app.run(debug=True)