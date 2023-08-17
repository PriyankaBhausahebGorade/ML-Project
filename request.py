import requests
from flask import Flask, request
url = 'http://localhost:5000'
int_features = [int(x) for x in request.form.values()]
r = requests.post(url,json={'year':int_features[0], 'month':int_features[1], 'pressure':int_features[2],'humidity':int_features[3], 'temperature':int_features[4], 'wind_speed':int_features[5],'wind_direction':int_features[6], 'dew_point':int_features[7]})

print(r.json())