import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'sl': 2.3, 'sw': 4.2, 'pl': 3.2, 'pw': 1.1})

print(r.json())
