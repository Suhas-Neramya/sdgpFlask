import requests

url = 'http://localhost:5000/predict'
response = requests.post(url)
print(response.status_code)
print(response.text)

