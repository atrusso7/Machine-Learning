import requests, json, sys, random, time

def hash():
    email = 'atrusso7@gmail.com'
    # r = requests.post('https://mlb.praetorian.com/api-auth-token/', data={'email':email})
    # r.json()
    # headers = r.json()
    # headers['Content-Type'] = 'application/json'
    r = requests.get('https://mlb.praetorian.com/hash', data={'email': email})
    hash = r.json()
    print(hash)

hash()
