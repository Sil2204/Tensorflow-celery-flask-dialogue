import requests
import random

sentence = raw_input("translate_client Enter sentence/ Enter exit:") 
p = {'sentence':sentence}
r = requests.get('http://localhost:5000/test', params = p)
print r.json()
gto = r.json()['goto']
r2 = requests.get("http://localhost:5000/test/result/{}".format(gto))
print r2.json()['result'.decode('utf-8')]
