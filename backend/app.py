from flask import Flask, render_template
from react.render import render_component
from flask import jsonify, json
import requests


app = Flask(__name__)

@app.route('/')
def index(name=None):
    return render_template("hello.html", name=name)

@app.route('/hello')
def hello():
    return 'Hello, PennApps 2017!'

@app.route('/submit/<username>', methods=['GET','POST'])
def post(username): 
    url = 'https://data.chastiser11.hasura-app.io/v1/query'
    payload = {"type": "insert", "args":{"table": "author", "objects": [{"id": 4, "name": username}]}}
    headers = {"content-type": "application/json", "authorization": "Bearer wj7fmf21w6lvu0l4u7vmdef1tqo0cykn"}
    r = requests.post(url, json.dumps(payload), headers=headers)
    if r.status_code == 200:
        return 'success'
    else:
        return 'error: ' + str(r.status_code) + '\n\n' + r.text

