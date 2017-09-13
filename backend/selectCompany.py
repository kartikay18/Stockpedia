import requests
import json

url = 'https://data.chastiser11.hasura-app.io/v1/query'

# Sample row input
stock_name = "GOOG"

headers = {"content-type": "application/json", "authorization": "Bearer wj7fmf21w6lvu0l4u7vmdef1tqo0cykn"}
payload = {"type": "select", "args":{"table": "stockTest", "columns": ["*"], "where":{ "stocksymbol": stock_name }}}
r = requests.post(url, json.dumps(payload), headers=headers)
if r.status_code == 200:
    print 'success'
else:
    print 'error: ' + str(r.status_code) + '\n\n' + r.text