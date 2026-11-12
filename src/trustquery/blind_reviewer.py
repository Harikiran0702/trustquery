"""Reviewer CLI: credentials stay outside the custodian repository."""
import argparse
import getpass
import json
import urllib.request

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8765')
    p.add_argument('action',choices=['case','submit'])
    p.add_argument('case_id')
    p.add_argument('--kind',choices=['initial','correction','resolution','internal'],default='initial')
    p.add_argument('--review',help='JSON review file; server records identity and time')
    a=p.parse_args()
    body={'case_id':a.case_id}
    if a.action=='submit':
        if not a.review:
            p.error('--review required')
        with open(a.review,encoding='utf-8-sig') as f:
            body.update(kind=a.kind,review=json.load(f))
    req=urllib.request.Request(a.url+'/'+a.action,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+getpass.getpass('Reviewer credential: '),'Content-Type':'application/json'})
    with urllib.request.urlopen(req) as response:
        print(response.read().decode())
if __name__=='__main__':
    main()
