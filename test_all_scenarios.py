# -*- coding: utf-8 -*-
"""
Prova end-to-end dei 5 intent di C-3PO su HOME-SERVER (via SSH + curl locale).
Credenziali come server_ops.py (chiave SSH o HOME_SERVER_PASSWORD).
"""
import json
import sys
import time

from server_ops import connect

sys.stdout.reconfigure(encoding='utf-8')

scenarios = [
    ("INTENT_CHAT", "come stai ?"),
    ("INTENT_CHAT", "Spiegami la differenza tra TCP e UDP"),
    ("INTENT_SOC", "quanta RAM sta usando il server?"),
    ("INTENT_IMAGE", "Crea un'immagine di un gatto astronauta"),
    ("INTENT_PDF", "Crea un documento PDF sulle best practice di sicurezza Docker"),
    ("INTENT_WEB_SEARCH", "Cerca su internet le novità di Linux kernel 6.12 e fammi un riassunto"),
]

ssh = connect()
sftp = ssh.open_sftp()

for expected, prompt in scenarios:
    print("\n==========================================")
    print(f"Atteso: {expected}\nPrompt: {prompt}")
    print("==========================================")
    with sftp.open('/tmp/test_query.json', 'w') as f:
        f.write(json.dumps({"prompt": prompt, "messages": []}))

    t0 = time.time()
    cmd = 'curl -s -k -X POST https://127.0.0.1/api/ai-chat/c3po-query -H "Content-Type: application/json" -d @/tmp/test_query.json'
    stdin, stdout, stderr = ssh.exec_command(cmd)
    res_str = stdout.read().decode('utf-8')
    dur = time.time() - t0
    try:
        data = json.loads(res_str)
        intent = data.get('intent')
        flag = 'OK' if intent == expected else 'DIVERSO'
        print(f"Durata: {dur:.2f}s | intent={intent} [{flag}] | fastPath={data.get('fastPath')}")
        print((data.get('response') or '')[:600])
        if data.get('file'):
            print(f"File: {data['file'].get('url')} ({data['file'].get('sizeFormatted')})")
    except Exception as e:
        print(f"Risposta non JSON ({e}): {res_str[:500]}")

sftp.close()
ssh.close()
