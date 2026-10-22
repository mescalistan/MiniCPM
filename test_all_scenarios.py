import paramiko, json, time, sys

sys.stdout.reconfigure(encoding='utf-8')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.1.180', username='gabibbo', password='225569')

scenarios = [
    ("Scenario 1 (CHAT)", "Ciao, come stai?"),
    ("Scenario 2 (IMAGE_GEN)", "Crea un'immagine di un gatto astronauta"),
    ("Scenario 3 (PDF_GEN)", "Crea un documento PDF sulle best practice di sicurezza Docker"),
    ("Scenario 4 (WEB_SEARCH)", "Cerca su internet le novità di Linux kernel 6.12 e fammi un riassunto")
]

sftp = ssh.open_sftp()

for title, prompt in scenarios:
    print(f"\n==========================================")
    print(f"Testing: {title}")
    print(f"Prompt: {prompt}")
    print(f"==========================================")
    payload = json.dumps({"prompt": prompt, "messages": []})
    with sftp.open('/tmp/test_query.json', 'w') as f:
        f.write(payload)

    t0 = time.time()
    cmd = 'curl -s -k -X POST https://127.0.0.1/api/ai-chat/c3po-query -H "Content-Type: application/json" -d @/tmp/test_query.json'
    stdin, stdout, stderr = ssh.exec_command(cmd)
    res_str = stdout.read().decode('utf-8')
    err_str = stderr.read().decode('utf-8')
    dur = time.time() - t0
    print(f"Duration: {dur:.2f}s")
    try:
        data = json.loads(res_str)
        ans = data.get('response') or data.get('reply') or ''
        print(f"Success: {data.get('success')}")
        print(f"Response Preview:\n{ans[:600]}")
        if data.get('file'):
            print(f"File attached: {data.get('file')}")
    except Exception as e:
        print(f"Raw Output: {res_str[:500]}")
        print(f"Error: {e}")

sftp.close()
ssh.close()

