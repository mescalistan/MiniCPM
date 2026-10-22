import sys, paramiko

def run_ssh(cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.1.180', username='gabibbo', password='225569')
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read()
    err = stderr.read()
    ssh.close()
    return out.decode('utf-8', errors='replace'), err.decode('utf-8', errors='replace')

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'echo ok'
    out, err = run_ssh(cmd)
    if out: sys.stdout.buffer.write(out.encode('utf-8'))
    if err: sys.stderr.buffer.write(err.encode('utf-8'))
