# -*- coding: utf-8 -*-
"""
Esegue un comando sul server HOME-SERVER via SSH.

Credenziali: chiave SSH (consigliata, `ssh-copy-id gabibbo@192.168.1.180`)
oppure variabile d'ambiente HOME_SERVER_PASSWORD. Nessuna password nel codice.

Uso: python server_ops.py "docker ps"
"""
import os
import sys

import paramiko

HOST = os.environ.get("HOME_SERVER_HOST", "192.168.1.180")
USER = os.environ.get("HOME_SERVER_USER", "gabibbo")


def connect():
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
    ssh.connect(HOST, username=USER, password=os.environ.get("HOME_SERVER_PASSWORD"),
                look_for_keys=True, allow_agent=True, timeout=15)
    return ssh


def run_ssh(cmd):
    ssh = connect()
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
