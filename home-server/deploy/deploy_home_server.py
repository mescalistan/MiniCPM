# -*- coding: utf-8 -*-
"""
Deploy dei moduli C-3PO su HOME-SERVER (Intel N100).

Cosa fa, in ordine:
  1. copia i file di home-server/ai-soc-app/lib in ai-soc-app/lib sul server
     (backup .bak-<timestamp> di ogni file sovrascritto)
  2. patcha ai-soc-app/server.js: il vecchio handler /api/ai-chat/c3po-query
     viene sostituito dalla chiamata a C3poOrchestrator
  3. patcha ai-soc-app/lib/media-studio.js: timeout Pollinations 5s -> 30s,
     niente stile "luxury dark" forzato su ogni immagine
  4. aggiunge cheerio a ai-soc-app/package.json
  5. riavvia ai-soc-app (oppure ricostruisce l'immagine con --rebuild)

Credenziali: chiave SSH (~/.ssh) oppure variabile d'ambiente HOME_SERVER_PASSWORD.
Nessuna password nel codice.

Uso:
  # prova a secco su copie locali dei file del server (nessuna connessione)
  python home-server/deploy/deploy_home_server.py --offline <cartella-con-copie>
  # deploy vero
  python home-server/deploy/deploy_home_server.py --rebuild
"""

import argparse
import json
import os
import posixpath
import sys
import time
from pathlib import Path

HOST = os.environ.get("HOME_SERVER_HOST", "192.168.1.180")
USER = os.environ.get("HOME_SERVER_USER", "gabibbo")
REMOTE_ROOT = os.environ.get("HOME_SERVER_ROOT", "/home/gabibbo/HOME-SERVER")

LOCAL_LIB = Path(__file__).resolve().parent.parent / "ai-soc-app" / "lib"
LIB_FILES = [
    "prompts.json",
    "ai-engine-client.js",
    "intent-router.js",
    "prompt-builder.js",
    "image-intent.js",
    "web-research-service.js",
    "markdown-pdf.js",
    "c3po-orchestrator.js",
    "ai-chat-service.js",
    "ollama-service.js",
    "fast-path.js",
]

NEW_C3PO_HANDLER = """app.post(['/api/ai-chat/c3po-query', '/api/soc/chat/c3po'], authenticate, async (req, res) => {
  try {
    const { username, messages, prompt, attachments } = req.body || {};
    let userPrompt = String(prompt || '').trim();
    if (!userPrompt && Array.isArray(messages) && messages.length > 0) {
      const lastUser = [...messages].reverse().find(m => m.role === 'user');
      if (lastUser) userPrompt = String(lastUser.content || '').trim();
    }
    if (!userPrompt && (!Array.isArray(attachments) || attachments.length === 0)) {
      return res.status(400).json({ error: 'Prompt o allegato obbligatorio' });
    }

    // Router deterministico + prompt dedicato per intent (lib/c3po-orchestrator.js)
    const result = await aiChatService.orchestrator.run({
      prompt: userPrompt,
      history: Array.isArray(messages) ? messages : [],
      attachments: Array.isArray(attachments) ? attachments : [],
      username: username || req.user?.username || 'utente'
    });

    res.json({
      success: !result.error,
      response: result.response,
      reply: result.response,
      intent: result.intent,
      file: result.file || null,
      sources: result.sources || [],
      fastPath: Boolean(result.fastPath)
    });
  } catch (err) {
    console.error('[ai-soc-app] Errore c3po-query:', err);
    res.status(500).json({ error: 'Errore generazione C-3PO: ' + err.message });
  }
});

"""


class PatchError(RuntimeError):
    pass


def replace_once(code: str, old: str, new: str, label: str) -> str:
    count = code.count(old)
    if count != 1:
        raise PatchError(f"{label}: ancora trovata {count} volte (attesa 1). Il file sul server e' cambiato?")
    return code.replace(old, new)


def patch_server_js(code: str) -> str:
    if "aiChatService.orchestrator.run(" in code:
        print("  server.js: handler gia' aggiornato, salto")
        return code

    # Telemetria e fast-path anche per l'orchestratore interno di AiChatService
    code = replace_once(
        code,
        "const aiChatService = new AiChatService(socDb, {\n",
        "const aiChatService = new AiChatService(socDb, {\n  telemetryCollector,\n  fastPath: fastPathEngine,\n",
        "costruttore AiChatService",
    )

    start = code.find("app.post(['/api/ai-chat/c3po-query'")
    end_label = code.find("// REPORT NOTIFICATION SETTINGS")
    if start == -1 or end_label == -1:
        raise PatchError("server.js: marker del handler c3po-query non trovati")
    end = code.rfind("// ---", start, end_label)
    if end == -1:
        raise PatchError("server.js: separatore prima di REPORT NOTIFICATION SETTINGS non trovato")
    return code[:start] + NEW_C3PO_HANDLER + code[end:]


def patch_media_studio(code: str) -> str:
    if "AbortSignal.timeout(30000)" in code:
        print("  media-studio.js: gia' aggiornato, salto")
        return code
    # Pollinations impiega 5-20 s: con 5 s quasi ogni immagine finiva nel fallback SVG
    code = replace_once(code, "signal: AbortSignal.timeout(5000)", "signal: AbortSignal.timeout(30000)", "timeout fetch Pollinations")
    code = replace_once(
        code,
        "this.withTimeout(this._executeGenerateImage(params), 15000, 'Generazione Immagine')",
        "this.withTimeout(this._executeGenerateImage(params), 40000, 'Generazione Immagine')",
        "timeout generateImage",
    )
    # Lo stile lo decide l'orchestratore (image-intent.js), non un suffisso fisso
    code = replace_once(
        code,
        "`${cleanPrompt}, masterpiece, high quality, luxury dark aesthetic, detailed`",
        "/detailed|high quality/i.test(cleanPrompt) ? cleanPrompt : `${cleanPrompt}, high quality, detailed`",
        "prompt arricchito Pollinations",
    )
    return code


def patch_package_json(text: str) -> str:
    data = json.loads(text)
    deps = data.setdefault("dependencies", {})
    if "cheerio" in deps:
        print("  package.json: cheerio gia' presente, salto")
        return text
    deps["cheerio"] = "^1.0.0"
    data["dependencies"] = dict(sorted(deps.items()))
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


PATCHES = {
    "ai-soc-app/server.js": patch_server_js,
    "ai-soc-app/lib/media-studio.js": patch_media_studio,
    "ai-soc-app/package.json": patch_package_json,
}


def run_offline(directory: Path) -> None:
    """Applica le patch a copie locali (nomi con '/' sostituito da '__')."""
    for rel, fn in PATCHES.items():
        src = directory / rel.replace("/", "__")
        if not src.exists():
            print(f"  {rel}: copia locale assente ({src.name}), salto")
            continue
        patched = fn(src.read_text(encoding="utf-8"))
        out = src.with_suffix(src.suffix + ".patched")
        out.write_text(patched, encoding="utf-8", newline="\n")
        print(f"  {rel}: OK -> {out.name}")


def connect():
    import paramiko

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.WarningPolicy())
    password = os.environ.get("HOME_SERVER_PASSWORD")
    ssh.connect(HOST, username=USER, password=password, look_for_keys=True, allow_agent=True, timeout=15)
    return ssh


def run(ssh, cmd: str) -> str:
    _, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(f"Comando fallito ({code}): {cmd}\n{err or out}")
    return out


def deploy(rebuild: bool) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ssh = connect()
    sftp = ssh.open_sftp()

    dirty = run(ssh, f"git -C {REMOTE_ROOT} status --porcelain 2>/dev/null || true").strip()
    if dirty:
        print("ATTENZIONE: HOME-SERVER ha modifiche locali non committate; un `git pull` in avvia_server.sh")
        print("potrebbe rifiutarsi di sovrascriverle. Committa queste modifiche nel repo HOME-SERVER.\n")

    def backup(remote: str) -> None:
        run(ssh, f"[ -f '{remote}' ] && cp -p '{remote}' '{remote}.bak-{stamp}' || true")

    print("[1/4] Moduli lib/")
    for name in LIB_FILES:
        remote = posixpath.join(REMOTE_ROOT, "ai-soc-app/lib", name)
        backup(remote)
        sftp.put(str(LOCAL_LIB / name), remote)
        print(f"  {name}")

    print("[2/4] Patch server.js, media-studio.js, package.json")
    for rel, fn in PATCHES.items():
        remote = posixpath.join(REMOTE_ROOT, rel)
        with sftp.open(remote, "r") as f:
            original = f.read().decode("utf-8")
        patched = fn(original)
        if patched != original:
            backup(remote)
            with sftp.open(remote, "w") as f:
                f.write(patched.encode("utf-8"))
            print(f"  {rel}: aggiornato")

    print("[3/4] Verifica sintassi nel container")
    files = " ".join(f"lib/{n}" for n in LIB_FILES if n.endswith(".js"))
    run(ssh, f"cd {REMOTE_ROOT}/ai-soc-app && docker run --rm -v \"$PWD\":/app -w /app node:22-alpine sh -c 'for f in server.js {files}; do node --check $f || exit 1; done'")

    print("[4/4] Riavvio ai-soc-app")
    if rebuild:
        print(run(ssh, f"cd {REMOTE_ROOT} && docker compose build ai-soc-app && docker compose up -d ai-soc-app"))
    else:
        print(run(ssh, "docker restart ai-soc-app"))
    print(f"Fatto. Backup con suffisso .bak-{stamp} accanto a ogni file modificato.")

    sftp.close()
    ssh.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", type=Path, help="applica le patch a copie locali, senza connessione")
    parser.add_argument("--rebuild", action="store_true", help="ricostruisce l'immagine ai-soc-app (serve per installare cheerio)")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    if args.offline:
        run_offline(args.offline)
    else:
        deploy(args.rebuild)


if __name__ == "__main__":
    main()
