/**
 * Fast-Path & Instant Telemetry Caching Engine
 * Bypasses LLM for standard recurring queries (< 15ms TTFT, 0 CPU thermal impact)
 *
 * Tutti i numeri vengono dalla telemetria reale: se un dato manca si scrive
 * "n/d", mai un valore plausibile inventato (un SOC che mostra "0 attacchi"
 * quando non ha letto i log e' peggio di uno che non risponde).
 */

const ND = 'n/d';

class FastPathEngine {
  constructor(telemetryCollector, socDb) {
    this.collector = telemetryCollector;
    this.db = socDb;
  }

  async telemetry() {
    try {
      return await this.collector.collectFullTelemetry();
    } catch (_) {
      return null;
    }
  }

  async checkFastPath(rawPrompt, username = 'Signore') {
    if (!rawPrompt || typeof rawPrompt !== 'string') return null;
    const p = rawPrompt.trim().toLowerCase();
    const formalTitle = (username && username.toLowerCase() !== 'utente') ? username : 'signore';

    // 1. STATO DEL SERVER / SALUTE / COME STA IL SERVER
    if (
      /^(?:come sta il server|stato (?:del )?server|salute (?:del )?server|come va il server|server health|health|stato salute|come sta la macchina)\??$/i.test(p) ||
      (/(?:stato|salute|condizioni).*(?:server|sistema|macchina)/i.test(p) && !/pdf|word|immagine|video|crea|genera/i.test(p))
    ) {
      const t = await this.telemetry();
      if (!t) return `Non riesco a leggere la telemetria in questo momento, ${formalTitle}. Verifica \`docker-proxy\` e riprova.`;
      const s = t.system || {};
      const c = t.containers || {};
      const threats = Array.isArray(t.ssh?.bruteForceSuspects) ? t.ssh.bruteForceSuspects.length : ND;
      const issues = Array.isArray(t.issues) && t.issues.length ? t.issues.join('; ') : 'nessuna';

      return `Ecco il bollettino in tempo reale del server (192.168.1.180), ${formalTitle}:

- 🩺 **Indice di Salute**: **${t.score ?? ND}/100** (${t.securityLevel || ND})
- 📦 **Container Docker**: **${c.running ?? ND}/${c.total ?? ND}** in esecuzione, ${c.unhealthy ?? 0} unhealthy
- 🌡️ **Temperatura CPU**: **${s.cpuTemp ?? ND}°C**
- 🧠 **Memoria RAM**: **${s.ramUsedFormatted || ND}** usati su ${s.ramTotalFormatted || ND} (${s.ramUsedPct ?? ND}%)
- 🛡️ **IP sospetti SSH**: **${threats}**

Criticità rilevate: ${issues}.`;
    }

    // 2. STATO CONTAINER DOCKER
    if (
      /^(?:quanti container|stato container|container attivi|quali container|elenco container|docker)\??$/i.test(p) ||
      (/(?:stato|quanti|lista|elenco).*(?:container|docker)/i.test(p) && !/pdf|word|crea/i.test(p))
    ) {
      const t = await this.telemetry();
      const c = t?.containers;
      if (!c) return `Non riesco a interrogare Docker in questo momento, ${formalTitle}.`;
      const list = Array.isArray(c.list) ? c.list : [];
      const problems = list.filter(x => x.state !== 'running' || x.unhealthy);
      const running = list.filter(x => x.state === 'running').map(x => `\`${x.name}\``);

      return `Container Docker: **${c.running}/${c.total} in esecuzione**, ${c.healthy ?? 0} healthy, ${c.unhealthy ?? 0} unhealthy.

${problems.length ? `⚠️ **Da controllare**: ${problems.map(x => `\`${x.name}\` (${x.unhealthy ? 'unhealthy' : x.state})`).join(', ')}\n\n` : ''}**In esecuzione**: ${running.join(', ') || ND}`;
    }

    // 3. USO RAM / CPU / TEMPERATURA
    if (
      /^(?:ram|memoria|cpu|temperatura|carico)\??$/i.test(p) ||
      (/(?:quanta|uso|consumo|stato).*(?:ram|memoria|cpu|temperatura|carico)/i.test(p) && !/pdf|word|crea/i.test(p))
    ) {
      const t = await this.telemetry();
      const s = t?.system;
      if (!s) return `Non riesco a leggere le metriche di sistema in questo momento, ${formalTitle}.`;
      const load = Array.isArray(s.loadAvg) ? s.loadAvg.map(l => l.toFixed(2)).join(', ') : ND;
      const thermal = typeof s.cpuTemp === 'number'
        ? (s.cpuTemp < 65 ? 'sotto la soglia di attenzione (65°C)' : 'sopra la soglia di attenzione: controlla ventilazione e `sensors`')
        : 'non disponibile';

      return `Metriche istantanee del server:

- 🌡️ **Temperatura CPU**: **${s.cpuTemp ?? ND}°C** (${thermal})
- ⚙️ **Load Average (1, 5, 15 min)**: \`${load}\` su 4 core
- 🧠 **RAM Usata**: **${s.ramUsedFormatted || ND}** / **${s.ramTotalFormatted || ND}** (${s.ramUsedPct ?? ND}%)`;
    }

    // 4. MINACCE / SICUREZZA / INTRUSIONI
    if (
      /^(?:minacce|sicurezza|intrusioni|allarmi)\??$/i.test(p) ||
      (/(?:ci sono|stato|controllo|rilevate).*(?:minacce|attacchi|intrusioni|allarmi|anomalie|incidenti)/i.test(p) && !/pdf|crea/i.test(p))
    ) {
      const t = await this.telemetry();
      if (!t) return `Non riesco a leggere log e telemetria di sicurezza in questo momento, ${formalTitle}: non posso confermare l'assenza di minacce.`;
      const ssh = t.ssh || {};
      const ngx = t.nginx || {};
      const suspects = Array.isArray(ssh.bruteForceSuspects) ? ssh.bruteForceSuspects : [];

      return `Rapporto di sicurezza dalla telemetria attuale, ${formalTitle}:

- 🚫 **Accessi SSH falliti**: **${ssh.failedAttempts ?? ND}**
- 🎯 **IP sospetti di brute-force**: **${suspects.length}**${suspects.length ? ` (${suspects.slice(0, 5).map(s => `\`${s.ip || s}\``).join(', ')})` : ''}
- 🌐 **Nginx**: ${ngx.totalRequests ?? ND} richieste, ${ngx.clientErrors4xx ?? ND} errori 4xx, ${ngx.serverErrors5xx ?? ND} errori 5xx
- 🩺 **Livello di sicurezza**: **${t.securityLevel || ND}** (${t.score ?? ND}/100)

${Array.isArray(t.issues) && t.issues.length ? `Criticità: ${t.issues.join('; ')}.` : 'Nessuna criticità registrata dalla telemetria.'}`;
    }

    // 5. IDENTITÀ C-3PO & CAPACITÀ
    if (/^(?:chi sei|cosa puoi fare|cosa sai fare|quali sono le tue funzioni|aiuto|help)\??$/i.test(p)) {
      return `Sono **C-3PO**, l'assistente di HOME-SERVER. Ecco cosa posso fare:

- 💬 **Conversazione e codice**: domande generali, spiegazioni, programmazione
- 🩺 **Diagnostica del server**: stato, container, CPU, RAM, sicurezza con dati reali
- 🌐 **Ricerca web**: *"cerca su internet le novità di Linux 6.12"*, con fonti citate
- 📄 **Documenti PDF e Word**: *"crea un PDF sulle best practice di sicurezza Docker"*
- 🎨 **Immagini**: *"disegna un gatto astronauta in stile acquerello"*

Come posso esservi utile, ${formalTitle}?`;
    }

    return null; // Not a fast-path pattern: route to LLM
  }
}

module.exports = FastPathEngine;
