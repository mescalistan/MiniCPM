/**
 * Prompt Builder per MiniCPM-2B
 *
 * Regole che valgono per ogni intent:
 * 1. Il system prompt e' statico (prompts.json): stesso prefisso a ogni
 *    richiesta = KV riusato da llama-server (cache_prompt / --cache-reuse).
 * 2. I dati dinamici (telemetria, fonti web, allegati) vanno nel turno
 *    `user`, prima della domanda, in formato chiave=valore compatto.
 * 3. La history viene ripulita: le vecchie risposte con immagini, link a file
 *    e card sono la fonte del "prompt leakage" (il 2B le ricopia alla lettera).
 */

const SYSTEM_PROMPTS = require('./prompts.json');

const HISTORY_LIMITS = {
  INTENT_CHAT: { maxMessages: 6, maxChars: 600 },
  INTENT_SOC: { maxMessages: 2, maxChars: 400 },
  // RAG, documenti e immagini sono single-turn: niente history
  INTENT_WEB_SEARCH: { maxMessages: 0, maxChars: 0 },
  INTENT_PDF: { maxMessages: 0, maxChars: 0 },
  INTENT_IMAGE: { maxMessages: 0, maxChars: 0 }
};

/**
 * Toglie da una risposta passata tutto cio' che il modello non deve imitare:
 * immagini markdown, link a media generati, HTML, blocchi "Fonti consultate".
 */
function stripArtifacts(text) {
  return String(text || '')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')                          // ![alt](url)
    .replace(/\[([^\]]*)\]\((?:\/api\/|https?:\/\/)[^)]*\)/g, '$1')  // [testo](url) -> testo
    .replace(/<video[\s\S]*?<\/video>/gi, '')
    .replace(/<[^>]+>/g, '')
    .replace(/\*\*Fonti consultate:?\*\*[\s\S]*$/i, '')
    .replace(/^.*(?:📄|🎨|🎥|📝)\s*\*\*\[?(?:Scarica|Visualizza)[^\n]*$/gim, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/**
 * Ripulisce la history del frontend.
 * - solo ruoli user/assistant, niente errori o messaggi di sistema
 * - risposte "tool" (immagini, file) eliminate: non sono conversazione
 * - ruoli alternati, si parte da un turno user, niente duplicato del prompt
 */
function sanitizeHistory(history, { intent = 'INTENT_CHAT', currentPrompt = '' } = {}) {
  const limits = HISTORY_LIMITS[intent] || HISTORY_LIMITS.INTENT_CHAT;
  if (!Array.isArray(history) || limits.maxMessages === 0) return [];

  const cleaned = [];
  for (const msg of history) {
    if (!msg || typeof msg.content !== 'string') continue;
    const role = msg.role === 'user' ? 'user' : (msg.role === 'assistant' ? 'assistant' : null);
    if (!role) continue;
    if (/^(?:⚠️|\[Errore|Errore )/.test(msg.content.trim())) continue;
    if (msg.fastPath || msg.file || msg.toolResult) continue;

    let content = role === 'assistant' ? stripArtifacts(msg.content) : msg.content.trim();
    if (!content) continue;
    if (content.length > limits.maxChars) content = content.slice(0, limits.maxChars) + ' […]';

    const last = cleaned[cleaned.length - 1];
    if (last && last.role === role) {
      last.content = `${last.content}\n${content}`.slice(0, limits.maxChars * 2);
    } else {
      cleaned.push({ role, content });
    }
  }

  // Il frontend spesso accoda gia' il prompt corrente: lo toglie per non duplicarlo
  const tail = cleaned[cleaned.length - 1];
  if (tail && tail.role === 'user' && tail.content.trim() === String(currentPrompt).trim()) {
    cleaned.pop();
  }
  // Un turno user senza risposta in coda romperebbe l'alternanza
  if (cleaned.length && cleaned[cleaned.length - 1].role === 'user') cleaned.pop();

  let window = cleaned.slice(-limits.maxMessages);
  while (window.length && window[0].role !== 'user') window = window.slice(1);
  return window;
}

function fmtNum(n, digits = 1) {
  return typeof n === 'number' && Number.isFinite(n) ? n.toFixed(digits) : 'n/d';
}

/**
 * Telemetria in una riga chiave=valore (~60 token invece di ~250 in prosa).
 * Accetta l'output di TelemetryCollector.collectFullTelemetry().
 */
function compactTelemetry(t) {
  if (!t) return 'telemetria=non_disponibile';
  const s = t.system || {};
  const c = t.containers || {};
  const ssh = t.ssh || {};
  const ngx = t.nginx || {};
  const load = Array.isArray(s.loadAvg) ? s.loadAvg.map(v => fmtNum(v, 2)).join('/') : 'n/d';
  const down = Array.isArray(c.list)
    ? c.list.filter(x => x.state !== 'running' || x.unhealthy).map(x => `${x.name}:${x.unhealthy ? 'unhealthy' : x.state}`).slice(0, 6)
    : [];
  const issues = Array.isArray(t.issues) && t.issues.length ? t.issues.slice(0, 4).join('; ') : 'nessuna';

  return [
    `ora=${new Date(t.timestamp || Date.now()).toISOString().slice(0, 16)}`,
    `score=${t.score ?? 'n/d'}/100 (${t.securityLevel || 'n/d'})`,
    `cpu_temp=${s.cpuTemp ?? 'n/d'}C load=${load}`,
    `ram=${s.ramUsedFormatted || 'n/d'}/${s.ramTotalFormatted || 'n/d'} (${s.ramUsedPct ?? 'n/d'}%)`,
    `container=${c.running ?? 'n/d'}/${c.total ?? 'n/d'} unhealthy=${c.unhealthy ?? 0}${down.length ? ` problemi=[${down.join(', ')}]` : ''}`,
    `ssh_falliti=${ssh.failedAttempts ?? 0} ip_sospetti=${Array.isArray(ssh.bruteForceSuspects) ? ssh.bruteForceSuspects.length : 0}`,
    `nginx_4xx=${ngx.clientErrors4xx ?? 0} nginx_5xx=${ngx.serverErrors5xx ?? 0}`,
    `criticita=${issues}`
  ].join('\n');
}

/**
 * Turno user con il contesto dinamico prima della domanda.
 */
function buildUserTurn(prompt, { telemetry = null, sources = null, extraContext = '' } = {}) {
  const parts = [];
  if (telemetry) parts.push(`[TELEMETRIA]\n${telemetry}`);
  if (sources) parts.push(`[FONTI]\n${sources}`);
  if (extraContext) parts.push(`[CONTESTO]\n${extraContext}`);
  if (!parts.length) return String(prompt).trim();
  parts.push(`[DOMANDA]\n${String(prompt).trim()}`);
  return parts.join('\n\n');
}

/**
 * Messaggi completi per /v1/chat/completions.
 * @param {string} kind chiave di prompts.json (chat, soc, rag, doc, translate)
 */
function buildMessages(kind, userContent, history = []) {
  const system = SYSTEM_PROMPTS[kind] || SYSTEM_PROMPTS.chat;
  return [
    { role: 'system', content: system },
    ...history,
    { role: 'user', content: userContent }
  ];
}

module.exports = {
  SYSTEM_PROMPTS,
  sanitizeHistory,
  stripArtifacts,
  compactTelemetry,
  buildUserTurn,
  buildMessages
};
