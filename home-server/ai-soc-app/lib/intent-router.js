/**
 * Intent Router deterministico (zero LLM, < 1 ms)
 *
 * Un modello da 2B non sa scegliere tra cinque compiti descritti nello stesso
 * system prompt: li mescola e ricopia gli esempi. Qui la scelta la fa il
 * codice, e al modello arriva un solo compito con un prompt dedicato.
 *
 * Priorita': PDF/DOCX > IMAGE > WEB_SEARCH > SOC > CHAT
 */

const INTENTS = Object.freeze({
  CHAT: 'INTENT_CHAT',
  SOC: 'INTENT_SOC',
  IMAGE: 'INTENT_IMAGE',
  WEB_SEARCH: 'INTENT_WEB_SEARCH',
  PDF: 'INTENT_PDF'
});

// Il matching avviene su testo minuscolo e senza accenti ("novità" -> "novita")
function normalize(text) {
  return String(text || '')
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/[’`]/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

const CREATE_VERB = String.raw`(?:crea|creami|creare|genera|generami|generare|fai|fammi|fare|disegna|disegnami|illustra|illustrami|produci|realizza|realizzami|prepara|preparami|redigi|scrivi|scrivimi|esporta|stampa|voglio|vorrei|mi serve|puoi fare|potresti fare)`;

// --- PDF / DOCX -------------------------------------------------------------
const DOC_FORMAT_RE = /\b(pdf|docx|word)\b/;
const DOC_REQUEST_RE = new RegExp(String.raw`\b${CREATE_VERB}\b.{0,40}\b(?:pdf|docx|word|documento|report|relazione|dossier|guida|manuale|scheda tecnica)\b`);
const DOC_ANALYZE_RE = /\b(?:leggi|analizza|riassumi|apri|esamina|traduci)\b.{0,30}\b(?:pdf|documento|allegato|file)\b/;

// --- IMMAGINI ---------------------------------------------------------------
const IMAGE_NOUN = String.raw`(?:immagine|immagini|illustrazione|illustrazioni|foto|fotografia|disegno|disegni|logo|wallpaper|sfondo|ritratto|dipinto|quadro|rendering|render|poster|locandina|icona|avatar|banner|vignetta|fumetto)`;
const IMAGE_REQUEST_RE = new RegExp(String.raw`\b${CREATE_VERB}\b.{0,30}\b${IMAGE_NOUN}\b`);
const IMAGE_IMPERATIVE_RE = /^(?:c-?3po[ ,:]*)?(?:per favore |puoi |potresti )?(?:disegna|disegnami|illustra|illustrami|dipingi|dipingimi)\b/;
const IMAGE_ANALYZE_RE = /\b(?:analizza|descrivi|leggi|cosa (?:c'e|mostra)|guarda)\b.{0,30}\b(?:immagine|foto|screenshot)\b/;

// --- RICERCA WEB ------------------------------------------------------------
const URL_RE = /https?:\/\/[^\s<>"')]+/i;
// "cerca di spiegarmi" non e' una ricerca: negative lookahead su "di"
const SEARCH_VERB_RE = /\b(?:cerca|cercami|ricerca|googla|search)\b(?! di\b)/;
const FIND_ON_WEB_RE = /\b(?:trova|trovami|dimmi)\b.{0,40}\b(?:su internet|sul web|online|in rete|in internet|notizie|fonti)\b/;
const FRESHNESS_RE = /\b(?:ultime notizie|ultim[ae] novita|novita (?:su|di|del|della|dello|sul|sulla)|aggiornament[io] su|notizie (?:su|di|del|della|sul|sulla)|cosa e successo|che e successo|risultati? (?:della|di) (?:partita|elezioni)|prezzo attuale|quotazione|meteo|previsioni del tempo|uscit[oa] (?:oggi|ieri)|ultima versione|ultima release)\b/;

// --- SOC / TELEMETRIA -------------------------------------------------------
// Parole forti: bastano da sole. Parole deboli ("memoria", "sicurezza", "rete")
// hanno anche un senso comune: servono due indizi, oppure una domanda sullo stato.
const SOC_STRONG_RE = /\b(?:server|home-server|container|containers|docker|compose|cpu|ram|swap|uptime|logs?|nginx|fail2ban|firewall|iptables|ufw|ssh|sentinel|soc|telemetria|kernel|systemd|journalctl|n100|nipogi)\b/;
const SOC_WEAK_RE = /\b(?:memoria|disco|dischi|spazio|storage|temperatura|carico|load|sicurezza|attacc\w*|intrusion\w*|minacc\w*|alert|allarm\w*|anomal\w*|rete|porte?|ip|processi|servizi|sqlite|wal|backup)\b/g;
// Domande sullo stato *attuale* della macchina: solo queste ricevono la telemetria live
const LIVE_STATE_RE = /\b(?:stato|salute|come sta|come va|quant[aio]|quali|attiv[io]|in esecuzione|running|consum\w*|utilizz\w*|occupat\w*|libera|adesso|ora|ultim[ei]|oggi|problemi|errori|down|fermi|unhealthy|bloccat\w*|rilevat\w*|ci sono)\b/;

/**
 * Classifica una richiesta utente.
 * @param {string} prompt testo dell'utente
 * @param {object} [ctx] { hasAttachments }
 * @returns {{ intent: string, reason: string, params: object }}
 */
function classifyIntent(prompt, ctx = {}) {
  const raw = String(prompt || '').trim();
  const p = normalize(raw);

  if (!p) {
    return { intent: INTENTS.CHAT, reason: 'empty', params: {} };
  }

  // Con allegati si chiede quasi sempre di lavorare sull'allegato: niente tool
  if (ctx.hasAttachments) {
    return { intent: INTENTS.CHAT, reason: 'attachments', params: {} };
  }

  // 1. Documento PDF / DOCX
  if (!DOC_ANALYZE_RE.test(p) && DOC_REQUEST_RE.test(p) && (DOC_FORMAT_RE.test(p) || /\b(?:documento|report|relazione|dossier)\b/.test(p))) {
    const fmt = p.match(DOC_FORMAT_RE);
    const format = fmt && (fmt[1] === 'docx' || fmt[1] === 'word') ? 'docx' : 'pdf';
    return { intent: INTENTS.PDF, reason: 'doc-request', params: { format } };
  }

  // 2. Immagine
  if (!IMAGE_ANALYZE_RE.test(p) && (IMAGE_REQUEST_RE.test(p) || IMAGE_IMPERATIVE_RE.test(p))) {
    return { intent: INTENTS.IMAGE, reason: 'image-request', params: {} };
  }

  // 3. Ricerca web / URL esplicito
  const urlMatch = raw.match(URL_RE);
  if (urlMatch) {
    return { intent: INTENTS.WEB_SEARCH, reason: 'explicit-url', params: { url: urlMatch[0].replace(/[.,;:!?]+$/, '') } };
  }
  if (SEARCH_VERB_RE.test(p) || FIND_ON_WEB_RE.test(p) || FRESHNESS_RE.test(p)) {
    return { intent: INTENTS.WEB_SEARCH, reason: 'search-request', params: {} };
  }

  // 4. SOC / diagnostica del server
  const liveState = LIVE_STATE_RE.test(p);
  const weakHits = (p.match(SOC_WEAK_RE) || []).length;
  if (SOC_STRONG_RE.test(p) || weakHits >= 2 || (weakHits === 1 && liveState)) {
    return { intent: INTENTS.SOC, reason: 'soc-keyword', params: { needsTelemetry: liveState } };
  }

  // 5. Conversazione libera (saluti, cultura generale, codice)
  return { intent: INTENTS.CHAT, reason: 'default', params: {} };
}

/**
 * Estrae la query di ricerca togliendo verbi e formule di cortesia.
 */
function extractSearchQuery(prompt) {
  const cleaned = String(prompt || '')
    .replace(URL_RE, ' ')
    .replace(/^\s*c-?3po[\s,:]*/i, '')
    // coda "e fammi un riassunto", ", poi spiegami..."
    .replace(/[\s,]+(?:e\s+|poi\s+)?(?:fammi|dammi|scrivi|scrivimi|fai|riassumi|riassumimi|spiegami)\b[^]*$/i, '')
    .replace(/\b(?:per favore|per piacere|grazie)\b/gi, ' ')
    .replace(/^\s*(?:puoi |potresti )?(?:cerca|cercami|ricerca|googla|trova|trovami|dimmi)\b/i, ' ')
    .replace(/\b(?:su internet|sul web|online|in rete|in internet)\b/gi, ' ')
    .replace(/[?!]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return cleaned || String(prompt || '').trim();
}

/**
 * Estrae l'argomento di un documento ("crea un PDF sulle best practice Docker"
 * -> "best practice Docker").
 */
function extractDocTopic(prompt) {
  let t = String(prompt || '')
    .replace(/^\s*c-?3po[\s,:]*/i, '')
    .replace(/\b(?:per favore|per piacere|grazie)\b/gi, ' ');
  // Tutto cio' che segue "su/sul/sulle/riguardo/intitolato/per" dopo il nome del documento
  const m = t.match(/\b(?:pdf|docx|word|documento|report|relazione|dossier|guida|manuale|scheda tecnica)\b[^]*?\b(?:su|sul|sulla|sullo|sui|sugli|sulle|riguardo(?: a| al| alla| ai| agli| alle)?|intitolat[oa]|dal titolo|per|circa|relativ[oa] (?:a|al|alla|ai|agli|alle)|che (?:spieghi|descriva|tratti))\s+([^]+)$/i);
  if (m) {
    t = m[1];
  } else {
    t = t
      .replace(new RegExp(String.raw`\b${CREATE_VERB}\b`, 'gi'), ' ')
      .replace(/\b(?:un|uno|una|il|lo|la|i|gli|le|file|in|formato|documento|report|pdf|docx|word)\b/gi, ' ');
  }
  t = t.replace(/[."'«»]+/g, ' ').replace(/\s+/g, ' ').trim();
  return t ? t.charAt(0).toUpperCase() + t.slice(1, 120) : 'Documento tecnico';
}

module.exports = { INTENTS, classifyIntent, extractSearchQuery, extractDocTopic, normalize };
