/**
 * C-3PO Orchestrator: un solo punto di ingresso per chat-app (JSON) e per
 * l'assistente web (SSE).
 *
 *   prompt ─► classifyIntent() ─┬─ INTENT_IMAGE      traduzione + Pollinations, risposta a template
 *                               ├─ INTENT_PDF        Markdown dal modello -> markdown-pdf.js
 *                               ├─ INTENT_WEB_SEARCH ricerca + scraping -> RAG con fonti numerate
 *                               ├─ INTENT_SOC        telemetria chiave=valore nel turno user
 *                               └─ INTENT_CHAT       prompt pulito, history ripulita
 *
 * Il modello riceve sempre un solo compito e un system prompt statico.
 */

const fs = require('fs');
const path = require('path');

const { INTENTS, classifyIntent, extractSearchQuery, extractDocTopic } = require('./intent-router');
const { sanitizeHistory, compactTelemetry, buildUserTurn, buildMessages } = require('./prompt-builder');
const { extractImageSubject, detectStyle, translateSubject, buildImagePrompt } = require('./image-intent');
const { renderMarkdownPdf, normalizeModelMarkdown, parseMarkdown, blocksToSections } = require('./markdown-pdf');

const TELEMETRY_TTL_MS = 10000;
const MAX_EXTRA_CONTEXT = 2400;   // slot da 2048 token: oltre questo l'allegato non ci sta

class C3poOrchestrator {
  /**
   * @param {object} deps
   * @param {import('./ai-engine-client')} deps.engine
   * @param {object} [deps.telemetryCollector] collectFullTelemetry()
   * @param {object} [deps.mediaStudio] generateImage(), createWord(), outputDir
   * @param {object} [deps.webResearch] searchAndScrape(), formatSources()
   * @param {object} [deps.fastPath] checkFastPath()
   * @param {object} [deps.db] saveMediaItem()
   */
  constructor({ engine, telemetryCollector = null, mediaStudio = null, webResearch = null, fastPath = null, db = null }) {
    this.engine = engine;
    this.telemetryCollector = telemetryCollector;
    this.mediaStudio = mediaStudio;
    this.webResearch = webResearch;
    this.fastPath = fastPath;
    this.db = db;
    this.telemetryCache = { at: 0, value: null };
  }

  // ------------------------------------------------------------------------
  // Contesto
  // ------------------------------------------------------------------------
  async getTelemetry() {
    if (!this.telemetryCollector) return null;
    if (Date.now() - this.telemetryCache.at < TELEMETRY_TTL_MS) return this.telemetryCache.value;
    try {
      const value = await Promise.race([
        this.telemetryCollector.collectFullTelemetry(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout telemetria')), 3000))
      ]);
      this.telemetryCache = { at: Date.now(), value };
      return value;
    } catch (err) {
      console.warn('[C3PO] Telemetria non disponibile:', err.message);
      return null;
    }
  }

  static attachmentsToContext(attachments = []) {
    if (!Array.isArray(attachments) || !attachments.length) return '';
    let ctx = '';
    for (const att of attachments) {
      const name = att.filename || 'allegato';
      const text = att.text || att.content || '';
      if (text) ctx += `--- ${name} ---\n${text}\n`;
    }
    return ctx.length > MAX_EXTRA_CONTEXT ? `${ctx.slice(0, MAX_EXTRA_CONTEXT)}\n[…allegato troncato]` : ctx;
  }

  /**
   * Classifica e prepara la richiesta.
   * Restituisce `{ intent, done: true, result }` per gli intent risolti senza
   * streaming (immagine, documento, fast-path) oppure `{ intent, messages,
   * profile, suffix, meta }` da mandare al modello.
   */
  async prepare({ prompt, history = [], attachments = [], extraContext = '', username = 'utente', isLive = false }) {
    const userPrompt = String(prompt || '').trim();
    let attachmentCtx = [extraContext, C3poOrchestrator.attachmentsToContext(attachments)].filter(Boolean).join('\n');
    if (attachmentCtx.length > MAX_EXTRA_CONTEXT) {
      attachmentCtx = `${attachmentCtx.slice(0, MAX_EXTRA_CONTEXT)}\n[…contenuto troncato]`;
    }
    const route = classifyIntent(userPrompt, { hasAttachments: Boolean(attachmentCtx) });
    const { intent, params } = route;

    // Fast-path deterministico (stato server, container, chi sei) senza LLM
    if (this.fastPath && !attachmentCtx && (intent === INTENTS.SOC || intent === INTENTS.CHAT)) {
      const fast = await this.fastPath.checkFastPath(userPrompt, username).catch(() => null);
      if (fast) return { intent, done: true, result: { intent, response: fast, fastPath: true } };
    }

    if (intent === INTENTS.IMAGE) {
      return { intent, done: true, result: await this.handleImage(userPrompt, username) };
    }
    if (intent === INTENTS.PDF) {
      return { intent, done: true, result: await this.handleDocument(userPrompt, username, params.format) };
    }

    if (intent === INTENTS.WEB_SEARCH) {
      const query = extractSearchQuery(userPrompt);
      let research = { sources: [], context: 'Nessuna fonte disponibile.' };
      if (this.webResearch) {
        try {
          research = await this.webResearch.searchAndScrape(query, { maxPages: 3, explicitUrl: params.url || null });
        } catch (err) {
          research.context = `Ricerca non riuscita: ${err.message}`;
        }
      }
      if (!research.sources.length) {
        const response = `Non sono riuscito a trovare fonti web per "${query}". Riprova tra poco o riformula la richiesta.`;
        return { intent, done: true, result: { intent, response, sources: [] } };
      }
      return {
        intent,
        profile: 'rag',
        messages: buildMessages('rag', buildUserTurn(userPrompt, { sources: research.context })),
        suffix: this.webResearch.formatSources(research.sources),
        meta: { sources: research.sources.map(({ n, title, url }) => ({ n, title, url })) }
      };
    }

    if (intent === INTENTS.SOC) {
      const telemetry = params.needsTelemetry ? compactTelemetry(await this.getTelemetry()) : null;
      const cleanHistory = sanitizeHistory(history, { intent, currentPrompt: userPrompt });
      return {
        intent,
        profile: 'soc',
        messages: buildMessages('soc', buildUserTurn(userPrompt, { telemetry }), cleanHistory),
        suffix: '',
        meta: { telemetry: Boolean(telemetry) }
      };
    }

    // INTENT_CHAT
    const cleanHistory = sanitizeHistory(history, { intent: INTENTS.CHAT, currentPrompt: userPrompt });
    return {
      intent,
      profile: isLive ? 'live' : 'chat',
      messages: buildMessages(isLive ? 'live' : 'chat', buildUserTurn(userPrompt, { extraContext: attachmentCtx }), cleanHistory),
      suffix: '',
      meta: {}
    };
  }

  // ------------------------------------------------------------------------
  // INTENT_IMAGE
  // ------------------------------------------------------------------------
  async handleImage(prompt, username) {
    const intent = INTENTS.IMAGE;
    if (!this.mediaStudio) {
      return { intent, response: 'La generazione di immagini non e\' disponibile su questo server.' };
    }
    const subject = extractImageSubject(prompt);
    const style = detectStyle(prompt);
    const { english, method } = await translateSubject(subject, this.engine);
    const imagePrompt = buildImagePrompt(english, style);

    try {
      const file = await this.mediaStudio.generateImage({
        prompt: imagePrompt,
        title: subject.slice(0, 60),
        userId: username
      });
      const response = `Ecco l'immagine richiesta: **${subject}**\n\n![${subject}](${file.url})\n\n🎨 **[Visualizza / Scarica immagine](${file.url})**`;
      return { intent, response, file, imagePrompt, translation: method };
    } catch (err) {
      console.warn('[C3PO] Generazione immagine fallita:', err.message);
      return { intent, response: `Non sono riuscito a generare l'immagine di **${subject}**: ${err.message}.`, imagePrompt };
    }
  }

  // ------------------------------------------------------------------------
  // INTENT_PDF (e DOCX)
  // ------------------------------------------------------------------------
  async handleDocument(prompt, username, format = 'pdf') {
    const intent = INTENTS.PDF;
    const topic = extractDocTopic(prompt);

    // Report sullo stato del server: il modello riceve la telemetria vera
    const wantsTelemetry = /\b(server|container|telemetria|salute|stato del sistema|sicurezza del server|soc)\b/i.test(topic);
    const telemetry = wantsTelemetry ? compactTelemetry(await this.getTelemetry()) : null;

    const userTurn = buildUserTurn(
      `Argomento: ${topic}\nRichiesta: ${prompt}\nLunghezza: 350-550 parole. Usa una tabella se ci sono dati da confrontare.`,
      { telemetry }
    );

    let markdown;
    try {
      markdown = await this.engine.chatCompletion({ messages: buildMessages('doc', userTurn), profile: 'doc' });
    } catch (err) {
      return { intent, response: `Non sono riuscito a redigere il documento su **${topic}**: il motore AI non ha risposto (${err.message}).` };
    }
    markdown = normalizeModelMarkdown(markdown, topic);
    if (markdown.replace(/[#\s*-]/g, '').length < 120) {
      return { intent, response: `Il contenuto generato per **${topic}** era troppo breve per un documento. Riprova specificando meglio l'argomento.` };
    }

    const title = topic.length > 90 ? `${topic.slice(0, 87)}...` : topic;
    const subtitle = 'Documento tecnico • HOME-SERVER AI Studio';
    const author = `C-3PO per ${username}`;
    const blocks = parseMarkdown(markdown);
    const sections = blocksToSections(blocks);

    let file;
    try {
      file = format === 'docx' && this.mediaStudio?.createWord
        ? await this.mediaStudio.createWord({ title, subtitle, author, content: '', sections, userId: username })
        : await this.writePdf({ title, subtitle, author, markdown, sections, username });
    } catch (err) {
      console.warn('[C3PO] Rendering documento fallito:', err.message);
      return { intent, response: `${markdown}\n\n⚠️ Non sono riuscito a creare il file ${format.toUpperCase()}: ${err.message}` };
    }

    const summary = sections.find(s => /sommario|sintesi/i.test(s.heading))?.text || sections[0]?.text || '';
    const outline = sections.map(s => s.heading).filter(Boolean).slice(0, 8).join(' · ');
    const icon = format === 'docx' ? '📝' : '📄';
    const response = [
      `${icon} Documento pronto: **${title}**${file.pages ? ` (${file.pages} pagine)` : ''}.`,
      summary ? `\n${summary.slice(0, 500)}` : '',
      outline ? `\n**Sezioni:** ${outline}` : '',
      `\n${icon} **[Scarica ${format.toUpperCase()}: ${title}](${file.url})** (${file.sizeFormatted})`
    ].join('\n');
    return { intent, response, file, markdown };
  }

  async writePdf({ title, subtitle, author, markdown, sections, username }) {
    const outputDir = this.mediaStudio?.outputDir || path.join(__dirname, '..', 'data', 'generated');
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
    const filename = `doc-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}.pdf`;
    const outputPath = path.join(outputDir, filename);

    const { pages } = await renderMarkdownPdf({ markdown, title, subtitle, author, outputPath });
    const stat = fs.statSync(outputPath);

    // Sidecar per l'anteprima nel browser (/api/ai-chat/media-preview)
    try {
      fs.writeFileSync(`${outputPath}.meta.json`, JSON.stringify({
        title, subtitle, author, content: markdown, sections, tables: [], type: 'pdf', generatedAt: new Date().toISOString()
      }, null, 2));
    } catch (_) {}
    const db = this.db || this.mediaStudio?.db;
    if (db && typeof db.saveMediaItem === 'function') {
      try { db.saveMediaItem({ filename, fileType: 'pdf', filePath: outputPath, size: stat.size, userId: username }); } catch (_) {}
    }

    return {
      filename,
      path: outputPath,
      url: `/api/ai-chat/media/${filename}`,
      size: stat.size,
      sizeFormatted: `${(stat.size / 1024).toFixed(1)} KB`,
      type: 'pdf',
      title,
      pages
    };
  }

  // ------------------------------------------------------------------------
  // API pubbliche
  // ------------------------------------------------------------------------

  /**
   * Risposta JSON completa (chat-app, /api/ai-chat/c3po-query).
   * @returns {Promise<{ intent, response, file?, sources?, fastPath? }>}
   */
  async run(input) {
    const prep = await this.prepare(input);
    if (prep.done) return prep.result;

    let text;
    try {
      text = await this.engine.chatCompletion({ messages: prep.messages, profile: prep.profile });
    } catch (err) {
      console.warn('[C3PO] AI Engine non raggiungibile:', err.message);
      return { intent: prep.intent, response: '⚠️ Il motore AI locale non risponde in questo momento. Riprova tra qualche secondo.', error: true };
    }
    if (!text) text = 'Non ho una risposta affidabile a questa domanda.';
    return { intent: prep.intent, response: text + (prep.suffix || ''), ...prep.meta };
  }

  /**
   * Risposta SSE (assistente web). Gli intent con file vengono risolti per
   * intero e inviati come evento `file_generated` + testo.
   */
  async stream(input, res, extraMeta = {}) {
    const prep = await this.prepare(input);
    if (prep.done) {
      const { result } = prep;
      if (result.file) {
        res.write(`data: ${JSON.stringify({ type: 'file_generated', file: result.file, done: false })}\n\n`);
      }
      // Con la card del file a schermo bastano titolo e riassunto: niente link nel testo
      const text = result.file
        ? result.response.replace(/!\[[^\]]*\]\([^)]*\)/g, '').replace(/\n.*\*\*\[(?:Scarica|Visualizza)[^\n]*/g, '').trim()
        : result.response;
      res.write(`data: ${JSON.stringify({ token: text, done: false })}\n\n`);
      res.write(`data: ${JSON.stringify({ done: true, final: text, intent: result.intent, file: result.file || null, fastPath: Boolean(result.fastPath), ...extraMeta })}\n\n`);
      return res.end();
    }

    return this.engine.streamChat({
      messages: prep.messages,
      profile: prep.profile,
      suffix: prep.suffix,
      extraMeta: { intent: prep.intent, ...prep.meta, ...extraMeta }
    }, res);
  }
}

module.exports = C3poOrchestrator;
