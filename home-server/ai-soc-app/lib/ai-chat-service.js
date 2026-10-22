/**
 * AI Assistant & Multimodal Studio Engine
 * Handles Text/Code Chat, Vision/OCR Document Analysis, Safe Server Path Reading & SSE Streaming
 * La generazione (intent, prompt, tool) e' delegata a C3poOrchestrator.
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const AiEngineClient = require('./ai-engine-client');
const C3poOrchestrator = require('./c3po-orchestrator');
const webResearchService = require('./web-research-service');

class AiChatService {
  constructor(db, options = {}) {
    this.db = db;
    this.aiCoreUrl = options.aiCoreUrl || options.ollamaHost || process.env.AI_CORE_URL || process.env.OLLAMA_HOST || 'http://minicpm2b-n100:8080';
    this.textModel = options.textModel || process.env.AI_MODEL || process.env.OLLAMA_MODEL || 'minicpm-2b';
    this.visionModel = options.visionModel || 'moondream';
    // Endpoint Ollama-compatibile per la visione: con Ollama dismesso non c'e'
    this.visionHost = options.visionHost || process.env.VISION_HOST || null;
    this.mediaStudio = options.mediaStudio || null;
    this.sessionContexts = new Map();
    this.client = new AiEngineClient({
      aiCoreUrl: this.aiCoreUrl,
      model: this.textModel
    });
    this.orchestrator = options.orchestrator || new C3poOrchestrator({
      engine: this.client,
      telemetryCollector: options.telemetryCollector || null,
      mediaStudio: this.mediaStudio,
      webResearch: webResearchService,
      fastPath: options.fastPath || null,
      db
    });

    // Allowed roots for readFileSafe (inside container)
    this.allowedRoots = [
      '/app/cloud-data',
      '/app/data',
      '/app/archive-data',
      '/app/nginx-logs',
      '/app/autoheal-facts'
    ];
  }

  clearContext(sessionId, userId) {
    if (sessionId) {
      this.sessionContexts.delete(sessionId);
    }
    if (userId) {
      this.sessionContexts.delete(`user_${userId}`);
    }
    return { cleared: true };
  }

  // ----------------------------------------------------
  // SAFE LOCAL PATH READER (Local Path Reader Tool)
  // ----------------------------------------------------

  readFileSafe(rawPath) {
    if (!rawPath || typeof rawPath !== 'string') {
      throw new Error('Percorso file non specificato.');
    }

    let targetPath = rawPath.trim();

    // Map common host paths to container mount paths
    if (targetPath.startsWith('/home/gabibbo/HOME-SERVER/cloud-data')) {
      targetPath = targetPath.replace('/home/gabibbo/HOME-SERVER/cloud-data', '/app/cloud-data');
    } else if (targetPath.startsWith('/home/gabibbo/HOME-SERVER/ai-soc-data')) {
      targetPath = targetPath.replace('/home/gabibbo/HOME-SERVER/ai-soc-data', '/app/data');
    } else if (targetPath.startsWith('/home/gabibbo/HOME-SERVER/archive-data')) {
      targetPath = targetPath.replace('/home/gabibbo/HOME-SERVER/archive-data', '/app/archive-data');
    } else if (targetPath.startsWith('cloud-data/')) {
      targetPath = path.join('/app/cloud-data', targetPath.replace(/^cloud-data\//, ''));
    } else if (targetPath.startsWith('ai-soc-data/')) {
      targetPath = path.join('/app/data', targetPath.replace(/^ai-soc-data\//, ''));
    }

    // Resolve absolute path and normalize away '..'
    const resolved = path.resolve(targetPath);

    // Verify against allowed root whitelist
    const isAllowed = this.allowedRoots.some(root => {
      const normalizedRoot = path.resolve(root);
      return resolved === normalizedRoot || resolved.startsWith(normalizedRoot + path.sep);
    });

    if (!isAllowed) {
      throw new Error(`Accesso negato: il percorso "${rawPath}" non rientra nelle directory autorizzate (cloud-data, archive-data, ai-soc-data).`);
    }

    if (!fs.existsSync(resolved)) {
      throw new Error(`File non trovato: ${rawPath}`);
    }

    const stat = fs.statSync(resolved);
    if (!stat.isFile()) {
      throw new Error(`Il percorso specificato non è un file regolare: ${rawPath}`);
    }

    // Read content safely (max 500 KB to preserve memory and context)
    const MAX_SIZE = 500 * 1024;
    const isTruncated = stat.size > MAX_SIZE;
    const fd = fs.openSync(resolved, 'r');
    const buffer = Buffer.alloc(Math.min(stat.size, MAX_SIZE));
    fs.readSync(fd, buffer, 0, buffer.length, 0);
    fs.closeSync(fd);

    const ext = path.extname(resolved).toLowerCase();
    let content = '';

    if (ext === '.pdf') {
      try {
        const pdfParse = require('pdf-parse');
        const pdfData = pdfParse(buffer);
        content = pdfData.text || '';
      } catch (_) {
        content = buffer.toString('utf8');
      }
    } else {
      content = buffer.toString('utf8');
    }

    return {
      path: rawPath,
      resolvedPath: resolved,
      size: stat.size,
      sizeFormatted: (stat.size / 1024).toFixed(1) + ' KB',
      isTruncated,
      content
    };
  }

  // ----------------------------------------------------
  // MULTIMODAL & DOCUMENT EXTRACTION
  // ----------------------------------------------------

  async extractAttachmentData({ filename, buffer, mimeType }) {
    const ext = path.extname(filename).toLowerCase();
    const isPdf = ext === '.pdf' || mimeType === 'application/pdf';
    const isImage = ['.png', '.jpg', '.jpeg', '.webp'].includes(ext) || (mimeType && mimeType.startsWith('image/'));

    // 1. PDF Extraction
    if (isPdf) {
      try {
        const pdfParse = require('pdf-parse');
        const parsed = await pdfParse(buffer);
        return {
          filename,
          type: 'pdf',
          size: buffer.length,
          text: parsed.text || '',
          pages: parsed.numpages || 1,
          preview: (parsed.text || '').slice(0, 300)
        };
      } catch (err) {
        return {
          filename,
          type: 'pdf',
          size: buffer.length,
          text: `[Errore lettura PDF: ${err.message}]`,
          pages: 1,
          preview: ''
        };
      }
    }

    // 2. Image Multimodal / Vision Extraction (via Moondream / Vision Model)
    if (isImage) {
      const base64 = buffer.toString('base64');
      let visionDescription = null;

      try {
        visionDescription = await this.describeImageWithVision(base64);
      } catch (_) {}
      // Senza modello di visione lo si dice chiaramente: una descrizione finta
      // ("Immagine analizzata con successo") spinge il modello a inventare.
      if (!visionDescription) {
        visionDescription = `Immagine ${filename} (${(buffer.length / 1024).toFixed(1)} KB): analisi visiva non disponibile su questo server.`;
      }

      return {
        filename,
        type: 'image',
        size: buffer.length,
        base64: base64.slice(0, 500) + '...',
        visionText: visionDescription,
        text: `[IMMAGINE ALLEGATA (${filename})]:\n${visionDescription}`
      };
    }

    // 3. Text, Markdown, CSV, Code files
    const text = buffer.toString('utf8');
    return {
      filename,
      type: 'text',
      size: buffer.length,
      text,
      preview: text.slice(0, 300)
    };
  }

  /**
   * Descrizione dell'immagine via endpoint Ollama-compatibile (VISION_HOST).
   * Restituisce null se non configurato o se la risposta non e' utilizzabile.
   */
  async describeImageWithVision(imageBase64) {
    if (!this.visionHost) return null;
    const url = new URL(this.visionHost);
    const postData = JSON.stringify({
      model: this.visionModel,
      prompt: 'Descrivi in dettaglio cosa mostra questa immagine ed estrai qualsiasi testo o dato visibile (OCR):',
      images: [imageBase64],
      stream: false,
      options: {
        num_ctx: 2048,
        num_thread: 4
      }
    });

    return new Promise((resolve) => {
      const req = http.request({
        hostname: url.hostname,
        port: url.port || 11434,
        path: '/api/generate',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData)
        },
        timeout: 25000
      }, (res) => {
        let raw = '';
        res.on('data', chunk => raw += chunk);
        res.on('end', () => {
          try {
            const data = JSON.parse(raw);
            resolve(data.response || null);
          } catch (_) {
            resolve(null);
          }
        });
      });

      req.on('error', () => resolve(null));
      req.on('timeout', () => { req.destroy(); resolve(null); });
      req.write(postData);
      req.end();
    });
  }

  // ----------------------------------------------------
  // SSE STREAMING CHAT (Productivity, Code & Multimodal)
  // ----------------------------------------------------

  async streamChat(options = {}, res) {
    const {
      prompt,
      history = [],
      attachments = [],
      serverFilePath = null,
      sessionId = null,
      userId = 'mattia',
      isLive = false,
      mode = null
    } = options;
    // 1. Setup SSE headers IMMEDIATELY so the client connection opens with 0 latency
    if (!res.headersSent) {
      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); // Prevents Nginx buffering
      res.flushHeaders?.();
    }
    // Flush an initial keepalive/handshake comment
    res.write(': stream-start\n\n');

    // A. Lettura esplicita di un file del server (percorsi in whitelist)
    let filePathToRead = serverFilePath;
    if (!filePathToRead) {
      const pathMatch = prompt.match(/(?:leggi|analizza|mostra|riassumi|apri|esamina)\s+(?:il\s+file\s+)?(?:in\s+)?([/\w.-]+(?:cloud-data|ai-soc-data|archive-data)[/\w.-]*)/i);
      if (pathMatch && pathMatch[1]) {
        filePathToRead = pathMatch[1].trim();
      }
    }

    let extraContext = '';
    if (filePathToRead) {
      try {
        const fileData = this.readFileSafe(filePathToRead);
        extraContext = `--- ${fileData.path} (${fileData.sizeFormatted}) ---\n${fileData.content}`;
      } catch (err) {
        extraContext = `Lettura file non riuscita (${filePathToRead}): ${err.message}`;
      }
    }

    // B. Intent, prompt, tool e streaming: tutto nell'orchestratore
    const isLiveMode = Boolean(isLive || mode === 'live');
    await this.orchestrator.stream({
      prompt,
      history,
      attachments,
      extraContext,
      username: userId,
      isLive: isLiveMode
    }, res, { sessionId, isLive: isLiveMode });
  }
}

module.exports = AiChatService;
