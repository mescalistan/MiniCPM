/**
 * AI Engine Client per llama-server (API OpenAI-compatibile, JSON e SSE)
 * Calibrato per MiniCPM-2B Q4_K_M su Intel N100 (4 E-core, AVX2 + VNNI)
 *
 * Perche' niente presence_penalty / frequency_penalty:
 * il vocabolario di MiniCPM ricade sui token-byte (<0xC3>, <0xA8>, ...) per
 * molte lettere accentate. Le penalita' additive colpiscono proprio quei byte
 * dopo pochi usi, il campionatore sceglie un byte "vicino" e l'UTF-8 si rompe:
 * da qui le sequenze tipo "relúúojoj". Il controllo delle ripetizioni resta a
 * repeat_penalty (moltiplicativo, lieve) e a min_p.
 */

// Profili di campionamento per intent. presence/frequency sono sempre 0.
const SAMPLING_PROFILES = {
  chat:      { temperature: 0.5,  top_p: 0.8, top_k: 40, min_p: 0.05, repeat_penalty: 1.05, repeat_last_n: 128, max_tokens: 320, timeoutMs: 60000 },
  live:      { temperature: 0.4,  top_p: 0.8, top_k: 40, min_p: 0.05, repeat_penalty: 1.05, repeat_last_n: 64,  max_tokens: 90,  timeoutMs: 30000 },
  soc:       { temperature: 0.2,  top_p: 0.8, top_k: 20, min_p: 0.05, repeat_penalty: 1.05, repeat_last_n: 128, max_tokens: 256, timeoutMs: 60000 },
  rag:       { temperature: 0.2,  top_p: 0.8, top_k: 20, min_p: 0.05, repeat_penalty: 1.05, repeat_last_n: 128, max_tokens: 320, timeoutMs: 75000 },
  doc:       { temperature: 0.35, top_p: 0.85, top_k: 40, min_p: 0.05, repeat_penalty: 1.07, repeat_last_n: 256, max_tokens: 750, timeoutMs: 150000 },
  translate: { temperature: 0.0,  top_p: 1.0, top_k: 1,  min_p: 0.0,  repeat_penalty: 1.0,  repeat_last_n: 0,   max_tokens: 60,  timeoutMs: 20000 }
};

// Marcatori del template MiniCPM: se il modello li genera sta "inventando"
// il turno successivo, quindi si ferma li'.
const DEFAULT_STOP = ['<用户>', '<AI>', '</s>', '<|im_end|>', '<|im_start|>', '<|endoftext|>'];

const SPECIAL_TOKEN_RE = /<用户>|<AI>|<\/s>|<s>|<\|im_(?:start|end)\|>|<\|endoftext\|>/g;

/**
 * Ripulisce l'output del modello: token speciali trapelati, caratteri di
 * sostituzione UTF-8, cicli di sillabe ("ogeogeoge") e righe ripetute.
 */
function sanitizeModelText(raw) {
  if (!raw) return '';
  let text = String(raw)
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<\/?think>/gi, '')
    .replace(SPECIAL_TOKEN_RE, '')
    .replace(/�+/g, '');

  // Sillabe ripetute 4+ volte di fila -> una sola occorrenza; righe di testo
  // identiche consecutive -> una sola. I blocchi di codice restano intatti
  // (due "}" di fila sono legittimi).
  const lines = text.split('\n');
  const deduped = [];
  let inFence = false;
  for (let line of lines) {
    if (/^\s*```/.test(line)) {
      inFence = !inFence;
      deduped.push(line);
      continue;
    }
    if (!inFence) {
      line = line.replace(/(\p{L}{2,12}?)\1{3,}/gu, '$1');
      const prev = deduped[deduped.length - 1];
      if (/\p{L}{3}/u.test(line) && prev !== undefined && line.trim() === prev.trim()) continue;
    }
    deduped.push(line);
  }
  text = deduped.join('\n');

  // Una frase lunga che ricompare: si tronca alla seconda occorrenza
  const sentences = text.match(/[^.!?\n]{40,}[.!?]/g) || [];
  const seen = new Set();
  for (const s of sentences) {
    const key = s.trim().toLowerCase();
    if (seen.has(key)) {
      const first = text.indexOf(s);
      const second = text.indexOf(s, first + s.length);
      if (second > -1) text = text.slice(0, second);
      break;
    }
    seen.add(key);
  }

  return text.replace(/\n{3,}/g, '\n\n').trim();
}

/**
 * Vero se la coda dello stream e' degenerata in un ciclo: serve a interrompere
 * la generazione invece di bruciare CPU fino a max_tokens.
 */
function isDegenerateTail(tail) {
  if (!tail || tail.length < 24) return false;
  if (/(\p{L}{1,6})\1{5,}$/u.test(tail)) return true;   // "ojojojojojoj"
  if (/�{2,}/.test(tail)) return true;             // byte UTF-8 rotti
  // Stessa frase 3 volte di fila; il blocco deve contenere parole, cosi'
  // separatori di tabella ("|---|---|") e righe "=====" non fanno scattare nulla.
  const m = tail.match(/([\s\S]{8,80}?)\1{2,}$/);
  return Boolean(m && /\p{L}{3}/u.test(m[1]));
}

class AiEngineClient {
  constructor(options = {}) {
    this.baseUrl = (options.aiCoreUrl || process.env.AI_CORE_URL || process.env.OLLAMA_HOST || 'http://minicpm2b-n100:8080').replace(/\/+$/, '');
    this.model = options.model || process.env.AI_MODEL || process.env.OLLAMA_MODEL || 'minicpm-2b';
    this.defaultStop = DEFAULT_STOP;
  }

  /**
   * Corpo della richiesta: profilo + override espliciti del chiamante.
   */
  buildBody({ messages, profile = 'chat', max_tokens, temperature, stop, stream }) {
    const p = SAMPLING_PROFILES[profile] || SAMPLING_PROFILES.chat;
    return {
      model: this.model,
      messages,
      stream: Boolean(stream),
      max_tokens: max_tokens || p.max_tokens,
      temperature: typeof temperature === 'number' ? temperature : p.temperature,
      top_p: p.top_p,
      top_k: p.top_k,
      min_p: p.min_p,
      repeat_penalty: p.repeat_penalty,
      repeat_last_n: p.repeat_last_n,
      presence_penalty: 0,
      frequency_penalty: 0,
      // Riusa il KV del prefisso statico (system prompt) tra una richiesta e l'altra
      cache_prompt: true,
      stop: stop || this.defaultStop
    };
  }

  timeoutFor(profile) {
    return (SAMPLING_PROFILES[profile] || SAMPLING_PROFILES.chat).timeoutMs;
  }

  /**
   * Status check of the autonomous runtime
   */
  async getStatus() {
    try {
      const res = await fetch(`${this.baseUrl}/health`, { signal: AbortSignal.timeout(3500) });
      if (res.ok) {
        return {
          state: 'ACTIVE',
          label: '🟢 ATTIVO IN RAM (llama.cpp)',
          engine: 'llama.cpp C++ Nativo (Intel N100 AVX2+VNNI)',
          model: this.model,
          ramMb: 3900,
          ramFormatted: '3.9 GB',
          details: {
            runner: 'llama-server C++',
            threads: 3,
            kv_cache: 'f16',
            flash_attn: true,
            endpoint: this.baseUrl
          }
        };
      }
    } catch (_) {}

    return {
      state: 'OFFLINE',
      label: '⚠️ RUNTIME NON RAGGIUNGIBILE',
      engine: 'llama.cpp',
      model: this.model,
      ramMb: 0,
      ramFormatted: '0 MB',
      details: { endpoint: this.baseUrl }
    };
  }

  /**
   * Non-streaming Chat Completion. Restituisce il testo gia' ripulito.
   * `profile` sceglie il campionamento (chat, soc, rag, doc, translate, live).
   */
  async chatCompletion({ messages, profile = 'chat', max_tokens, temperature, stop = null }) {
    const res = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(this.buildBody({ messages, profile, max_tokens, temperature, stop, stream: false })),
      signal: AbortSignal.timeout(this.timeoutFor(profile))
    });

    if (!res.ok) {
      throw new Error(`AI Engine HTTP error ${res.status}`);
    }

    const data = await res.json();
    return sanitizeModelText(data.choices?.[0]?.message?.content || '');
  }

  /**
   * SSE Streaming Chat with Client Response Piping (Optimized for Intel N100)
   * Interrompe la generazione se l'output degenera in un ciclo.
   * `suffix` viene accodato dopo il testo del modello (es. elenco fonti).
   */
  async streamChat({ messages, profile = 'chat', max_tokens, temperature, stop = null, extraMeta = {}, suffix = '' }, res, onDone = null) {
    if (!res.headersSent) {
      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      res.flushHeaders?.();
    }

    let upstream;
    try {
      upstream = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.buildBody({ messages, profile, max_tokens, temperature, stop, stream: true })),
        signal: AbortSignal.timeout(this.timeoutFor(profile))
      });
    } catch (err) {
      res.write(`data: ${JSON.stringify({ error: `Connessione AI Engine fallita: ${err.message}`, done: true })}\n\n`);
      return res.end();
    }

    if (!upstream.ok) {
      res.write(`data: ${JSON.stringify({ error: `AI Engine risponde con HTTP ${upstream.status}`, done: true })}\n\n`);
      return res.end();
    }

    const reader = upstream.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let fullText = '';
    let tokenCount = 0;
    let degenerated = false;
    const startTime = Date.now();
    let isStreamDone = false;

    try {
      while (!isStreamDone) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data:')) continue;
          const dataStr = trimmed.slice(5).trim();
          if (dataStr === '[DONE]') {
            isStreamDone = true;
            break;
          }

          try {
            const parsed = JSON.parse(dataStr);
            const token = parsed.choices?.[0]?.delta?.content;
            if (token) {
              fullText += token;
              tokenCount++;
              if (isDegenerateTail(fullText.slice(-120))) {
                degenerated = true;
                isStreamDone = true;
                break;
              }
              res.write(`data: ${JSON.stringify({ token, done: false })}\n\n`);
            }
          } catch (_) {}
        }
      }
      // Chiudere il reader chiude la connessione: llama-server smette di generare
      try { await reader.cancel(); } catch (_) {}

      if (suffix) {
        res.write(`data: ${JSON.stringify({ token: suffix, done: false })}\n\n`);
      }
      const cleanText = sanitizeModelText(fullText) + suffix;
      const elapsedSec = (Date.now() - startTime) / 1000;
      const tokPerSec = elapsedSec > 0 ? (tokenCount / elapsedSec).toFixed(1) : null;
      // `final` contiene il testo ripulito: il client puo' sostituire quello accumulato dai token
      res.write(`data: ${JSON.stringify({ done: true, final: cleanText, degenerated, tokPerSec, evalCount: tokenCount, ...extraMeta })}\n\n`);
      res.end();

      if (typeof onDone === 'function') {
        onDone(cleanText, tokPerSec);
      }
    } catch (streamErr) {
      res.write(`data: ${JSON.stringify({ error: `Errore stream: ${streamErr.message}`, done: true })}\n\n`);
      res.end();
    }
  }
}

module.exports = AiEngineClient;
module.exports.SAMPLING_PROFILES = SAMPLING_PROFILES;
module.exports.sanitizeModelText = sanitizeModelText;
module.exports.isDegenerateTail = isDegenerateTail;
