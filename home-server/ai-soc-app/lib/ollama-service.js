/**
 * AI SOC Sentinel — Inference Engine Service
 * Connects to Autonomous llama.cpp native runtime on Intel N100
 * Handles Hardware Ground-Truth Injection, RAG, and Low-Latency SSE Streaming
 *
 * System prompt statico ('soc' in prompts.json): telemetria e memoria RAG
 * vanno nel turno user, cosi' llama-server riusa il KV del prefisso.
 */

const AiEngineClient = require('./ai-engine-client');
const { sanitizeHistory, compactTelemetry, buildUserTurn, buildMessages } = require('./prompt-builder');

class OllamaService {
  constructor(db, collector, options = {}) {
    this.db = db;
    this.collector = collector;
    this.client = new AiEngineClient({
      aiCoreUrl: options.aiCoreUrl || options.ollamaHost || process.env.AI_CORE_URL || process.env.OLLAMA_HOST || 'http://minicpm2b-n100:8080',
      model: options.model || process.env.AI_MODEL || process.env.OLLAMA_MODEL || 'minicpm-2b'
    });
  }

  async getBrainStatus() {
    return this.client.getStatus();
  }

  // Memoria storica SOC (soc-memory.db) in righe compatte
  buildKnowledgeContext(prompt) {
    const matches = this.db.searchKnowledge(prompt, 3);
    if (!matches.length) return '';
    return matches.map((m, i) =>
      `${i + 1}. sintomo="${m.symptom}" causa="${m.root_cause}" fix=\`${m.remediation_cmd}\` efficacia=${(m.effectiveness_score * 100).toFixed(0)}%`
    ).join('\n');
  }

  async streamChat({ prompt, history = [] }, res) {
    let liveTelemetry = null;
    try {
      liveTelemetry = await this.collector.collectFullTelemetry();
    } catch (_) {}

    const userTurn = buildUserTurn(prompt, {
      telemetry: compactTelemetry(liveTelemetry),
      extraContext: this.buildKnowledgeContext(prompt)
    });
    const messages = buildMessages('soc', userTurn, sanitizeHistory(history, { intent: 'INTENT_SOC', currentPrompt: prompt }));

    // Salva il messaggio utente nella cronologia persistente del SOC
    if (this.db && typeof this.db.addSocChatMessage === 'function') {
      this.db.addSocChatMessage({ role: 'user', content: prompt, eventType: 'user_prompt' });
    }

    await this.client.streamChat({ messages, profile: 'soc', max_tokens: 384 }, res, (fullText) => {
      // Salva la risposta dell'assistente nella cronologia persistente del SOC
      if (this.db && typeof this.db.addSocChatMessage === 'function') {
        this.db.addSocChatMessage({ role: 'assistant', content: fullText, eventType: 'ai_response' });
      }

      this.db.recordEvent({
        event_type: 'ai_chat_interaction',
        severity: 'info',
        title: 'Interazione SOC Sentinel AI',
        description: `Interrogazione utente analizzata. Risposta generata con successo (${fullText.length} caratteri).`,
        metadata: { promptPreview: prompt.substring(0, 100), responseLength: fullText.length }
      });
    });
  }

  async generateCompletion(prompt, customOptions = {}) {
    return this.client.chatCompletion({
      messages: buildMessages('soc', prompt),
      profile: 'soc',
      max_tokens: customOptions.max_tokens || 384,
      temperature: customOptions.temperature
    });
  }
}

module.exports = OllamaService;
