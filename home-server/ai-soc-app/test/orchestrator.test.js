const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const C3poOrchestrator = require('../lib/c3po-orchestrator');
const { SYSTEM_PROMPTS } = require('../lib/prompt-builder');

// Motore finto: registra le richieste e risponde in base al profilo
function fakeEngine(replies = {}) {
  const calls = [];
  return {
    calls,
    async chatCompletion({ messages, profile }) {
      calls.push({ messages, profile });
      return replies[profile] ?? 'risposta';
    },
    async streamChat() { throw new Error('non usato'); }
  };
}

const TELEMETRY = {
  score: 95, securityLevel: 'ECCELLENTE', issues: [],
  system: { cpuTemp: 42, loadAvg: [0.2, 0.3, 0.3], ramUsedFormatted: '3.0 GB', ramTotalFormatted: '11.2 GB', ramUsedPct: 27 },
  containers: { running: 23, total: 23, unhealthy: 0, list: [] },
  ssh: { failedAttempts: 0, bruteForceSuspects: [] }, nginx: {}
};

test('"come stai ?": prompt chat pulito, niente telemetria, niente immagini dalla history', async () => {
  const engine = fakeEngine({ chat: 'Sto bene, grazie! Come posso aiutarti?' });
  const orch = new C3poOrchestrator({ engine, telemetryCollector: { collectFullTelemetry: async () => TELEMETRY } });
  const history = [
    { role: 'user', content: 'disegna il server' },
    { role: 'assistant', content: "È un pazienato da fare, ma qui è la mia rendita: ![x](https://image.pollinations.ai/p)" },
    { role: 'user', content: 'come stai ?' }
  ];
  const res = await orch.run({ prompt: 'come stai ?', history });
  assert.equal(res.intent, 'INTENT_CHAT');
  assert.equal(res.response, 'Sto bene, grazie! Come posso aiutarti?');
  const sent = engine.calls[0];
  assert.equal(sent.profile, 'chat');
  assert.equal(sent.messages[0].content, SYSTEM_PROMPTS.chat);
  const payload = JSON.stringify(sent.messages);
  assert.ok(!payload.includes('!['), 'immagine della history inviata al modello');
  assert.ok(!payload.includes('TELEMETRIA'));
  assert.equal(sent.messages[sent.messages.length - 1].content, 'come stai ?');
});

test('SOC: telemetria nel turno user, system invariato (cache dei prompt)', async () => {
  const engine = fakeEngine({ soc: 'Tutto regolare.' });
  const orch = new C3poOrchestrator({ engine, telemetryCollector: { collectFullTelemetry: async () => TELEMETRY } });
  await orch.run({ prompt: 'quanta RAM sta usando il server?' });
  const sent = engine.calls[0];
  assert.equal(sent.profile, 'soc');
  assert.equal(sent.messages[0].content, SYSTEM_PROMPTS.soc);
  assert.match(sent.messages[1].content, /^\[TELEMETRIA\][\s\S]*ram=3\.0 GB\/11\.2 GB/);
});

test('ricerca web: fonti numerate al modello, elenco link aggiunto dal codice', async () => {
  const engine = fakeEngine({ rag: 'Linux 6.12 introduce sched_ext [1].' });
  const webResearch = {
    async searchAndScrape(query) {
      assert.equal(query, 'le novità di Linux kernel 6.12');
      return { query, sources: [{ n: 1, title: 'Kernel.org', url: 'https://kernel.org', text: 'sched_ext...' }], context: '[1] Kernel.org\nsched_ext...' };
    },
    formatSources: require('../lib/web-research-service').formatSources
  };
  const orch = new C3poOrchestrator({ engine, webResearch });
  const res = await orch.run({ prompt: 'Cerca su internet le novità di Linux kernel 6.12 e fammi un riassunto' });
  assert.equal(res.intent, 'INTENT_WEB_SEARCH');
  assert.match(res.response, /\[1\]\.\n\n\*\*Fonti consultate:\*\*\n1\. \[Kernel\.org\]\(https:\/\/kernel\.org\)$/);
  assert.match(engine.calls[0].messages[1].content, /^\[FONTI\]/);
});

test('immagine: il modello traduce soltanto, la risposta e il link li scrive il codice', async () => {
  const engine = fakeEngine({ translate: 'An astronaut cat' });
  let received;
  const mediaStudio = {
    async generateImage(opts) { received = opts; return { url: '/api/ai-chat/media/img-1.jpg', filename: 'img-1.jpg' }; }
  };
  const orch = new C3poOrchestrator({ engine, mediaStudio });
  const res = await orch.run({ prompt: "Crea un'immagine di un gatto astronauta in stile acquerello" });
  assert.equal(res.intent, 'INTENT_IMAGE');
  assert.match(received.prompt, /^An astronaut cat, watercolor painting/);
  assert.equal(engine.calls.length, 1);
  assert.equal(engine.calls[0].profile, 'translate');
  assert.match(res.response, /!\[gatto astronauta in stile acquerello\]\(\/api\/ai-chat\/media\/img-1\.jpg\)/);
});

test('PDF: Markdown del modello -> file PDF reale con anteprima', async () => {
  const md = `## Sommario\nLe best practice Docker riducono la superficie di attacco e limitano l'impatto di una compromissione dell'host.\n\n## Raccomandazioni\n- Utente non-root\n- Filesystem in sola lettura\n- Limiti di memoria con mem_limit\n\n| Opzione | Valore |\n|---|---|\n| read_only | true |`;
  const engine = fakeEngine({ doc: md });
  const outputDir = fs.mkdtempSync(path.join(os.tmpdir(), 'c3po-'));
  const orch = new C3poOrchestrator({ engine, mediaStudio: { outputDir } });
  const res = await orch.run({ prompt: 'Crea un documento PDF sulle best practice di sicurezza Docker', username: 'mattia' });
  assert.equal(res.intent, 'INTENT_PDF');
  assert.equal(res.file.type, 'pdf');
  assert.ok(fs.existsSync(res.file.path));
  assert.ok(fs.existsSync(`${res.file.path}.meta.json`));
  assert.match(res.response, /Documento pronto: \*\*Best practice di sicurezza Docker\*\*/);
  assert.match(res.response, /\*\*Sezioni:\*\* Sommario · Raccomandazioni/);
  assert.equal(engine.calls[0].profile, 'doc');
});

test('motore AI giu: messaggio onesto, niente risposta inventata', async () => {
  const engine = { async chatCompletion() { throw new Error('ECONNREFUSED'); } };
  const orch = new C3poOrchestrator({ engine });
  const res = await orch.run({ prompt: 'ciao' });
  assert.equal(res.error, true);
  assert.match(res.response, /non risponde/);
});
