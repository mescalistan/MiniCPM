const test = require('node:test');
const assert = require('node:assert/strict');
const { sanitizeHistory, compactTelemetry, buildUserTurn, buildMessages, SYSTEM_PROMPTS } = require('../lib/prompt-builder');
const { sanitizeModelText, isDegenerateTail, SAMPLING_PROFILES } = require('../lib/ai-engine-client');
const { cleanTranslation, extractImageSubject, dictionaryTranslate } = require('../lib/image-intent');

// La risposta "trapelata" vista in produzione
const LEAKED = "È un pazienato da fare, ma qui è la mia rendita dell'illustrazione futurista del nostro HOME-SERVER illuminato al buio: ![HOME-SERVER](https://image.pollinations.ai/prompt/x)";

test('history: le immagini delle risposte precedenti non arrivano al modello', () => {
  const history = [
    { role: 'user', content: 'disegna il server' },
    { role: 'assistant', content: LEAKED },
    { role: 'user', content: 'come stai ?' }
  ];
  const out = sanitizeHistory(history, { intent: 'INTENT_CHAT', currentPrompt: 'come stai ?' });
  assert.equal(out.length, 2);
  assert.equal(out[0].role, 'user');
  assert.ok(!JSON.stringify(out).includes('!['));
  assert.ok(!JSON.stringify(out).includes('pollinations'));
});

test('history: niente history per RAG / PDF / immagini', () => {
  const history = [{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }];
  assert.deepEqual(sanitizeHistory(history, { intent: 'INTENT_WEB_SEARCH' }), []);
  assert.deepEqual(sanitizeHistory(history, { intent: 'INTENT_PDF' }), []);
});

test('history: ruoli alternati, errori e system scartati', () => {
  const out = sanitizeHistory([
    { role: 'system', content: 'regole' },
    { role: 'assistant', content: 'benvenuto' },
    { role: 'user', content: 'uno' },
    { role: 'user', content: 'due' },
    { role: 'assistant', content: '⚠️ errore motore' },
    { role: 'assistant', content: 'ok' }
  ]);
  assert.deepEqual(out.map(m => m.role), ['user', 'assistant']);
  assert.equal(out[0].content, 'uno\ndue');
});

test('system prompt statici e senza esempi da ricopiare', () => {
  for (const [k, v] of Object.entries(SYSTEM_PROMPTS)) {
    if (k.startsWith('_')) continue;
    assert.ok(!/!\[|https?:|\d+%|°C/.test(v), `prompt ${k} contiene esempi o dati dinamici`);
  }
  const m = buildMessages('chat', 'ciao');
  assert.equal(m[0].content, SYSTEM_PROMPTS.chat);
});

test('telemetria compatta chiave=valore nel turno user', () => {
  const t = compactTelemetry({
    timestamp: '2026-10-22T20:00:00Z', score: 92, securityLevel: 'BUONO', issues: [],
    system: { cpuTemp: 44, loadAvg: [0.5, 0.4, 0.3], ramUsedFormatted: '3.1 GB', ramTotalFormatted: '11.2 GB', ramUsedPct: 28 },
    containers: { running: 22, total: 23, unhealthy: 1, list: [{ name: 'mail-app', state: 'running', unhealthy: true }] },
    ssh: { failedAttempts: 3, bruteForceSuspects: [] },
    nginx: { clientErrors4xx: 2, serverErrors5xx: 0 }
  });
  assert.match(t, /container=22\/23 unhealthy=1 problemi=\[mail-app:unhealthy\]/);
  assert.match(t, /ram=3\.1 GB\/11\.2 GB \(28%\)/);
  const turn = buildUserTurn('come sta il server?', { telemetry: t });
  assert.ok(turn.startsWith('[TELEMETRIA]'));
  assert.ok(turn.endsWith('[DOMANDA]\ncome sta il server?'));
});

test('campionamento: nessuna penalita additiva in nessun profilo', () => {
  for (const p of Object.values(SAMPLING_PROFILES)) {
    assert.equal(p.presence_penalty, undefined);
    assert.equal(p.frequency_penalty, undefined);
    assert.ok(p.repeat_penalty <= 1.1);
  }
});

test('sanitizeModelText: token speciali, byte rotti e cicli', () => {
  assert.equal(sanitizeModelText('Ciao!<用户>Utente: altro'), 'Ciao!Utente: altro');
  assert.equal(sanitizeModelText('uniform ogeogeogeogeoge fine'), 'uniform oge fine');
  assert.equal(sanitizeModelText('rel��ojoj'), 'relojoj');
  assert.equal(sanitizeModelText('riga uno\nriga uno\nriga due'), 'riga uno\nriga due');
  // Il codice non si tocca
  const code = '```js\nif (a) {\n  }\n}\n```';
  assert.equal(sanitizeModelText(code), code);
});

test('isDegenerateTail: ferma i cicli ma non tabelle o separatori', () => {
  assert.equal(isDegenerateTail('la risposta è relúúojojojojojojoj'), true);
  // In streaming il ciclo viene colto alla ripetizione successiva
  assert.equal(isDegenerateTail('Il server funziona. Il server funziona. Il server funziona. Il server funziona.'), true);
  assert.equal(isDegenerateTail('| a | b | c | d |\n|---|---|---|---|---|---|'), false);
  assert.equal(isDegenerateTail('========================================'), false);
  assert.equal(isDegenerateTail('Una risposta normale e sensata in italiano.'), false);
});

test('immagini: soggetto e validazione della traduzione', () => {
  assert.equal(extractImageSubject("C-3PO, creami un'immagine di un gatto astronauta"), 'gatto astronauta');
  assert.equal(extractImageSubject('disegnami un drago sopra un castello'), 'drago sopra un castello');
  assert.equal(cleanTranslation('An astronaut cat floating in space'), 'An astronaut cat floating in space');
  assert.equal(cleanTranslation('Here is the prompt: cat'), null);
  assert.equal(cleanTranslation('![x](https://a.b)'), null);
  assert.equal(dictionaryTranslate('gatto astronauta'), 'cat astronaut');
});
