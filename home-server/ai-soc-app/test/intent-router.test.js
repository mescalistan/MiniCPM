const test = require('node:test');
const assert = require('node:assert/strict');
const { INTENTS, classifyIntent, extractSearchQuery, extractDocTopic } = require('../lib/intent-router');

const cases = [
  // Saluti e conversazione: mai telemetria, mai tool
  ['come stai ?', INTENTS.CHAT],
  ['Ciao, come stai?', INTENTS.CHAT],
  ['Spiegami la differenza tra TCP e UDP', INTENTS.CHAT],
  ['Scrivi una funzione Python che inverte una lista', INTENTS.CHAT],
  ['cerca di spiegarmi la ricorsione', INTENTS.CHAT],
  ['come funziona la memoria umana?', INTENTS.CHAT],
  ['analizza questa immagine', INTENTS.CHAT],
  // SOC
  ['quanta RAM sta usando il server?', INTENTS.SOC],
  ['Ci sono container unhealthy?', INTENTS.SOC],
  ['ci sono minacce?', INTENTS.SOC],
  ['controlla i log di nginx', INTENTS.SOC],
  ['come configuro fail2ban per ssh?', INTENTS.SOC],
  // Immagini
  ["Crea un'immagine di un gatto astronauta", INTENTS.IMAGE],
  ['disegnami un drago sopra un castello', INTENTS.IMAGE],
  ['C-3PO, genera un logo minimalista per HOME-SERVER', INTENTS.IMAGE],
  ['fammi una foto realistica di un tramonto sul mare', INTENTS.IMAGE],
  // Documenti
  ['Crea un documento PDF sulle best practice di sicurezza Docker', INTENTS.PDF],
  ['prepara un report in pdf sullo stato del server', INTENTS.PDF],
  ['scrivimi un documento word sulla gestione dei backup', INTENTS.PDF],
  // Ricerca web
  ['Cerca su internet le novità di Linux kernel 6.12 e fammi un riassunto', INTENTS.WEB_SEARCH],
  ['quali sono le ultime notizie su Docker?', INTENTS.WEB_SEARCH],
  ['riassumi https://example.com/articolo', INTENTS.WEB_SEARCH]
];

for (const [prompt, expected] of cases) {
  test(`intent: "${prompt}" -> ${expected}`, () => {
    assert.equal(classifyIntent(prompt).intent, expected);
  });
}

test('SOC: telemetria solo per domande sullo stato attuale', () => {
  assert.equal(classifyIntent('quanta RAM sta usando il server?').params.needsTelemetry, true);
  assert.equal(classifyIntent('come configuro fail2ban per ssh?').params.needsTelemetry, false);
});

test('PDF vs DOCX', () => {
  assert.equal(classifyIntent('crea un pdf su Linux').params.format, 'pdf');
  assert.equal(classifyIntent('crea un file word sul budget').params.format, 'docx');
});

test('allegati: sempre CHAT, niente tool', () => {
  assert.equal(classifyIntent('crea un pdf da questo file', { hasAttachments: true }).intent, INTENTS.CHAT);
});

test('URL esplicito estratto senza punteggiatura finale', () => {
  const r = classifyIntent('leggi https://example.com/pagina.html.');
  assert.equal(r.intent, INTENTS.WEB_SEARCH);
  assert.equal(r.params.url, 'https://example.com/pagina.html');
});

test('query di ricerca ripulita', () => {
  assert.equal(extractSearchQuery('Cerca su internet le novità di Linux kernel 6.12 e fammi un riassunto'), 'le novità di Linux kernel 6.12');
});

test('argomento del documento', () => {
  assert.equal(extractDocTopic('Crea un documento PDF sulle best practice di sicurezza Docker'), 'Best practice di sicurezza Docker');
  assert.equal(extractDocTopic('prepara un report in pdf sullo stato del server'), 'Stato del server');
});
