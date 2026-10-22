const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { parseMarkdown, parseInline, normalizeModelMarkdown, renderMarkdownPdf, blocksToSections, toWinAnsi } = require('../lib/markdown-pdf');

const OUT_DIR = path.join(__dirname, 'output');

// Output tipico di un 2B: titolo H1 non richiesto, grassetti, tabella, elenchi misti
const MODEL_MD = `Ecco il documento richiesto:

# Sicurezza dei container Docker

## Sommario
L'hardening dei container riduce la **superficie di attacco** dell'host e limita i danni di una compromissione.

## Best practice
- Eseguire i processi come utente **non-root** (\`USER app\`)
- Montare il filesystem in *sola lettura*
* Rimuovere le capability con \`cap_drop: [ALL]\`

**Configurazione consigliata**

| Opzione | Valore | Motivo |
|---|---|---|
| read_only | true | Impedisce scritture persistenti |
| mem_limit | 512m | Evita esaurimento della RAM |
| cap_drop | ALL | Privilegi minimi |

### Comandi utili
\`\`\`bash
docker run --read-only --cap-drop ALL nginx
\`\`\`

1. Aggiornare le immagini base
2. Scansionare le dipendenze

Spero che questo documento ti sia utile!`;

test('inline: grassetto, corsivo, codice, link; asterischi spaiati eliminati', () => {
  const runs = parseInline('testo **forte** e *corsivo* con `code` e [link](https://a.it) e ** spaiato');
  assert.deepEqual(runs.filter(r => r.bold).map(r => r.text), ['forte']);
  assert.deepEqual(runs.filter(r => r.italic).map(r => r.text), ['corsivo']);
  assert.deepEqual(runs.filter(r => r.code).map(r => r.text), ['code']);
  assert.equal(runs.find(r => r.link).link, 'https://a.it');
  assert.ok(!runs.map(r => r.text).join('').includes('**'));
});

test('normalizzazione: via preambolo, H1 iniziale e chiusura', () => {
  const md = normalizeModelMarkdown(MODEL_MD, 'Sicurezza Docker');
  assert.ok(md.startsWith('## Sommario'));
  assert.ok(!/Spero che/.test(md));
});

test('parser: blocchi strutturati, nessun carattere markdown nel testo', () => {
  const blocks = parseMarkdown(normalizeModelMarkdown(MODEL_MD));
  const types = blocks.map(b => b.type);
  assert.deepEqual(types, ['h2', 'p', 'h2', 'ul', 'h3', 'table', 'h3', 'code', 'ol']);
  const table = blocks.find(b => b.type === 'table');
  assert.equal(table.header.length, 3);
  assert.equal(table.rows.length, 3);
  const ul = blocks.find(b => b.type === 'ul');
  assert.equal(ul.items.length, 3);   // '-' e '*' nello stesso elenco
  const allText = JSON.stringify(blocks.filter(b => b.type !== 'code'));
  assert.ok(!/\*\*|###|\|---/.test(allText));
});

test('sezioni per Word / anteprima', () => {
  const sections = blocksToSections(parseMarkdown(normalizeModelMarkdown(MODEL_MD)));
  assert.equal(sections[0].heading, 'Sommario');
  assert.match(sections[0].text, /superficie di attacco/);
  assert.ok(sections.some(s => s.bullets.length === 3));
});

test('WinAnsi: emoji eliminate, accenti italiani conservati', () => {
  assert.equal(toWinAnsi('perché → città ✓ 🚀'), 'perché -> città v ');
});

test('rendering PDF A4: file valido, nessuna pagina bianca, testo senza markdown', async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const outputPath = path.join(OUT_DIR, 'test-doc.pdf');
  // Documento lungo per forzare il cambio pagina anche dentro la tabella
  const rows = Array.from({ length: 60 }, (_, i) => `| riga ${i + 1} | valore ${i * 3} | descrizione abbastanza lunga da andare a capo nella cella ${i} |`).join('\n');
  const md = `${normalizeModelMarkdown(MODEL_MD)}\n\n## Inventario\n\n| Nome | Valore | Note |\n|---|---|---|\n${rows}\n`;
  const { pages } = await renderMarkdownPdf({ markdown: md, title: 'Sicurezza dei container Docker', subtitle: 'Test', outputPath });

  const buf = fs.readFileSync(outputPath);
  assert.equal(buf.subarray(0, 5).toString(), '%PDF-');
  assert.ok(pages >= 2 && pages <= 6, `pagine: ${pages}`);
  // Il conteggio degli oggetti /Type /Page deve coincidere: niente pagine bianche extra
  const pageObjects = (buf.toString('latin1').match(/\/Type \/Page\b/g) || []).length;
  assert.equal(pageObjects, pages);
});
