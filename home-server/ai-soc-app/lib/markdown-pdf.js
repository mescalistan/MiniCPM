/**
 * Markdown -> PDF A4 professionale (pdfkit)
 *
 * Il vecchio createPdf() stampava il testo del modello cosi' com'era: '**',
 * '###' e '|---|' finivano nel PDF. Qui il Markdown viene prima convertito in
 * blocchi strutturati (titoli, paragrafi, elenchi, tabelle, codice) con testo
 * inline in "run" (grassetto, corsivo, codice, link), poi impaginato.
 *
 * Correzioni rispetto al renderer precedente:
 * - bufferPages: true, altrimenti switchToPage() sulle pagine gia' scritte fallisce
 * - il pie' di pagina si scrive con margine inferiore azzerato: scrivere sotto
 *   il margine fa aggiungere a pdfkit una pagina bianca per ogni pagina
 * - tabelle con altezza di riga calcolata e intestazione ripetuta a cambio pagina
 */

const fs = require('fs');

let PDFDocument = null;
function getPDFDocument() {
  if (!PDFDocument) PDFDocument = require('pdfkit');
  return PDFDocument;
}

// --------------------------------------------------------------------------
// Font: TTF Unicode se presente nel container, altrimenti Helvetica (WinAnsi)
// --------------------------------------------------------------------------
const TTF_CANDIDATES = [
  ['/usr/share/fonts/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/dejavu/DejaVuSans-Oblique.ttf', '/usr/share/fonts/dejavu/DejaVuSansMono.ttf'],
  ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf']
];

function resolveFonts() {
  for (const set of TTF_CANDIDATES) {
    if (set.every(f => fs.existsSync(f))) {
      return { unicode: true, regular: set[0], bold: set[1], italic: set[2], mono: set[3] };
    }
  }
  return { unicode: false, regular: 'Helvetica', bold: 'Helvetica-Bold', italic: 'Helvetica-Oblique', mono: 'Courier' };
}

// Caratteri fuori da WinAnsi (cp1252) che i modelli usano spesso
const WINANSI_MAP = { '→': '->', '←': '<-', '⇒': '=>', '✓': 'v', '✔': 'v', '✗': 'x', '✘': 'x', '≥': '>=', '≤': '<=', '≠': '!=', '×': 'x', '•': '•', '−': '-', '‑': '-', ' ': ' ' };
const CP1252_EXTRA = '€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ';

function toWinAnsi(str) {
  let out = '';
  for (const ch of String(str)) {
    if (WINANSI_MAP[ch] !== undefined) { out += WINANSI_MAP[ch]; continue; }
    const code = ch.codePointAt(0);
    if ((code >= 0x20 && code <= 0x7e) || (code >= 0xa0 && code <= 0xff) || ch === '\n' || CP1252_EXTRA.includes(ch)) {
      out += ch;
    }
    // emoji e simboli non rappresentabili vengono scartati
  }
  return out;
}

function stripEmoji(str) {
  return String(str).replace(/[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}\u{200D}]/gu, '');
}

// --------------------------------------------------------------------------
// Parser Markdown -> blocchi
// --------------------------------------------------------------------------

/**
 * Testo inline -> run [{ text, bold, italic, code, link }]
 * Asterischi e cancelletti spaiati vengono eliminati, non stampati.
 */
function parseInline(text) {
  const runs = [];
  const re = /(\*\*|__)(.+?)\1|`([^`]+)`|\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)|(?<![\w*])[*_](?!\s)(.+?)(?<!\s)[*_](?![\w*])/g;
  let last = 0;
  let m;
  const push = (t, style = {}) => {
    const clean = t.replace(/\*{1,3}|_{2,}|(^|\s)#{1,6}\s/g, '$1');
    if (clean) runs.push({ text: clean, bold: false, italic: false, code: false, link: null, ...style });
  };
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) push(text.slice(last, m.index));
    if (m[2] !== undefined) push(m[2], { bold: true });
    else if (m[3] !== undefined) runs.push({ text: m[3], bold: false, italic: false, code: true, link: null });
    else if (m[4] !== undefined) push(m[4], { link: m[5] });
    else if (m[6] !== undefined) push(m[6], { italic: true });
    last = re.lastIndex;
  }
  if (last < text.length) push(text.slice(last));
  return runs.length ? runs : [{ text: '', bold: false, italic: false, code: false, link: null }];
}

function plainText(runs) {
  return runs.map(r => r.text).join('');
}

function splitTableRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
}

/**
 * Markdown -> [{ type: h1|h2|h3|p|ul|ol|table|code|quote|hr, ... }]
 */
function parseMarkdown(markdown) {
  const lines = String(markdown || '').replace(/\r\n?/g, '\n').split('\n');
  const blocks = [];
  let para = [];

  const flushPara = () => {
    if (para.length) {
      blocks.push({ type: 'p', runs: parseInline(para.join(' ').replace(/\s+/g, ' ').trim()) });
      para = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const t = line.trim();

    if (!t) { flushPara(); continue; }

    // Blocco di codice
    if (/^```/.test(t)) {
      flushPara();
      const code = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
      blocks.push({ type: 'code', text: code.join('\n') });
      continue;
    }

    // Titoli
    const h = t.match(/^(#{1,6})\s+(.*?)\s*#*$/);
    if (h) {
      flushPara();
      const level = Math.min(h[1].length, 3);
      blocks.push({ type: `h${level}`, runs: parseInline(h[2].replace(/\*\*/g, '')) });
      continue;
    }
    // Riga interamente in grassetto -> sottotitolo (i 2B lo fanno spesso)
    const boldLine = t.match(/^\*\*([^*]{3,80})\*\*:?$/);
    if (boldLine) {
      flushPara();
      blocks.push({ type: 'h3', runs: parseInline(boldLine[1]) });
      continue;
    }

    // Linea orizzontale
    if (/^(?:-{3,}|\*{3,}|_{3,})$/.test(t)) { flushPara(); blocks.push({ type: 'hr' }); continue; }

    // Tabella: riga con | seguita da separatore |---|
    if (t.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[i + 1])) {
      flushPara();
      const header = splitTableRow(t);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        const cells = splitTableRow(lines[i]);
        while (cells.length < header.length) cells.push('');
        rows.push(cells.slice(0, header.length));
        i++;
      }
      i--;
      blocks.push({ type: 'table', header: header.map(parseInline), rows: rows.map(r => r.map(parseInline)) });
      continue;
    }

    // Elenchi
    const ul = t.match(/^[-*•+]\s+(.*)$/);
    const ol = t.match(/^(\d{1,2})[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const type = ul ? 'ul' : 'ol';
      const items = [];
      let j = i;
      while (j < lines.length) {
        const lt = lines[j].trim();
        const m = type === 'ul' ? lt.match(/^[-*•+]\s+(.*)$/) : lt.match(/^\d{1,2}[.)]\s+(.*)$/);
        if (m) {
          items.push({ runs: parseInline(m[1]), depth: /^\s{2,}/.test(lines[j]) ? 1 : 0 });
          j++;
        } else if (lt && /^\s{2,}/.test(lines[j]) && items.length && !/^[-*•+]|\d+[.)]/.test(lt)) {
          // riga di continuazione indentata
          const lastItem = items[items.length - 1];
          lastItem.runs = parseInline(`${plainText(lastItem.runs)} ${lt}`);
          j++;
        } else {
          break;
        }
      }
      i = j - 1;
      blocks.push({ type, items });
      continue;
    }

    // Citazione
    if (/^>\s?/.test(t)) {
      flushPara();
      blocks.push({ type: 'quote', runs: parseInline(t.replace(/^>\s?/, '')) });
      continue;
    }

    para.push(t);
  }
  flushPara();
  return blocks;
}

/**
 * Normalizza l'output del modello prima del parsing:
 * - toglie un eventuale titolo H1 iniziale (il titolo lo mette il renderer)
 * - abbassa gli altri H1 a H2
 * - toglie preamboli tipo "Ecco il documento:" e chiusure "Spero ti sia utile"
 */
function normalizeModelMarkdown(markdown, title = '') {
  let md = String(markdown || '').trim();
  md = md.replace(/^(?:certo|ecco|di seguito)[^\n]{0,80}:\s*\n+/i, '');
  const firstH1 = md.match(/^#\s+(.+)\n/);
  if (firstH1) md = md.slice(firstH1[0].length);
  md = md.replace(/^#\s+/gm, '## ');
  md = md.replace(/\n+(?:spero (?:che )?(?:questo|ti|le)[^\n]*|fammi sapere[^\n]*|se hai bisogno[^\n]*)\s*$/i, '');
  return md.trim() || `## Sommario\n\n${title}`;
}

// --------------------------------------------------------------------------
// Renderer
// --------------------------------------------------------------------------
const THEME = {
  accent: '#0369A1',
  accentLight: '#E0F2FE',
  text: '#1E293B',
  muted: '#64748B',
  rule: '#CBD5E1',
  tableHeader: '#0F4C75',
  tableStripe: '#F1F5F9',
  codeBg: '#F1F5F9'
};

const MARGINS = { top: 64, bottom: 64, left: 56, right: 56 };

class PdfComposer {
  constructor(doc, fonts) {
    this.doc = doc;
    this.fonts = fonts;
    this.clean = fonts.unicode ? stripEmoji : (s => toWinAnsi(stripEmoji(s)));
    this.left = MARGINS.left;
    this.width = doc.page.width - MARGINS.left - MARGINS.right;
  }

  get bottom() {
    return this.doc.page.height - MARGINS.bottom;
  }

  ensureSpace(height) {
    if (this.doc.y + height > this.bottom) {
      this.doc.addPage();
      this.doc.y = MARGINS.top;
    }
  }

  fontFor(run) {
    if (run.code) return this.fonts.mono;
    if (run.bold) return this.fonts.bold;
    if (run.italic) return this.fonts.italic;
    return this.fonts.regular;
  }

  /** Altezza stimata di un insieme di run alla larghezza data. */
  measureRuns(runs, size, width, lineGap = 3) {
    this.doc.font(this.fonts.regular).fontSize(size);
    return this.doc.heightOfString(this.clean(plainText(runs)) || ' ', { width, lineGap });
  }

  /** Scrive run con stili diversi sulla stessa riga logica (continued). */
  writeRuns(runs, { x = this.left, width = this.width, size = 10.5, color = THEME.text, lineGap = 3, align = 'left' } = {}) {
    const doc = this.doc;
    const parts = runs.map(r => ({ ...r, text: this.clean(r.text) })).filter(r => r.text);
    if (!parts.length) return;
    parts.forEach((r, idx) => {
      const opts = { width, lineGap, align, continued: idx < parts.length - 1 };
      if (r.link) Object.assign(opts, { link: r.link, underline: true });
      else Object.assign(opts, { link: null, underline: false });
      doc.font(this.fontFor(r)).fontSize(r.code ? size - 1 : size).fillColor(r.link ? THEME.accent : color);
      if (idx === 0) doc.text(r.text, x, doc.y, opts);
      else doc.text(r.text, opts);
    });
  }

  heading(block) {
    const sizes = { h1: 16, h2: 13.5, h3: 11.5 };
    const size = sizes[block.type];
    const h = this.measureRuns(block.runs, size, this.width);
    // Il titolo non resta da solo in fondo alla pagina: servono ~3 righe dopo
    this.ensureSpace(h + 50);
    this.doc.moveDown(block.type === 'h3' ? 0.4 : 0.8);
    this.writeRuns(block.runs.map(r => ({ ...r, bold: true })), { size, color: block.type === 'h3' ? THEME.text : THEME.accent });
    if (block.type === 'h2') {
      const y = this.doc.y + 2;
      this.doc.moveTo(this.left, y).lineTo(this.left + this.width, y).lineWidth(0.6).strokeColor(THEME.rule).stroke();
      this.doc.y = y + 4;
    }
    this.doc.moveDown(0.3);
  }

  paragraph(block) {
    const h = this.measureRuns(block.runs, 10.5, this.width);
    this.ensureSpace(Math.min(h, 40));
    this.writeRuns(block.runs, { size: 10.5 });
    this.doc.moveDown(0.6);
  }

  list(block) {
    block.items.forEach((item, idx) => {
      const indent = 14 + item.depth * 16;
      const marker = block.type === 'ol' ? `${idx + 1}.` : (item.depth ? '–' : '•');
      const textX = this.left + indent + 12;
      const textW = this.width - indent - 12;
      const h = this.measureRuns(item.runs, 10.5, textW, 2);
      this.ensureSpace(Math.min(h, 30));
      const y = this.doc.y;
      this.doc.font(this.fonts.bold).fontSize(10.5).fillColor(THEME.accent).text(this.clean(marker), this.left + indent - 2, y, { width: 14, lineBreak: false });
      this.doc.y = y;
      this.writeRuns(item.runs, { x: textX, width: textW, size: 10.5, lineGap: 2 });
      this.doc.moveDown(0.25);
    });
    this.doc.x = this.left;
    this.doc.moveDown(0.4);
  }

  table(block) {
    const doc = this.doc;
    const cols = block.header.length;
    const pad = 5;
    const size = 9;

    // Larghezze proporzionali al contenuto, con minimo leggibile
    const lens = block.header.map((hdr, c) => Math.max(
      plainText(hdr).length,
      ...block.rows.map(r => Math.min(plainText(r[c] || []).length, 60))
    ) + 4);
    const total = lens.reduce((a, b) => a + b, 0);
    let widths = lens.map(l => Math.max(55, (l / total) * this.width));
    const scale = this.width / widths.reduce((a, b) => a + b, 0);
    widths = widths.map(w => w * scale);

    const rowHeight = (cells, bold) => Math.max(...cells.map((cell, c) => {
      doc.font(bold ? this.fonts.bold : this.fonts.regular).fontSize(size);
      return doc.heightOfString(this.clean(plainText(cell)) || ' ', { width: widths[c] - pad * 2, lineGap: 1 });
    })) + pad * 2;

    const drawRow = (cells, { header = false, stripe = false } = {}) => {
      const h = rowHeight(cells, header);
      const y = doc.y;
      if (header) doc.rect(this.left, y, this.width, h).fill(THEME.tableHeader);
      else if (stripe) doc.rect(this.left, y, this.width, h).fill(THEME.tableStripe);
      let x = this.left;
      cells.forEach((cell, c) => {
        doc.font(header ? this.fonts.bold : this.fonts.regular).fontSize(size).fillColor(header ? '#FFFFFF' : THEME.text);
        doc.text(this.clean(plainText(cell)), x + pad, y + pad, { width: widths[c] - pad * 2, lineGap: 1 });
        x += widths[c];
      });
      doc.moveTo(this.left, y + h).lineTo(this.left + this.width, y + h).lineWidth(0.4).strokeColor(THEME.rule).stroke();
      doc.y = y + h;
    };

    this.ensureSpace(rowHeight(block.header, true) + rowHeight(block.rows[0] || block.header, false) + 6);
    doc.moveDown(0.3);
    drawRow(block.header, { header: true });
    block.rows.forEach((row, r) => {
      const h = rowHeight(row, false);
      if (doc.y + h > this.bottom) {
        doc.addPage();
        doc.y = MARGINS.top;
        drawRow(block.header, { header: true });   // intestazione ripetuta
      }
      drawRow(row, { stripe: r % 2 === 1 });
    });
    doc.x = this.left;
    doc.moveDown(0.8);
  }

  code(block) {
    const doc = this.doc;
    const pad = 8;
    const lines = this.clean(block.text).split('\n');
    // Spezza il blocco in pezzi che stanno in una pagina
    let chunk = [];
    const flush = () => {
      if (!chunk.length) return;
      const text = chunk.join('\n');
      doc.font(this.fonts.mono).fontSize(8.5);
      const h = doc.heightOfString(text, { width: this.width - pad * 2, lineGap: 1 }) + pad * 2;
      this.ensureSpace(h);
      const y = doc.y;
      doc.rect(this.left, y, this.width, h).fill(THEME.codeBg);
      doc.fillColor(THEME.text).font(this.fonts.mono).fontSize(8.5).text(text, this.left + pad, y + pad, { width: this.width - pad * 2, lineGap: 1 });
      doc.y = y + h;
      chunk = [];
    };
    for (const line of lines) {
      chunk.push(line);
      if (chunk.length >= 40) flush();
    }
    flush();
    doc.x = this.left;
    doc.moveDown(0.6);
  }

  quote(block) {
    const h = this.measureRuns(block.runs, 10.5, this.width - 16);
    this.ensureSpace(h + 4);
    const y = this.doc.y;
    this.doc.rect(this.left, y, 3, h).fill(THEME.accent);
    this.doc.y = y;
    this.writeRuns(block.runs.map(r => ({ ...r, italic: !r.bold })), { x: this.left + 12, width: this.width - 16, color: THEME.muted });
    this.doc.x = this.left;
    this.doc.moveDown(0.6);
  }

  hr() {
    this.ensureSpace(12);
    const y = this.doc.y + 4;
    this.doc.moveTo(this.left, y).lineTo(this.left + this.width, y).lineWidth(0.6).strokeColor(THEME.rule).stroke();
    this.doc.y = y + 8;
  }

  render(blocks) {
    for (const b of blocks) {
      if (b.type === 'h1' || b.type === 'h2' || b.type === 'h3') this.heading(b);
      else if (b.type === 'p') this.paragraph(b);
      else if (b.type === 'ul' || b.type === 'ol') this.list(b);
      else if (b.type === 'table') this.table(b);
      else if (b.type === 'code') this.code(b);
      else if (b.type === 'quote') this.quote(b);
      else if (b.type === 'hr') this.hr();
    }
  }
}

/**
 * Intestazione della prima pagina: barra, titolo, sottotitolo, metadati.
 */
function drawCover(doc, composer, { title, subtitle, author }) {
  const { left, width } = composer;
  doc.rect(left, MARGINS.top - 20, width, 5).fill(THEME.accent);
  doc.y = MARGINS.top;
  doc.font(composer.fonts.bold).fontSize(22).fillColor('#0F172A').text(composer.clean(title), left, doc.y, { width });
  if (subtitle) {
    doc.moveDown(0.2);
    doc.font(composer.fonts.regular).fontSize(12).fillColor(THEME.muted).text(composer.clean(subtitle), { width });
  }
  doc.moveDown(0.4);
  const date = new Date().toLocaleString('it-IT', { dateStyle: 'long', timeStyle: 'short' });
  doc.font(composer.fonts.regular).fontSize(8.5).fillColor('#94A3B8').text(composer.clean(`${author}   |   ${date}`), { width });
  doc.moveDown(0.6);
  const y = doc.y;
  doc.moveTo(left, y).lineTo(left + width, y).lineWidth(1).strokeColor(THEME.rule).stroke();
  doc.y = y + 14;
}

/**
 * Intestazione corrente e pie' di pagina su tutte le pagine.
 * Il margine inferiore viene azzerato durante la scrittura: senza, pdfkit
 * aggiunge una pagina vuota ogni volta che si scrive sotto il margine.
 */
function drawPageChrome(doc, composer, { title }) {
  const range = doc.bufferedPageRange();
  for (let i = range.start; i < range.start + range.count; i++) {
    doc.switchToPage(i);
    const savedBottom = doc.page.margins.bottom;
    doc.page.margins.bottom = 0;
    if (i > range.start) {
      doc.font(composer.fonts.regular).fontSize(8).fillColor('#94A3B8')
        .text(composer.clean(title), composer.left, MARGINS.top - 34, { width: composer.width, align: 'left', lineBreak: false });
      doc.moveTo(composer.left, MARGINS.top - 20).lineTo(composer.left + composer.width, MARGINS.top - 20).lineWidth(0.4).strokeColor(THEME.rule).stroke();
    }
    doc.font(composer.fonts.regular).fontSize(8).fillColor('#94A3B8')
      .text(composer.clean(`HOME-SERVER AI Studio   •   Pagina ${i + 1} di ${range.count}`), composer.left, doc.page.height - 40, { width: composer.width, align: 'center', lineBreak: false });
    doc.page.margins.bottom = savedBottom;
  }
}

/**
 * Genera il PDF su disco.
 * @returns {Promise<{ pages: number, blocks: Array }>}
 */
function renderMarkdownPdf({ markdown, title = 'Documento', subtitle = '', author = 'C-3PO • HOME-SERVER AI Studio', outputPath }) {
  const PDFKit = getPDFDocument();
  const fonts = resolveFonts();
  const blocks = parseMarkdown(markdown);

  return new Promise((resolve, reject) => {
    try {
      const doc = new PDFKit({
        size: 'A4',
        margins: MARGINS,
        bufferPages: true,
        info: { Title: title, Author: author, Subject: subtitle || title, Creator: 'HOME-SERVER AI Studio' }
      });
      if (fonts.unicode) {
        doc.registerFont('U-Regular', fonts.regular);
        doc.registerFont('U-Bold', fonts.bold);
        doc.registerFont('U-Italic', fonts.italic);
        doc.registerFont('U-Mono', fonts.mono);
        Object.assign(fonts, { regular: 'U-Regular', bold: 'U-Bold', italic: 'U-Italic', mono: 'U-Mono' });
      }

      const stream = fs.createWriteStream(outputPath);
      stream.on('error', reject);
      stream.on('finish', () => resolve({ pages: pageCount, blocks }));
      doc.pipe(stream);

      const composer = new PdfComposer(doc, fonts);
      drawCover(doc, composer, { title, subtitle, author });
      composer.render(blocks);
      drawPageChrome(doc, composer, { title });
      const pageCount = doc.bufferedPageRange().count;
      doc.end();
    } catch (err) {
      reject(err);
    }
  });
}

/**
 * Blocchi -> sezioni { heading, text, bullets } (formato di MediaStudio.createWord
 * e della card di anteprima nel frontend).
 */
function blocksToSections(blocks) {
  const sections = [];
  let cur = null;
  const ensure = () => { if (!cur) { cur = { heading: '', text: '', bullets: [] }; sections.push(cur); } return cur; };
  for (const b of blocks) {
    if (b.type === 'h2' || b.type === 'h1' || b.type === 'h3') {
      cur = { heading: plainText(b.runs), text: '', bullets: [] };
      sections.push(cur);
    } else if (b.type === 'p' || b.type === 'quote') {
      const s = ensure();
      s.text = s.text ? `${s.text}\n${plainText(b.runs)}` : plainText(b.runs);
    } else if (b.type === 'ul' || b.type === 'ol') {
      ensure().bullets.push(...b.items.map(it => plainText(it.runs)));
    } else if (b.type === 'table') {
      ensure().bullets.push(...b.rows.map(r => r.map(plainText).join(' | ')));
    } else if (b.type === 'code') {
      const s = ensure();
      s.text = s.text ? `${s.text}\n${b.text}` : b.text;
    }
  }
  return sections.filter(s => s.heading || s.text || s.bullets.length);
}

module.exports = { renderMarkdownPdf, parseMarkdown, parseInline, normalizeModelMarkdown, blocksToSections, toWinAnsi };
