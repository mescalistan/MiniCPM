/**
 * Web Research & Scraping Service per HOME-SERVER
 * Zero RAM a riposo: nessun browser headless, nessun processo in background.
 * Ricerca: DuckDuckGo HTML -> DuckDuckGo Lite -> Wikipedia (fallback a cascata)
 * Estrazione: cheerio (se installato) con euristica "readability", altrimenti regex.
 *
 * Sicurezza: ogni URL (anche quelli scritti dall'utente) passa da un filtro
 * SSRF. Senza, "riassumi http://docker-proxy:2375/containers/json" farebbe
 * leggere al modello l'API Docker interna.
 */

const dns = require('dns').promises;
const net = require('net');

let cheerio = null;
try { cheerio = require('cheerio'); } catch (_) { /* fallback regex */ }

const USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36';
const MAX_HTML_BYTES = 1.5 * 1024 * 1024;
const CACHE_TTL_MS = 10 * 60 * 1000;
const CACHE_MAX = 30;

const BLOCKED_HOSTS = /^(?:localhost|.*\.local|.*\.internal|.*\.lan|.*\.home|[a-z0-9-]+)$/i; // nomi senza punto = container Docker

// --------------------------------------------------------------------------
// Filtro SSRF
// --------------------------------------------------------------------------
function isPrivateIp(ip) {
  if (net.isIPv4(ip)) {
    const [a, b] = ip.split('.').map(Number);
    return a === 10 || a === 127 || a === 0
      || (a === 169 && b === 254)
      || (a === 172 && b >= 16 && b <= 31)
      || (a === 192 && b === 168)
      || (a === 100 && b >= 64 && b <= 127)
      || a >= 224;
  }
  const v6 = ip.toLowerCase();
  if (v6.startsWith('::ffff:')) return isPrivateIp(v6.slice(7));
  return v6 === '::1' || v6 === '::' || v6.startsWith('fc') || v6.startsWith('fd') || v6.startsWith('fe80');
}

async function assertPublicUrl(rawUrl) {
  let url;
  try { url = new URL(rawUrl); } catch (_) { throw new Error('URL non valido'); }
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Solo URL http/https');
  if (url.port && !['80', '443', '8080', '8443'].includes(url.port)) throw new Error('Porta non consentita');
  const host = url.hostname.replace(/^\[|\]$/g, '');
  if (net.isIP(host)) {
    if (isPrivateIp(host)) throw new Error('Indirizzo di rete interno non consentito');
    return url;
  }
  if (BLOCKED_HOSTS.test(host)) throw new Error('Host interno non consentito');
  const addrs = await dns.lookup(host, { all: true });
  if (!addrs.length || addrs.some(a => isPrivateIp(a.address))) {
    throw new Error('Host risolto su rete interna');
  }
  return url;
}

// --------------------------------------------------------------------------
// Fetch con timeout, limite di dimensione e redirect verificati
// --------------------------------------------------------------------------
async function safeFetchText(rawUrl, { timeoutMs = 5000, accept = 'text/html,application/xhtml+xml', method = 'GET', body = null, headers = {} } = {}) {
  let current = rawUrl;
  const deadline = Date.now() + timeoutMs;
  for (let hop = 0; hop < 4; hop++) {
    await assertPublicUrl(current);
    const remaining = Math.max(500, deadline - Date.now());
    const res = await fetch(current, {
      method,
      body,
      redirect: 'manual',
      headers: { 'User-Agent': USER_AGENT, Accept: accept, 'Accept-Language': 'it-IT,it;q=0.9,en;q=0.7', ...headers },
      signal: AbortSignal.timeout(remaining)
    });
    if (res.status >= 300 && res.status < 400 && res.headers.get('location')) {
      current = new URL(res.headers.get('location'), current).toString();
      method = 'GET';
      body = null;
      continue;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const type = res.headers.get('content-type') || '';
    if (!/text\/|html|xml|json/i.test(type)) throw new Error(`Contenuto non testuale (${type})`);

    // Lettura a blocchi fino a MAX_HTML_BYTES
    const reader = res.body.getReader();
    const chunks = [];
    let size = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.length;
      chunks.push(value);
      if (size >= MAX_HTML_BYTES) { try { await reader.cancel(); } catch (_) {} break; }
    }
    return { text: Buffer.concat(chunks.map(c => Buffer.from(c))).toString('utf8'), finalUrl: current };
  }
  throw new Error('Troppi redirect');
}

// --------------------------------------------------------------------------
// Estrazione del testo principale
// --------------------------------------------------------------------------
const NOISE_SELECTORS = 'script,style,noscript,svg,iframe,form,nav,footer,header,aside,button,select,'
  + '[role=navigation],[role=banner],[role=contentinfo],[aria-hidden=true],'
  + '.cookie,.cookies,.cookie-banner,.newsletter,.share,.social,.related,.comments,.advert,.ads,.sidebar,.breadcrumb,.menu';

function decodeEntities(str) {
  return String(str || '')
    .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(Number(d)))
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)));
}

function normalizeBlocks(blocks, maxChars) {
  const seen = new Set();
  const out = [];
  let total = 0;
  for (const raw of blocks) {
    const b = raw.replace(/\s+/g, ' ').trim();
    if (b.length < 40 && !/^#/.test(b)) continue;     // menu, didascalie, bottoni
    const key = b.toLowerCase().slice(0, 80);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(b);
    total += b.length;
    if (total >= maxChars) break;
  }
  const text = out.join('\n');
  return text.length > maxChars ? `${text.slice(0, maxChars).replace(/\s+\S*$/, '')} […]` : text;
}

function extractWithCheerio(html, maxChars) {
  const $ = cheerio.load(html);
  const title = $('meta[property="og:title"]').attr('content') || $('title').first().text() || '';
  $(NOISE_SELECTORS).remove();

  // Contenitore principale: article/main, altrimenti il nodo con piu' testo nei <p>
  let root = $('article').first();
  if (!root.length) root = $('main, [role=main], #content, .content, .post, .entry-content').first();
  if (!root.length) {
    let best = null;
    let bestScore = 0;
    $('div, section').each((_, el) => {
      const score = $(el).children('p').text().length;
      if (score > bestScore) { bestScore = score; best = el; }
    });
    root = best ? $(best) : $('body');
  }

  const blocks = [];
  root.find('h1, h2, h3, p, li, td, blockquote, pre').each((_, el) => {
    const tag = el.tagName ? el.tagName.toLowerCase() : '';
    const text = $(el).text();
    if (/^h[1-3]$/.test(tag)) blocks.push(`# ${text}`);
    else if (tag === 'li') blocks.push(`- ${text}`);
    else blocks.push(text);
  });
  return { title: title.trim(), text: normalizeBlocks(blocks, maxChars) };
}

function extractWithRegex(html, maxChars) {
  const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  let body = html
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<(script|style|noscript|svg|iframe|form|nav|footer|header|aside)[^>]*>[\s\S]*?<\/\1>/gi, ' ');
  const article = body.match(/<(article|main)[^>]*>([\s\S]*?)<\/\1>/i);
  if (article) body = article[2];
  const blocks = [];
  const re = /<(h[1-3]|p|li|td|blockquote|pre)[^>]*>([\s\S]*?)<\/\1>/gi;
  let m;
  while ((m = re.exec(body)) !== null) {
    const text = decodeEntities(m[2].replace(/<[^>]+>/g, ' '));
    blocks.push(/^h/i.test(m[1]) ? `# ${text}` : (m[1].toLowerCase() === 'li' ? `- ${text}` : text));
  }
  return { title: decodeEntities(titleMatch ? titleMatch[1] : '').trim(), text: normalizeBlocks(blocks, maxChars) };
}

function extractMainText(html, maxChars = 1200) {
  if (!html) return { title: '', text: '' };
  try {
    return cheerio ? extractWithCheerio(html, maxChars) : extractWithRegex(html, maxChars);
  } catch (_) {
    return extractWithRegex(html, maxChars);
  }
}

// --------------------------------------------------------------------------
// Motori di ricerca
// --------------------------------------------------------------------------
function unwrapDuckUrl(href) {
  if (!href) return null;
  try {
    const u = new URL(href, 'https://duckduckgo.com');
    const target = u.searchParams.get('uddg');
    if (target) return decodeURIComponent(target);
    if (/duckduckgo\.com$/.test(u.hostname)) return null;   // link interni / annunci y.js
    return u.toString();
  } catch (_) {
    return null;
  }
}

function parseDuckHtml(html, max) {
  const results = [];
  if (cheerio) {
    const $ = cheerio.load(html);
    $('.result').each((_, el) => {
      if (results.length >= max) return false;
      if ($(el).hasClass('result--ad')) return undefined;
      const a = $(el).find('a.result__a').first();
      const url = unwrapDuckUrl(a.attr('href'));
      if (!url) return undefined;
      results.push({ title: a.text().trim(), url, snippet: $(el).find('.result__snippet').text().replace(/\s+/g, ' ').trim() });
      return undefined;
    });
    // DuckDuckGo Lite: tabella con link .result-link e celle .result-snippet
    if (!results.length) {
      const snippets = $('td.result-snippet').map((_, el) => $(el).text().replace(/\s+/g, ' ').trim()).get();
      $('a.result-link').each((i, el) => {
        if (results.length >= max) return false;
        const url = unwrapDuckUrl($(el).attr('href'));
        if (url) results.push({ title: $(el).text().trim(), url, snippet: snippets[i] || '' });
        return undefined;
      });
    }
    return results;
  }
  const re = /<a[^>]+class="(?:result__a|result-link)"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  let m;
  while ((m = re.exec(html)) !== null && results.length < max) {
    const url = unwrapDuckUrl(decodeEntities(m[1]));
    if (url) results.push({ title: decodeEntities(m[2].replace(/<[^>]+>/g, '')).trim(), url, snippet: '' });
  }
  return results;
}

async function searchDuckDuckGo(query, max) {
  const body = new URLSearchParams({ q: query, kl: 'it-it' }).toString();
  const endpoints = ['https://html.duckduckgo.com/html/', 'https://lite.duckduckgo.com/lite/'];
  for (const endpoint of endpoints) {
    try {
      const { text } = await safeFetchText(endpoint, {
        method: 'POST',
        body,
        timeoutMs: 4500,
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });
      const results = parseDuckHtml(text, max);
      if (results.length) return results;
    } catch (err) {
      console.warn(`[WebResearch] ${endpoint} non disponibile:`, err.message);
    }
  }
  return [];
}

async function searchWikipedia(query, max) {
  try {
    const url = `https://it.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&srlimit=${max}&format=json&utf8=1`;
    const { text } = await safeFetchText(url, { timeoutMs: 3500, accept: 'application/json' });
    const data = JSON.parse(text);
    return (data.query?.search || []).map(r => ({
      title: r.title,
      url: `https://it.wikipedia.org/wiki/${encodeURIComponent(r.title.replace(/ /g, '_'))}`,
      snippet: decodeEntities(String(r.snippet || '').replace(/<[^>]+>/g, ''))
    }));
  } catch (err) {
    console.warn('[WebResearch] Wikipedia non disponibile:', err.message);
    return [];
  }
}

// --------------------------------------------------------------------------
// Servizio
// --------------------------------------------------------------------------
class WebResearchService {
  constructor() {
    this.cache = new Map();
  }

  cacheGet(key) {
    const hit = this.cache.get(key);
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) return hit.value;
    this.cache.delete(key);
    return null;
  }

  cacheSet(key, value) {
    if (this.cache.size >= CACHE_MAX) this.cache.delete(this.cache.keys().next().value);
    this.cache.set(key, { at: Date.now(), value });
  }

  async search(query, max = 4) {
    const results = await searchDuckDuckGo(query, max);
    if (results.length) return results;
    return searchWikipedia(query, max);
  }

  async scrapeUrl(url, maxChars = 1200) {
    try {
      const { text: html, finalUrl } = await safeFetchText(url, { timeoutMs: 5000 });
      const { title, text } = extractMainText(html, maxChars);
      return text ? { url: finalUrl, title, text } : null;
    } catch (err) {
      console.warn(`[WebResearch] Scrape fallito per ${url}:`, err.message);
      return null;
    }
  }

  /**
   * Ricerca + scraping parallelo delle prime pagine.
   * @returns {{ sources: Array<{n,title,url,text}>, context: string, query: string }}
   *   `context` e' il blocco [FONTI] numerato da passare al modello.
   */
  async searchAndScrape(query, { maxPages = 3, explicitUrl = null, charsPerSource = 900, deadlineMs = 9000 } = {}) {
    const cacheKey = `${explicitUrl || ''}|${query}|${maxPages}`;
    const cached = this.cacheGet(cacheKey);
    if (cached) return cached;

    let candidates;
    if (explicitUrl) {
      candidates = [{ title: '', url: explicitUrl, snippet: '' }];
    } else {
      candidates = await this.search(query, maxPages + 2);
    }

    const picked = candidates.slice(0, explicitUrl ? 1 : maxPages);
    const budget = explicitUrl ? charsPerSource * 2 : charsPerSource;
    // Scadenza globale: le pagine lente vengono sostituite dallo snippet
    const timer = new Promise(resolve => setTimeout(() => resolve('timeout'), deadlineMs));
    const scraped = await Promise.all(picked.map(c => Promise.race([this.scrapeUrl(c.url, budget), timer.then(() => null)])));

    const sources = [];
    picked.forEach((c, i) => {
      const page = scraped[i];
      const text = page?.text || c.snippet;
      if (!text) return;
      sources.push({
        n: sources.length + 1,
        title: (page?.title || c.title || new URL(c.url).hostname).slice(0, 120),
        url: page?.url || c.url,
        text
      });
    });

    const context = sources.length
      ? sources.map(s => `[${s.n}] ${s.title}\n${s.text}`).join('\n\n')
      : 'Nessuna fonte disponibile.';

    const result = { query, sources, context };
    if (sources.length) this.cacheSet(cacheKey, result);
    return result;
  }

  /**
   * Elenco fonti in Markdown, generato dal codice (mai dal modello).
   */
  formatSources(sources) {
    if (!Array.isArray(sources) || !sources.length) return '';
    return `\n\n**Fonti consultate:**\n${sources.map(s => `${s.n}. [${s.title.replace(/[[\]]/g, '')}](${s.url})`).join('\n')}`;
  }
}

module.exports = new WebResearchService();
module.exports.WebResearchService = WebResearchService;
module.exports.extractMainText = extractMainText;
module.exports.isPrivateIp = isPrivateIp;
module.exports.assertPublicUrl = assertPublicUrl;
module.exports.parseDuckHtml = parseDuckHtml;
