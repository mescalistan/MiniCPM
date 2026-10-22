const test = require('node:test');
const assert = require('node:assert/strict');
const { extractMainText, isPrivateIp, assertPublicUrl, parseDuckHtml } = require('../lib/web-research-service');
const service = require('../lib/web-research-service');

const ARTICLE = `<!doctype html><html><head><title>Linux 6.12 rilasciato</title>
<script>var tracking = 1;</script><style>body{color:red}</style></head>
<body><header><nav><a href="/">Home</a><a href="/news">News</a></nav></header>
<div class="cookie-banner">Accetta i cookie per continuare la navigazione sul sito</div>
<main><article><h1>Linux 6.12 &egrave; disponibile</h1>
<p>Il kernel Linux 6.12 introduce il supporto ufficiale allo scheduler sched_ext basato su eBPF.</p>
<p>Tra le novit&agrave; ci sono anche il real-time PREEMPT_RT integrato nel ramo principale e nuovi driver grafici.</p>
<ul><li>Scheduler estensibile sched_ext scritto in eBPF dal kernel</li><li>PREEMPT_RT finalmente incluso nel kernel principale</li></ul>
</article></main><aside>Articoli correlati che non c'entrano nulla con la notizia</aside>
<footer>Copyright 2026 testata giornalistica di esempio</footer></body></html>`;

test('estrazione testo: niente boilerplate, script, cookie o footer', () => {
  const { title, text } = extractMainText(ARTICLE, 2000);
  assert.equal(title, 'Linux 6.12 rilasciato');
  assert.match(text, /sched_ext basato su eBPF/);
  assert.match(text, /- PREEMPT_RT finalmente incluso/);
  assert.doesNotMatch(text, /tracking|color:red|cookie|Copyright|correlati|News/);
});

test('estrazione testo: troncamento a confine di parola', () => {
  const { text } = extractMainText(ARTICLE, 120);
  assert.ok(text.length <= 125);
  assert.ok(text.endsWith('[…]'));
});

test('parser DuckDuckGo HTML: link uddg decodificati, annunci scartati', () => {
  const html = `
    <div class="result result--ad"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Annuncio</a></div>
    <div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fkernel.org%2F&rut=x">Kernel.org</a>
      <a class="result__snippet">The Linux Kernel Archives</a></div>`;
  const r = parseDuckHtml(html, 5);
  assert.equal(r.length, 1);
  assert.equal(r[0].url, 'https://kernel.org/');
  assert.equal(r[0].title, 'Kernel.org');
});

test('SSRF: IP privati riconosciuti', () => {
  for (const ip of ['127.0.0.1', '10.1.2.3', '172.18.0.5', '192.168.1.180', '169.254.169.254', '::1', '::ffff:192.168.1.1', 'fd00::1']) {
    assert.equal(isPrivateIp(ip), true, ip);
  }
  for (const ip of ['1.1.1.1', '93.184.216.34', '2606:4700::1111']) {
    assert.equal(isPrivateIp(ip), false, ip);
  }
});

test('SSRF: container Docker, localhost e porte interne rifiutati', async () => {
  await assert.rejects(assertPublicUrl('http://docker-proxy:2375/containers/json'), /Porta non consentita/);
  await assert.rejects(assertPublicUrl('http://minicpm2b-n100:8080/v1/models'), /Host interno/);
  await assert.rejects(assertPublicUrl('http://localhost/'), /Host interno/);
  await assert.rejects(assertPublicUrl('http://192.168.1.180/api'), /interno/);
  await assert.rejects(assertPublicUrl('file:///etc/passwd'), /Solo URL http/);
});

test('elenco fonti generato dal codice', () => {
  const md = service.formatSources([{ n: 1, title: 'Kernel [news]', url: 'https://kernel.org' }]);
  assert.equal(md, '\n\n**Fonti consultate:**\n1. [Kernel news](https://kernel.org)');
});
