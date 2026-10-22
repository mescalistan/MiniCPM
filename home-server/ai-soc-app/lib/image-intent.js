/**
 * INTENT_IMAGE: soggetto -> prompt inglese per Pollinations.ai
 *
 * L'LLM fa solo la traduzione (temperature 0, 60 token max) e il risultato
 * viene validato; la risposta all'utente e' un template fisso. Cosi' il
 * modello non ha occasione di inventare link o descrizioni dell'immagine.
 */

const { buildMessages } = require('./prompt-builder');

// Stili riconosciuti in italiano -> suffisso inglese per il generatore
const STYLE_MAP = [
  [/\b(?:foto ?realistic[ao]|realistic[ao]|fotografic[ao])\b/i, 'photorealistic, natural light, 35mm photo'],
  [/\bacquerell[oi]\b/i, 'watercolor painting'],
  [/\b(?:ad olio|a olio|dipinto ad olio)\b/i, 'oil painting'],
  [/\b(?:cartoon|cartone animato|fumetto)\b/i, 'cartoon style, clean lines'],
  [/\banime\b|\bmanga\b/i, 'anime style'],
  [/\bpixel ?art\b/i, 'pixel art'],
  [/\b(?:3d|tridimensional[ei])\b/i, '3d render, octane render'],
  [/\b(?:minimal[ei]?|minimalist[ao])\b/i, 'minimalist, flat design'],
  [/\b(?:futuristic[ao]|cyberpunk|fantascienza)\b/i, 'futuristic, cyberpunk, neon lights'],
  [/\b(?:schizzo|a matita|bozzetto)\b/i, 'pencil sketch']
];

// Dizionario minimo per il fallback senza LLM (soggetti piu' comuni)
const FALLBACK_DICT = {
  gatto: 'cat', gatti: 'cats', cane: 'dog', cani: 'dogs', astronauta: 'astronaut', spazio: 'space',
  montagna: 'mountain', montagne: 'mountains', mare: 'sea', spiaggia: 'beach', tramonto: 'sunset',
  alba: 'sunrise', notte: 'night', citta: 'city', città: 'city', foresta: 'forest', bosco: 'woods',
  drago: 'dragon', castello: 'castle', robot: 'robot', server: 'server', computer: 'computer',
  casa: 'house', auto: 'car', macchina: 'car', fiore: 'flower', fiori: 'flowers', albero: 'tree',
  luna: 'moon', sole: 'sun', stelle: 'stars', neve: 'snow', pioggia: 'rain', lago: 'lake',
  ritratto: 'portrait', donna: 'woman', uomo: 'man', bambino: 'child', cavallo: 'horse',
  uccello: 'bird', pesce: 'fish', illuminato: 'illuminated', buio: 'dark', futuristico: 'futuristic',
  futuristica: 'futuristic', rosso: 'red', blu: 'blue', verde: 'green', giallo: 'yellow', nero: 'black',
  bianco: 'white', con: 'with', di: 'of', nel: 'in the', nella: 'in the', sul: 'on the', sulla: 'on the',
  e: 'and', un: 'a', una: 'a', il: 'the', la: 'the', lo: 'the'
};

/**
 * "C-3PO, creami un'immagine di un gatto astronauta" -> "gatto astronauta"
 */
function extractImageSubject(prompt) {
  let s = String(prompt || '')
    .replace(/^\s*c-?3po[\s,:]*/i, '')
    .replace(/\b(?:per favore|per piacere|grazie|ti prego)\b/gi, ' ')
    .replace(/\b(?:puoi|potresti|riesci a|vorrei che|voglio che)\b/gi, ' ')
    .replace(/\b(?:crea|creami|creare|genera|generami|generare|fai|fammi|fare|disegna|disegnami|illustra|illustrami|dipingi|dipingimi|produci|realizza|realizzami|mostra|mostrami|voglio|vorrei)\b/gi, ' ')
    .replace(/\b(?:un'|una|uno|un|dell'|delle|degli|dei|l'|il|lo|la|le|gli|i)\s*(?=(?:immagine|immagini|illustrazione|illustrazioni|foto|fotografia|disegno|disegni|logo|wallpaper|sfondo|ritratto|dipinto|quadro|rendering|render|poster|locandina|icona|avatar|banner|vignetta)\b)/gi, ' ')
    .replace(/\b(?:immagine|immagini|illustrazione|illustrazioni|foto|fotografia|disegno|disegni|wallpaper|sfondo|dipinto|quadro|rendering|render|poster|locandina|icona|avatar|banner|vignetta)\b/gi, ' ')
    .replace(/^\s*(?:che (?:rappresenti|raffiguri|mostri)|raffigurante|rappresentante|con|di|del|della|dello|dei|degli|delle|su|sul|sulla)\b/i, ' ')
    .replace(/[🎨🖼️📝📊🛡️]/gu, ' ')
    .replace(/[?!."«»]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  // Una seconda passata per "di un ..." rimasti in testa
  s = s.replace(/^(?:di|del|della|dello|dei|degli|delle|con|su)\s+/i, '').replace(/^(?:un'|una|uno|un)\s*/i, '').trim();
  return s || 'paesaggio digitale futuristico';
}

function detectStyle(prompt) {
  for (const [re, en] of STYLE_MAP) {
    if (re.test(prompt)) return en;
  }
  return 'digital art';
}

function dictionaryTranslate(subject) {
  return subject
    .toLowerCase()
    .replace(/[^\p{L}\s'-]/gu, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .map(w => FALLBACK_DICT[w] || w)
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Valida l'output della traduzione: una riga, caratteri latini, niente
 * spiegazioni ("Here is...") o link.
 */
function cleanTranslation(raw) {
  if (!raw) return null;
  let t = String(raw).split('\n').map(l => l.trim()).find(Boolean) || '';
  t = t.replace(/^(?:english(?: prompt)?|prompt|translation)\s*:\s*/i, '').replace(/^["'`]+|["'`]+$/g, '').trim();
  if (t.length < 3 || t.length > 300) return null;
  if (/https?:|<|>|\]\(/i.test(t)) return null;
  if (/^(?:here|sure|certainly|ecco|certo)\b/i.test(t)) return null;
  const latin = (t.match(/[a-z0-9 ,.'()-]/gi) || []).length;
  if (latin / t.length < 0.9) return null;
  return t;
}

/**
 * Traduce il soggetto con l'LLM; se l'output non passa la validazione usa il dizionario.
 */
async function translateSubject(subject, engineClient) {
  if (engineClient) {
    try {
      const raw = await engineClient.chatCompletion({
        messages: buildMessages('translate', subject),
        profile: 'translate'
      });
      const ok = cleanTranslation(raw);
      if (ok) return { english: ok, method: 'llm' };
    } catch (_) {}
  }
  return { english: dictionaryTranslate(subject), method: 'dictionary' };
}

/**
 * Prompt finale per Pollinations: soggetto + stile + qualita', < 400 caratteri.
 */
function buildImagePrompt(english, style) {
  return `${english}, ${style}, highly detailed, sharp focus, professional composition`.slice(0, 400);
}

module.exports = { extractImageSubject, detectStyle, translateSubject, cleanTranslation, dictionaryTranslate, buildImagePrompt };
