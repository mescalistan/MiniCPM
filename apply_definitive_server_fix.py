import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.1.180', username='gabibbo', password='225569')

sftp = ssh.open_sftp()
with sftp.open('/home/gabibbo/HOME-SERVER/ai-soc-app/server.js', 'r') as f:
    code = f.read().decode('utf-8')

# Ensure webResearchService is imported
if "const webResearchService = require('./lib/web-research-service');" not in code:
    code = code.replace(
        "const MiniCpmSyncService = require('./lib/minicpm-sync');",
        "const MiniCpmSyncService = require('./lib/minicpm-sync');\nconst webResearchService = require('./lib/web-research-service');"
    )

start_marker = "app.post(['/api/ai-chat/c3po-query', '/api/soc/chat/c3po']"
end_marker = "app.get('/api/soc/reports/settings'"

start_idx = code.find(start_marker)
end_idx = code.find(end_marker, start_idx)

if start_idx == -1 or end_idx == -1:
    print("Markers not found! start:", start_idx, "end:", end_idx)
    sys.exit(1)

new_handler = """app.post(['/api/ai-chat/c3po-query', '/api/soc/chat/c3po'], async (req, res) => {
  try {
    const { username, messages, prompt, attachments } = req.body || {};
    let userPrompt = String(prompt || '').trim();
    if (!userPrompt && Array.isArray(messages) && messages.length > 0) {
      const lastUser = [...messages].reverse().find(m => m.role === 'user');
      if (lastUser) userPrompt = String(lastUser.content || '').trim();
    }
    if (!userPrompt && (!Array.isArray(attachments) || attachments.length === 0)) {
      return res.status(400).json({ error: 'Prompt o allegato obbligatorio' });
    }

    // 1. FAST-PATH EXECUTION (< 15ms TTFT)
    if (userPrompt && (!Array.isArray(attachments) || attachments.length === 0)) {
      const fastAnswer = await fastPathEngine.checkFastPath(userPrompt, username);
      if (fastAnswer) {
        return res.json({ success: true, response: fastAnswer, reply: fastAnswer, fastPath: true });
      }
    }

    // 2. DETERMINISTIC INTENT DETECTION
    const isImageIntent = /(?:crea|creami|genera|generami|fai|fammi|disegna|disegnami|illustra|illustrami|produci|mostra|mostrami|voglio)\\s+(?:(?:un(?:'|\\s+|o\\s+|a\\s+))?|(?:delle\\s+|dei\\s+|degli\\s+))?(?:immagine|immagini|illustrazione|illustrazioni|foto|fotografia|disegno|disegni|logo|grafica|sfondo|wallpaper|banner|rendering)/i.test(userPrompt)
      || /(?:illustrazione|immagine|grafica|disegno)\\s+(?:futuristica|digitale|del|di|per)/i.test(userPrompt);

    const isPdfIntent = /(?:crea|genera|fai|prepara|esporta|scarica|stampa|redigi|scrivi)\\s+(?:un\\s+|il\\s+)?(?:file\\s+|documento\\s+|report\\s+)?(?:in\\s+)?(?:pdf|documento|report)/i.test(userPrompt)
      || /(?:report|documento|promemoria|riassunto|scheda|guida)\\s+(?:in\\s+|come\\s+)?pdf/i.test(userPrompt);

    const isWebSearch = /https?:\\/\\/[^\\s]+/i.test(userPrompt)
      || /(?:cerca|cercami|trova|trovami|cerca su internet|cerca sul web|ultime notizie|novità su|notizie su|informazioni su|prezzo di|chi è|cosa è successo|aggiornamenti)\\b/i.test(userPrompt);

    const isServerQuery = /(?:server|container|docker|ram|cpu|temperatura|disco|spazio|soc|sentinel|sicurezza|fail2ban|firewall|nginx|alert|telemetria|stato del sistema|stato del server)/i.test(userPrompt);

    // =========================================================================
    // INTENT 1: IMAGE GENERATION (Zero LLM, 100% Focused MediaStudio)
    // =========================================================================
    if (isImageIntent && mediaStudio) {
      const cleanSubject = userPrompt
        .replace(/\\b(?:crea|creami|genera|generami|fai|fammi|disegna|disegnami|illustra|illustrami|produci|mostra|mostrami|voglio|un|uno|una|un'|l'|il|lo|la|delle|dei|degli|immagine|immagini|illustrazione|illustrazioni|foto|fotografia|disegno|disegni|logo|grafica|sfondo|wallpaper)\\b/gi, '')
        .replace(/[🎨🖼️📝📊🛡️]/g, '')
        .replace(/\\s+/g, ' ')
        .trim() || 'opera d\\'arte digitale';

      try {
        const imgDoc = await mediaStudio.generateImage({
          prompt: cleanSubject,
          title: cleanSubject.slice(0, 35),
          userId: username || 'mattia'
        });
        if (imgDoc) {
          const resp = `Ecco l'immagine richiesta: **${cleanSubject}**.\\n\\n![${imgDoc.title}](${imgDoc.url})\\n\\n🎨 **[Visualizza / Scarica Immagine ad Alta Risoluzione](${imgDoc.url})**`;
          return res.json({ success: true, response: resp, reply: resp, file: imgDoc });
        }
      } catch (err) {
        console.warn('[c3po-query] Errore tool Immagine:', err.message);
      }
    }

    // =========================================================================
    // INTENT 2: PDF DOCUMENT GENERATION (Structured Markdown -> Formatted PDF)
    // =========================================================================
    if (isPdfIntent && mediaStudio) {
      const cleanTopic = userPrompt
        .replace(/\\b(?:crea|genera|fai|prepara|esporta|scarica|stampa|redigi|scrivi|un|uno|una|il|lo|la|file|in|pdf|documento|report|promemoria|guida)\\b/gi, '')
        .replace(/\\s+/g, ' ')
        .trim() || 'Guida Tecnica di Sicurezza';

      const docPrompt = `Sei un redattore tecnico di sistemi. Redigi una guida sintetica ad alta densità (150-250 parole) su: \"${cleanTopic}\".
Struttura formale:
# ${cleanTopic.toUpperCase()}
## 1. Sintesi e Obiettivi
## 2. Best Practice e Configurazione (usa elenchi puntati con '*')
## 3. Raccomandazioni di Sicurezza
Italiano autorevole e chirurgico. Inizia subito con il Titolo (#).`;

      try {
        const markdownContent = await aiEngineClient.chatCompletion({
          messages: [
            { role: 'system', content: docPrompt },
            { role: 'user', content: `Redigi la guida tecnica su: ${cleanTopic}` }
          ],
          max_tokens: 320,
          temperature: 0.3
        });

        const pdfDoc = await mediaStudio.createPdf({
          title: cleanTopic.toUpperCase(),
          subtitle: 'HOME-SERVER AI Technical Guide',
          content: markdownContent,
          author: `C-3PO • HOME-SERVER AI Studio`,
          userId: username || 'mattia'
        });

        if (pdfDoc) {
          const resp = `${markdownContent}\\n\\n📄 **[Scarica Documento PDF: ${pdfDoc.title}](${pdfDoc.url})** (${pdfDoc.sizeFormatted})`;
          return res.json({ success: true, response: resp, reply: resp, file: pdfDoc });
        }
      } catch (err) {
        console.warn('[c3po-query] Errore tool PDF:', err.message);
      }
    }

    // =========================================================================
    // INTENT 3: WEB SEARCH & RAG
    // =========================================================================
    if (isWebSearch) {
      const urlMatch = userPrompt.match(/https?:\\/\\/[^\\s]+/i);
      const explicitUrl = urlMatch ? urlMatch[0] : null;

      const cleanQuery = userPrompt
        .replace(/https?:\\/\\/[^\\s]+/gi, '')
        .replace(/\\b(?:cerca|cercami|trova|trovami|cerca su internet|cerca sul web|ultime notizie|novità su|notizie su|informazioni su|dimmi qualcosa su|aggiornamenti su)\\b/gi, '')
        .replace(/\\s+/g, ' ')
        .trim() || userPrompt;

      let webContext = '';
      try {
        const webRes = await webResearchService.searchAndScrape(cleanQuery, { maxPages: 2, explicitUrl });
        webContext = webRes.ragPromptContext;
      } catch (err) {
        webContext = `[AVVISO RICERCA]: Impossibile completare la ricerca: ${err.message}`;
      }

      const webSystemPrompt = `Sei C-3PO, assistente di ricerca intelligente per HOME-SERVER.
Rispondi in modo brillante, chiaro e accurato in italiano alla richiesta dell'utente basandoti ESCLUSIVAMENTE sui dati estratti dalle seguenti fonti web.
Non inventare informazioni non presenti nelle fonti e non parlare del server a meno che non sia il tema della ricerca.
Alla fine della tua risposta elenca sempre le fonti consultate con i rispettivi link:
**Fonti consultate:**
- [Nome Fonte](URL)

${webContext}`;

      const contextMessages = [
        { role: 'system', content: webSystemPrompt },
        { role: 'user', content: userPrompt }
      ];

      const rawContent = await aiEngineClient.chatCompletion({
        messages: contextMessages,
        max_tokens: 220,
        temperature: 0.3
      });

      let answer = rawContent.replace(/<think>[\\s\\S]*?<\\/think>/gi, '').trim() || rawContent.trim();
      return res.json({ success: true, response: answer, reply: answer });
    }

    // =========================================================================
    // INTENT 4: SERVER TELEMETRY & SOC DIAGNOSTICS
    // =========================================================================
    if (isServerQuery) {
      let telemetryContext = '';
      try {
        const audit = await getLiveServerAudit();
        if (audit) {
          telemetryContext = `\\n[STATO ATTUALE HOST]: Salute ${audit.healthScore || 100}/100 | CPU: ${audit.cpuPercent}% | RAM: ${audit.memPercent}% (${audit.memUsedFormatted}/${audit.memTotalFormatted}) | Container attivi: ${audit.containers?.running || 23}/${audit.containers?.total || 23}.`;
        }
      } catch (tErr) {}

      const socSystemPrompt = `Sei C-3PO, Sentinel Security & Diagnostic Agent per HOME-SERVER su 192.168.1.180 (Intel N100 4-core, 12GB RAM).
Fornisci diagnosi precise, comandi Linux reali ed efficaci. Sii chirurgico e conciso.
${telemetryContext}`;

      const contextMessages = [
        { role: 'system', content: socSystemPrompt },
        ...(Array.isArray(messages) && messages.length > 0 ? messages.slice(-4) : []),
        { role: 'user', content: userPrompt }
      ];

      const rawContent = await aiEngineClient.chatCompletion({
        messages: contextMessages,
        max_tokens: 180,
        temperature: 0.3
      });

      let answer = rawContent.replace(/<think>[\\s\\S]*?<\\/think>/gi, '').trim() || rawContent.trim();
      return res.json({ success: true, response: answer, reply: answer });
    }

    // =========================================================================
    // INTENT 5: CHAT (Default - Conversazione Generale, Ragionamento & Coding)
    // =========================================================================
    const chatSystemPrompt = `Sei C-3PO, un assistente AI intelligente, chiaro, preciso e cordiale per HOME-SERVER.
Rispondi in italiano corretto, naturale e pertinente alla richiesta dell'utente, senza inventare dati e senza parlare dello stato del server a meno che non ti venga chiesto esplicitamente.`;

    const contextMessages = [
      { role: 'system', content: chatSystemPrompt },
      ...(Array.isArray(messages) && messages.length > 0 ? messages.slice(-4) : []),
      { role: 'user', content: userPrompt }
    ];

    const rawContent = await aiEngineClient.chatCompletion({
      messages: contextMessages,
      max_tokens: 160,
      temperature: 0.4
    });

    let answer = rawContent.replace(/<think>[\\s\\S]*?<\\/think>/gi, '').trim() || rawContent.trim();
    return res.json({ success: true, response: answer, reply: answer });

  } catch (err) {
    console.error('[ai-soc-app] Errore c3po-query:', err);
    res.status(500).json({ error: 'Errore generazione C-3PO: ' + err.message });
  }
});

"""

updated_code = code[:start_idx] + new_handler + code[end_idx:]

with sftp.open('/home/gabibbo/HOME-SERVER/ai-soc-app/server.js', 'w') as f:
    f.write(updated_code)
print("Updated /home/gabibbo/HOME-SERVER/ai-soc-app/server.js with definitive intent router!")

sftp.close()

# Restart ai-soc-app
stdin, stdout, stderr = ssh.exec_command("docker restart ai-soc-app && sleep 4 && docker ps --filter name=ai-soc-app")
print(stdout.read().decode('utf-8'))

ssh.close()
