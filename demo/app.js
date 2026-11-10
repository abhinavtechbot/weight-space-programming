const names = { search: 'Search', bash: 'Bash', danger: 'Danger', other: 'Other' };
let sequence = ['bash', 'danger'];
const $ = selector => document.querySelector(selector);
function expected(events) {
  if (!events.length) return null;
  if (events.includes('bash')) return events.includes('danger') ? 'Oracle' : 'Bash';
  return events.includes('search') ? 'Search' : 'NoRoute';
}
function resetResults() {
  for (const id of ['transformer-route','transformer-time','transformer-match','llm-route','llm-time','llm-match','llm-usage']) $('#' + id).textContent = '—';
  $('#error').textContent = '';
}
function render() {
  $('#sequence').replaceChildren();
  if (!sequence.length) { const empty = document.createElement('span'); empty.className = 'empty'; empty.textContent = 'Add an event to get started'; $('#sequence').append(empty); }
  sequence.forEach((token, index) => {
    const button = document.createElement('button'); button.className = 'chip ' + token; button.textContent = names[token] + ' ×';
    button.setAttribute('aria-label', 'Remove ' + names[token] + ' at position ' + (index + 1));
    button.addEventListener('click', () => { sequence.splice(index, 1); resetResults(); render(); }); $('#sequence').append(button);
  });
  $('#expected').textContent = expected(sequence) || 'Add an event';
  $('#run').disabled = !sequence.length;
  document.querySelectorAll('[data-token]').forEach(button => button.disabled = sequence.length >= 63);
}
document.querySelectorAll('[data-token]').forEach(button => button.addEventListener('click', () => { if (sequence.length < 63) sequence.push(button.dataset.token); resetResults(); render(); }));
document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => { sequence = ({ search: ['search'], bash: ['bash'], dangerous: ['bash', 'danger'], other: ['other'] })[button.dataset.preset].slice(); resetResults(); render(); }));
$('#clear').addEventListener('click', () => { sequence = []; resetResults(); render(); });
$('#run').addEventListener('click', async () => {
  const input = sequence.slice(), target = expected(input);
  if (!target) return;
  document.querySelectorAll('button').forEach(b => b.disabled = true);
  resetResults(); $('#run').textContent = 'Running both systems…';
  try {
    const response = await fetch('/api/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({events: input, llm: true}) });
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Request failed');
    $('#transformer-route').textContent = data.transformer.route;
    $('#transformer-time').textContent = (data.transformer.elapsed_ns / 1000).toFixed(1) + ' µs';
    $('#transformer-match').textContent = data.transformer.route === target ? 'Yes ✓' : 'No ✕';
    if (data.llm.error) { $('#llm-route').textContent = 'Unavailable'; $('#error').textContent = data.llm.error; }
    else {
      $('#llm-route').textContent = data.llm.route || '(empty response)';
      $('#llm-time').textContent = (data.llm.elapsed_ms / 1000).toFixed(2) + ' s';
      $('#llm-match').textContent = data.llm.route === target ? 'Yes ✓' : 'No ✕';
      $('#model-name').textContent = data.llm.model;
      const usage = data.llm.usage || {};
      const cost = typeof usage.cost === 'number' ? '$' + usage.cost.toFixed(6) : 'cost unavailable';
      $('#llm-usage').textContent = (usage.total_tokens ?? '—') + ' / ' + cost;
    }
  } catch (error) { $('#error').textContent = 'Could not complete the comparison: ' + error.message; }
  finally { document.querySelectorAll('button').forEach(b => b.disabled = false); $('#run').innerHTML = 'Run comparison <span>↗</span>'; render(); }
});
fetch('/api/status').then(r => r.json()).then(data => { $('#status').textContent = data.openrouter_configured ? 'OPENROUTER READY' : 'OPENROUTER KEY MISSING'; $('#model-name').textContent = data.model; }).catch(() => { $('#status').textContent = 'SERVER UNAVAILABLE'; });
$('#suite-run').addEventListener('click', async () => {
  const cases = [['other'], ['search'], ['bash'], ['danger'], ['search', 'bash'], ['search', 'danger'], ['bash', 'danger'], ['search', 'bash', 'danger']];
  let rustCorrect = 0, llmCorrect = 0, complete = 0;
  $('#suite-results').replaceChildren();
  document.querySelectorAll('button').forEach(b => b.disabled = true);
  for (const events of cases) {
    const row = document.createElement('div'); row.className = 'suite-row';
    row.textContent = events.map(event => names[event]).join(' + ') + ' · running…'; $('#suite-results').append(row);
    try {
      const response = await fetch('/api/compare', {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({events, llm: true})});
      const data = await response.json(); if (!response.ok) throw new Error(data.error);
      const target = expected(events), rust = data.transformer.route, llm = data.llm.route || 'error';
      rustCorrect += Number(rust === target); llmCorrect += Number(llm === target);
      row.textContent = events.map(event => names[event]).join(' + ') + ' → expected ' + target + ' · Rust ' + rust + ' · LLM ' + llm + (data.llm.error ? ' (' + data.llm.error + ')' : '');
    } catch (error) { row.textContent = events.join(' + ') + ' · request failed: ' + error.message; }
    complete++;
    $('#suite-summary').textContent = 'Completed ' + complete + '/8 · Rust ' + rustCorrect + '/8 · LLM ' + llmCorrect + '/8';
  }
  document.querySelectorAll('button').forEach(b => b.disabled = false); render();
});
render();
