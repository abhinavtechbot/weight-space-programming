const $ = selector => document.querySelector(selector);
const bulbs = ['left', 'middle', 'right'];
function showBulbs(id, active = []) {
  const panel = $('#' + id); panel.replaceChildren();
  bulbs.forEach(name => {
    const item = document.createElement('div'); item.className = 'bulb-item' + (active.includes(name) ? ' lit' : '');
    item.innerHTML = '<div class="bulb-glow"><div class="bulb-shape"><span></span></div><div class="bulb-base"></div></div><div class="bulb-name">' + name.toUpperCase() + '</div><div class="bulb-state">' + (active.includes(name) ? 'ON' : 'OFF') + '</div>';
    panel.append(item);
  });
}
function show(id, text) { $('#' + id).textContent = text; }
function calls(value) { return value.length ? value.map(s => s + '()').join(', ') : 'No calls'; }
showBulbs('rust-bulbs'); showBulbs('llm-bulbs');
fetch('/api/status').then(r => r.json()).then(data => show('model', data.model + (data.openrouter_configured ? ' · tool calling' : ' · key missing'))).catch(() => show('model', 'Server unavailable'));
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => { $('#prompt').value = button.dataset.example; }));
$('#prompt').addEventListener('keydown', event => { if (event.key === 'Enter') $('#run').click(); });
$('#run').addEventListener('click', async () => {
  const text = $('#prompt').value.trim(); if (!text) { show('error', 'Enter an instruction first.'); return; }
  $('#run').disabled = true; $('#run').textContent = 'Running…'; show('error', ''); show('comparison', 'Running both approaches…');
  try {
    const response = await fetch('/api/compare', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error);
    const a = data.transformer, b = data.llm;
    showBulbs('rust-bulbs', a.bulbs); show('tokens', a.tokens.length ? a.tokens.join(' · ') : 'No tokens'); show('rust-calls', calls(a.calls));
    show('rust-time', a.forward_us.toFixed(1) + ' µs'); show('rust-process', a.process_ms.toFixed(2) + ' ms');
    show('baseline', 'Original keyword baseline: ' + (data.baseline.bulbs.join(', ') || 'no bulbs') + ' · ' + data.baseline.elapsed_ms.toFixed(3) + ' ms (in-process)');
    if (b.error) { showBulbs('llm-bulbs'); show('llm-calls', 'Unavailable'); show('llm-time', '—'); show('llm-cost', '—'); show('llm-message', b.error); show('comparison', 'The transformer ran, but the LLM request failed.'); }
    else {
      showBulbs('llm-bulbs', b.bulbs); show('llm-calls', calls(b.calls)); show('llm-time', (b.elapsed_ms / 1000).toFixed(2) + ' s');
      const usage = b.usage || {}; show('llm-cost', (usage.total_tokens ?? '—') + ' / ' + (typeof usage.cost === 'number' ? '$' + usage.cost.toFixed(6) : 'unavailable'));
      show('llm-message', b.message || 'No text response'); show('model', b.model + ' · tool calling');
      show('comparison', JSON.stringify(a.bulbs) === JSON.stringify(b.bulbs) ? '✳  The network and LLM selected the same bulbs. This does not establish correctness.' : '↗  The network and LLM selected different bulbs. Compare their tool calls above.');
    }
  } catch (error) { show('error', error.message); show('comparison', 'Could not run the comparison.'); }
  finally { $('#run').disabled = false; $('#run').innerHTML = 'Run both <span>↗</span>'; }
});
$('#evaluate').addEventListener('click', async () => {
  const cases = [
    ['Please illuminate the leftmost and last bulbs', ['left','right']],
    ['Activate every bulb except the centre one', ['left','right']],
    ['Could you switch on the right lamp?', ['right']],
    ['Turn on the left and middle bulbs', ['left','middle']],
    ['Light all bulbs but the leftmost', ['middle','right']],
    ['Leave the left bulb dark and turn on the right bulb', ['right']],
    ['Switch on all three lights except the middle one', ['left','right']],
    ['Please turn on the first and third bulbs', ['left','right']],
    ['Illuminate only the centre bulb', ['middle']],
    ['Turn on the lamp at the far right', ['right']],
    ['Hello there', []],
    ['Turn on all the bulbs', ['left','middle','right']],
  ];
  const score = {network:0, baseline:0, llm:0, errors:0};
  const match = (a,b) => JSON.stringify(a) === JSON.stringify(b);
  const label = a => a.length ? a.join('+') : 'none';
  $('#eval-results').replaceChildren(); $('#evaluate').disabled = true; $('#run').disabled = true;
  for (let index = 0; index < cases.length; index++) {
    const [text, expected] = cases[index];
    const row = document.createElement('div'); row.className = 'eval-row'; row.textContent = text + ' · running…'; $('#eval-results').append(row);
    try {
      const response = await fetch('/api/compare', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
      const data = await response.json(); if (!response.ok) throw new Error(data.error);
      score.network += Number(match(data.transformer.bulbs,expected));
      score.baseline += Number(match(data.baseline.bulbs,expected));
      if (data.llm.error) score.errors++; else score.llm += Number(match(data.llm.bulbs,expected));
      row.textContent = text + ' → expected: ' + label(expected) + ' | network: ' + label(data.transformer.bulbs) + ' | keywords: ' + label(data.baseline.bulbs) + ' | LLM: ' + (data.llm.error ? 'error' : label(data.llm.bulbs));
    } catch (error) { score.errors++; row.textContent = text + ' · request error: ' + error.message; }
    show('eval-summary', 'Completed ' + (index+1) + '/12 · network ' + score.network + '/12 · keywords ' + score.baseline + '/12 · LLM ' + score.llm + '/12 · request errors ' + score.errors);
  }
  $('#evaluate').disabled = false; $('#run').disabled = false;
});
