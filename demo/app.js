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
    showBulbs('rust-bulbs', a.bulbs); show('tokens', a.tokens.length ? a.tokens.join(', ') : 'No matches'); show('rust-calls', calls(a.calls));
    show('rust-time', a.forward_us.toFixed(1) + ' µs'); show('rust-process', a.process_ms.toFixed(2) + ' ms');
    if (b.error) { showBulbs('llm-bulbs'); show('llm-calls', 'Unavailable'); show('llm-time', '—'); show('llm-cost', '—'); show('llm-message', b.error); show('comparison', 'The transformer ran, but the LLM request failed.'); }
    else {
      showBulbs('llm-bulbs', b.bulbs); show('llm-calls', calls(b.calls)); show('llm-time', (b.elapsed_ms / 1000).toFixed(2) + ' s');
      const usage = b.usage || {}; show('llm-cost', (usage.total_tokens ?? '—') + ' / ' + (typeof usage.cost === 'number' ? '$' + usage.cost.toFixed(6) : 'unavailable'));
      show('llm-message', b.message || 'No text response'); show('model', b.model + ' · tool calling');
      show('comparison', JSON.stringify(a.bulbs) === JSON.stringify(b.bulbs) ? '✳  Both approaches selected the same bulbs.' : '↗  The approaches selected different bulbs. Compare the text matches and tool calls above.');
    }
  } catch (error) { show('error', error.message); show('comparison', 'Could not run the comparison.'); }
  finally { $('#run').disabled = false; $('#run').innerHTML = 'Run both <span>↗</span>'; }
});
