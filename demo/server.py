#!/usr/bin/env python3
"""A local demo of a weight-written bulb selector and real LLM tool calling."""
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BINARY = ROOT.parent / 'target/release/route-demo'
MODEL = os.environ.get('OPENROUTER_MODEL', 'google/gemini-2.5-flash')
CERT_BUNDLE = '/etc/ssl/cert.pem'
TLS_CONTEXT = (ssl.create_default_context(cafile=CERT_BUNDLE)
               if ssl.get_default_verify_paths().cafile is None and Path(CERT_BUNDLE).is_file()
               else ssl.create_default_context())
BULBS = ('left', 'middle', 'right')
TOOLS = [{'type': 'function', 'function': {'name': 'turn_on_' + bulb,
          'description': 'Turn on the ' + bulb + ' bulb.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}
         for bulb in BULBS]

def tokenize(text):
    """Deliberately limited, inspectable preprocessing; never claims to be an LLM."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    action = bool(words & {'light', 'illuminate', 'activate', 'turn', 'switch', 'on'})
    # Negation and off requests are unsupported, rather than silently firing tools.
    unsupported = bool(words & {'off', 'not', 'never', 'dont', 'except', 'without'}) or "don't" in text.lower()
    names = [name for name in BULBS if name in words or (name == 'middle' and 'center' in words)]
    if 'all' in words and words & {'bulb', 'bulbs', 'lights', 'light'}:
        names = list(BULBS)
    return names if action and not unsupported else []

def execute_tools(calls):
    """Each run has a fresh virtual panel; tool calls update that panel."""
    panel = dict.fromkeys(BULBS, False)
    for call in calls:
        bulb = call.removeprefix('turn_on_')
        if bulb not in panel: raise ValueError('Unexpected tool')
        panel[bulb] = True
    return [bulb for bulb in BULBS if panel[bulb]]

def run_transformer(text):
    tokens = tokenize(text)
    start_total = time.perf_counter()
    result = subprocess.run([str(BINARY), *(tokens or ['none'])], check=True, capture_output=True, text=True, timeout=15)
    bits, elapsed_ns = result.stdout.strip().split()
    if len(bits) != 3 or any(bit not in '01' for bit in bits): raise ValueError('Invalid transformer output')
    selected = [bulb for bulb, bit in zip(BULBS, bits) if bit == '1']
    calls = ['turn_on_' + name for name in selected]
    return {'tokens': tokens, 'calls': calls,
            'bulbs': execute_tools(calls), 'forward_us': int(elapsed_ns) / 1000,
            'process_ms': (time.perf_counter() - start_total) * 1000}

def run_llm(text):
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        return {'error': 'OPENROUTER_API_KEY is not available through Infisical.'}
    payload = {'model': MODEL, 'messages': [
        {'role': 'system', 'content': 'You control three initially off bulbs. Use the available tools to turn on only the bulbs the user asks to light. You may call multiple tools. Do not call a tool for bulbs the user does not want on. If unclear, make no tool call.'},
        {'role': 'user', 'content': text}], 'tools': TOOLS, 'tool_choice': 'auto',
        'temperature': 0, 'max_tokens': 256, 'usage': {'include': True}}
    request = Request('https://openrouter.ai/api/v1/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=60, context=TLS_CONTEXT) as response:
            data = json.load(response)
        message = data['choices'][0]['message']
        tool_calls = message.get('tool_calls') or []
        allowed = {'turn_on_' + name: name for name in BULBS}
        calls = []
        for call in tool_calls:
            function = call.get('function', {})
            name = function.get('name')
            if name not in allowed: raise ValueError('Unexpected tool')
            args = json.loads(function.get('arguments') or '{}')
            if args != {}: raise ValueError('Unexpected arguments')
            calls.append(name)
        return {'calls': calls, 'bulbs': execute_tools(calls), 'message': message.get('content') or '',
                'elapsed_ms': (time.perf_counter() - start) * 1000,
                'model': data.get('model', MODEL), 'usage': data.get('usage', {})}
    except HTTPError as error:
        return {'error': f'OpenRouter returned HTTP {error.code}.'}
    except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
        return {'error': 'OpenRouter request failed or returned an unexpected response.'}

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path.split('?')[0] == '/api/status':
            return self.reply(200, {'model': MODEL, 'openrouter_configured': bool(os.environ.get('OPENROUTER_API_KEY'))})
        if self.path.split('?')[0] not in ('/', '/index.html', '/style.css', '/app.js'):
            return self.reply(404, {'error': 'Not found'})
        super().do_GET()

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != '/api/compare': return self.reply(404, {'error': 'Not found'})
        if self.headers.get_content_type() != 'application/json':
            return self.reply(415, {'error': 'Content-Type must be application/json.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096: raise ValueError()
            text = json.loads(self.rfile.read(length))['text']
            if not isinstance(text, str) or not text.strip() or len(text) > 500: raise ValueError()
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return self.reply(400, {'error': 'Enter a request of 1–500 characters.'})
        try:
            transformer = run_transformer(text)
        except (OSError, subprocess.SubprocessError, ValueError):
            return self.reply(500, {'error': 'Transformer execution failed. Build the release binary first.'})
        self.reply(200, {'transformer': transformer, 'llm': run_llm(text)})

if __name__ == '__main__':
    print('Serving http://127.0.0.1:8765', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
