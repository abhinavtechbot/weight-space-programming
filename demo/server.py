#!/usr/bin/env python3
"""Local demo server. Run through Infisical; never send its key to the browser."""
import json
import os
from pathlib import Path
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
RULES = ('Route a sequence of event tokens according to these exact rules. '
         'If Bash and Danger are present, return Oracle. Otherwise if Bash is present, '
         'return Bash. Otherwise if Search is present, return Search. Otherwise return NoRoute. '
         'Order and repetition do not matter; Other is inert. '
         'Return exactly one word: Search, Bash, Oracle, or NoRoute. No explanation.')

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path.split('?')[0] == '/api/status':
            return self.reply(200, {'model': MODEL, 'openrouter_configured': bool(os.environ.get('OPENROUTER_API_KEY'))})
        # Serve only the public demo assets, never server source or local files.
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
            if length <= 0 or length > 4096: raise ValueError()
            body = json.loads(self.rfile.read(length))
            events = body['events']
            if not isinstance(events, list) or not 1 <= len(events) <= 63 or any(x not in ('search', 'bash', 'danger', 'other') for x in events): raise ValueError()
            if not isinstance(body.get('llm', False), bool): raise ValueError()
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return self.reply(400, {'error': 'Provide 1–63 valid event tokens.'})
        try:
            out = subprocess.run([str(BINARY), *events], check=True, capture_output=True, text=True, timeout=15)
            route, elapsed_ns = out.stdout.strip().split()
        except (OSError, subprocess.SubprocessError, ValueError):
            return self.reply(500, {'error': 'Transformer execution failed. Build the release binary first.'})
        result = {'transformer': {'route': route, 'elapsed_ns': int(elapsed_ns)}}
        if body.get('llm'):
            key = os.environ.get('OPENROUTER_API_KEY')
            if not key:
                result['llm'] = {'error': 'OPENROUTER_API_KEY is not available through Infisical.'}
            else:
                request_body = json.dumps({'model': MODEL, 'messages': [{'role': 'system', 'content': RULES}, {'role': 'user', 'content': 'Events: ' + ', '.join(events)}], 'temperature': 0, 'max_tokens': 128, 'usage': {'include': True}}).encode()
                req = Request('https://openrouter.ai/api/v1/chat/completions', data=request_body, headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
                start = time.perf_counter()
                try:
                    with urlopen(req, timeout=60, context=TLS_CONTEXT) as response:
                        data = json.load(response)
                    content = data['choices'][0]['message']['content']
                    if not isinstance(content, str): raise ValueError('missing content')
                    result['llm'] = {'route': content.strip(), 'elapsed_ms': (time.perf_counter() - start) * 1000, 'model': data.get('model', MODEL), 'usage': data.get('usage', {})}
                except HTTPError as error:
                    result['llm'] = {'error': f'OpenRouter returned HTTP {error.code}.'}
                except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
                    result['llm'] = {'error': 'OpenRouter request failed or returned an unexpected response.'}
        self.reply(200, result)

if __name__ == '__main__':
    print('Serving http://127.0.0.1:8765', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
