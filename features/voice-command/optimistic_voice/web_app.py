"""Optimistic Smart Glasses - Live Web HUD & Testing Dashboard."""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimistic_voice.assistant.command_router import CommandRouter
from optimistic_voice.assistant.intent_classifier import IntentType, ParsedCommand
from optimistic_voice.audio.microphone import MicrophoneStream
from optimistic_voice.audio.speech_recognition import SpeechRecognizer
from optimistic_voice.audio.text_to_speech import TextToSpeechEngine
from optimistic_voice.audio.voice_activity import VoiceActivityDetector
from optimistic_voice.config import OptimisticConfig
from optimistic_voice.display.display_manager import DisplayManager
from optimistic_voice.main import OptimisticAssistant

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Optimistic Smart Glasses — Live Voice Dashboard</title>
  <style>
    :root {
      --bg: #07090e;
      --card-bg: rgba(18, 24, 38, 0.7);
      --card-border: rgba(0, 229, 255, 0.18);
      --cyan: #00e5ff;
      --cyan-glow: rgba(0, 229, 255, 0.35);
      --green: #10b981;
      --amber: #f59e0b;
      --red: #ef4444;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at 50% 0%, #0d1527 0%, var(--bg) 75%);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 32px 16px;
    }
    .container {
      width: 100%;
      max-width: 900px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    header {
      text-align: center;
      margin-bottom: 8px;
    }
    h1 {
      font-size: 26px;
      letter-spacing: 2px;
      color: var(--cyan);
      text-shadow: 0 0 16px var(--cyan-glow);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 6px;
    }
    .hud-card {
      background: #020408;
      border: 1px solid var(--cyan);
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 0 24px rgba(0, 229, 255, 0.15);
      position: relative;
      overflow: hidden;
    }
    .hud-card::before {
      content: "OPTIMISTIC HUD [LOCAL GLASSES DISPLAY]";
      position: absolute;
      top: 8px;
      left: 14px;
      font-size: 10px;
      letter-spacing: 1.5px;
      color: var(--cyan);
      opacity: 0.8;
    }
    .hud-status {
      font-size: 18px;
      color: #7dd3fc;
      margin-top: 14px;
      font-weight: 600;
    }
    .hud-msg {
      font-size: 24px;
      color: #ffffff;
      margin-top: 8px;
      min-height: 36px;
      font-weight: bold;
    }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      backdrop-filter: blur(12px);
    }
    .panel-title {
      font-size: 14px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--cyan);
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .info-box {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 12px;
    }
    .info-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }
    .info-val { font-size: 14px; font-weight: 600; margin-top: 4px; color: #fff; }
    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
    }
    .badge-live { background: rgba(16, 185, 129, 0.2); color: var(--green); border: 1px solid var(--green); }
    .badge-idle { background: rgba(148, 163, 184, 0.2); color: var(--text-muted); }
    .badge-speech { background: rgba(0, 229, 255, 0.2); color: var(--cyan); border: 1px solid var(--cyan); }
    .meter-container {
      width: 100%;
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      overflow: hidden;
      margin-top: 8px;
    }
    .meter-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--cyan), var(--green));
      transition: width 0.08s ease;
    }
    .controls-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    button {
      background: linear-gradient(135deg, #0ea5e9, #0284c7);
      color: #fff;
      border: none;
      padding: 10px 18px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    button:hover {
      background: linear-gradient(135deg, #38bdf8, #0ea5e9);
      box-shadow: 0 0 12px rgba(14, 165, 233, 0.5);
    }
    button.btn-stop {
      background: linear-gradient(135deg, #ef4444, #dc2626);
    }
    button.btn-stop:hover {
      background: linear-gradient(135deg, #f87171, #ef4444);
      box-shadow: 0 0 12px rgba(239, 68, 68, 0.5);
    }
    .input-row {
      display: flex;
      gap: 10px;
      margin-top: 14px;
    }
    input[type="text"] {
      flex: 1;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      padding: 12px 14px;
      color: #fff;
      font-size: 14px;
    }
    input[type="text"]:focus {
      outline: none;
      border-color: var(--cyan);
      box-shadow: 0 0 8px var(--cyan-glow);
    }
    .chips-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .chip {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 20px;
      padding: 6px 12px;
      font-size: 12px;
      color: #cbd5e1;
      cursor: pointer;
      transition: all 0.2s;
    }
    .chip:hover {
      background: rgba(0, 229, 255, 0.15);
      border-color: var(--cyan);
      color: #fff;
    }
    .log-box {
      background: #020408;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 14px;
      font-family: Consolas, monospace;
      font-size: 12px;
      height: 180px;
      overflow-y: auto;
      color: #a5f3fc;
      line-height: 1.6;
    }
    .log-entry { margin-bottom: 4px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1><span>👓</span> OPTIMISTIC SMART GLASSES</h1>
      <p class="subtitle">Live Voice Assistant & Natural Language Inspection Dashboard</p>
    </header>

    <!-- Virtual Smart Glasses Screen -->
    <div class="hud-card">
      <div id="hud-status" class="hud-status">Initializing...</div>
      <div id="hud-msg" class="hud-msg">Welcome back Optimistic innovators</div>
    </div>

    <!-- Live Diagnostics Panel -->
    <div class="panel">
      <div class="panel-title">
        <span>Hardware & Recognition Diagnostics</span>
        <span id="speech-badge" class="badge badge-idle">Microphone Idle</span>
      </div>
      <div class="info-grid">
        <div class="info-box">
          <div class="info-label">Input Microphone</div>
          <div id="info-mic" class="info-val">Loading...</div>
        </div>
        <div class="info-box">
          <div class="info-label">Active STT Engine</div>
          <div id="info-stt" class="info-val">Loading...</div>
        </div>
        <div class="info-box">
          <div class="info-label">Last Intent / Confidence</div>
          <div id="info-intent" class="info-val">—</div>
        </div>
        <div class="info-box">
          <div class="info-label">Audio Level (RMS)</div>
          <div id="info-rms" class="info-val">0.000</div>
          <div class="meter-container">
            <div id="meter-bar" class="meter-bar"></div>
          </div>
        </div>
      </div>

      <div class="controls-row">
        <button id="btn-mic-toggle" onclick="toggleMicrophone()">🎙️ Start Continuous Voice Listening</button>
        <button id="btn-browser-mic" onclick="toggleBrowserMic()">🌐 Use Browser Web-Mic</button>
        <button onclick="clearLogs()" style="background: rgba(255,255,255,0.1); color: #fff;">Clear Logs</button>
      </div>
    </div>

    <!-- Interactive Command Tester -->
    <div class="panel">
      <div class="panel-title">Try Natural Voice Commands</div>
      <p style="font-size: 13px; color: var(--text-muted);">
        Click any example natural sentence below to test the intent interpretation layer immediately:
      </p>
      <div class="chips-row">
        <div class="chip" onclick="sendQuickCommand('Can you open Google for me?')">"Can you open Google for me?"</div>
        <div class="chip" onclick="sendQuickCommand('Launch my browser and search Google')">"Launch my browser and search Google"</div>
        <div class="chip" onclick="sendQuickCommand('I want to take some notes in notepad')">"I want to take some notes in notepad"</div>
        <div class="chip" onclick="sendQuickCommand('Can you open the calculator please')">"Can you open the calculator please"</div>
        <div class="chip" onclick="sendQuickCommand('Search YouTube for chill lofi beats')">"Search YouTube for chill lofi beats"</div>
        <div class="chip" onclick="sendQuickCommand('Select the first video')">"Select the first video"</div>
        <div class="chip" onclick="sendQuickCommand('Pause the video')">"Pause the video"</div>
        <div class="chip" onclick="sendQuickCommand('Can you type hello world for me')">"Can you type hello world for me"</div>
        <div class="chip" onclick="sendQuickCommand('Please minimize this window')">"Please minimize this window"</div>
      </div>

      <div class="input-row">
        <input type="text" id="cmd-input" placeholder="Or type any custom voice command here (e.g. 'Go to Google')..." onkeydown="if(event.key==='Enter') executeTextCommand()">
        <button onclick="executeTextCommand()">Send Command</button>
      </div>
    </div>

    <!-- Live Event Feed -->
    <div class="panel">
      <div class="panel-title">Real-Time Event Stream</div>
      <div id="log-box" class="log-box">
        <div class="log-entry">[OPTIMISTIC] Connected to local voice companion on localhost.</div>
      </div>
    </div>
  </div>

  <script>
    let isListening = false;
    let browserRecognition = null;

    function addLog(msg) {
      const box = document.getElementById('log-box');
      const div = document.createElement('div');
      div.className = 'log-entry';
      div.innerText = msg;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    function clearLogs() {
      document.getElementById('log-box').innerHTML = '';
    }

    // Connect to Server-Sent Events (SSE) stream from backend
    const evtSource = new EventSource('/api/events');
    evtSource.onmessage = function(e) {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'status') {
          document.getElementById('info-mic').innerText = data.mic || 'None';
          document.getElementById('info-stt').innerText = data.engine || 'None';
          if (data.listening !== undefined) {
            isListening = data.listening;
            updateMicButton();
          }
        } else if (data.type === 'speech_detected') {
          const badge = document.getElementById('speech-badge');
          badge.className = 'badge badge-speech';
          badge.innerText = 'Speech Detected...';
          addLog('[OPTIMISTIC] Speech detected...');
        } else if (data.type === 'live_partial') {
          document.getElementById('hud-status').innerText = 'Hearing...';
          document.getElementById('hud-msg').innerText = data.text;
          addLog('[OPTIMISTIC] Live: "' + data.text + '"');
        } else if (data.type === 'heard') {
          document.getElementById('hud-status').innerText = 'Heard:';
          document.getElementById('hud-msg').innerText = data.text;
          const badge = document.getElementById('speech-badge');
          badge.className = 'badge badge-idle';
          badge.innerText = 'Microphone Idle';
          addLog('[OPTIMISTIC] Heard: "' + data.text + '"');
        } else if (data.type === 'intent') {
          document.getElementById('info-intent').innerText = data.intent + ' (' + Math.round(data.confidence * 100) + '%)';
          addLog('[OPTIMISTIC] Intent: ' + data.intent + ' (Confidence: ' + (data.confidence * 100).toFixed(0) + '%)');
        } else if (data.type === 'executed') {
          document.getElementById('hud-status').innerText = 'Done';
          document.getElementById('hud-msg').innerText = data.message;
          addLog('[OPTIMISTIC] Executing: ' + data.message);
        } else if (data.type === 'level') {
          document.getElementById('info-rms').innerText = data.rms.toFixed(4);
          const pct = Math.min(100, Math.max(0, data.rms * 500));
          document.getElementById('meter-bar').style.width = pct + '%';
        }
      } catch (err) {
        console.error(err);
      }
    };

    function updateMicButton() {
      const btn = document.getElementById('btn-mic-toggle');
      if (isListening) {
        btn.innerText = '⏹️ Stop Voice Listening';
        btn.className = 'btn-stop';
        document.getElementById('speech-badge').className = 'badge badge-live';
        document.getElementById('speech-badge').innerText = 'Listening Active';
      } else {
        btn.innerText = '🎙️ Start Continuous Voice Listening';
        btn.className = '';
        document.getElementById('speech-badge').className = 'badge badge-idle';
        document.getElementById('speech-badge').innerText = 'Microphone Idle';
      }
    }

    async function toggleMicrophone() {
      const action = isListening ? 'stop' : 'start';
      await fetch('/api/voice/' + action, { method: 'POST' });
    }

    async function executeTextCommand() {
      const input = document.getElementById('cmd-input');
      const val = input.value.trim();
      if (!val) return;
      input.value = '';
      await sendQuickCommand(val);
    }

    async function sendQuickCommand(text) {
      addLog('User > ' + text);
      const res = await fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
      });
      const data = await res.json();
      document.getElementById('info-intent').innerText = data.intent + ' (' + Math.round(data.confidence * 100) + '%)';
      document.getElementById('hud-status').innerText = data.success ? 'Executing...' : 'Clarification needed';
      document.getElementById('hud-msg').innerText = data.message;
      addLog('[OPTIMISTIC] Result: ' + data.message);
    }

    function toggleBrowserMic() {
      if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert('Web Speech API is not supported in this browser. Use Chrome or Edge.');
        return;
      }
      if (browserRecognition) {
        browserRecognition.stop();
        browserRecognition = null;
        document.getElementById('btn-browser-mic').innerText = '🌐 Use Browser Web-Mic';
        return;
      }
      const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
      browserRecognition = new SpeechRec();
      browserRecognition.continuous = false;
      browserRecognition.interimResults = true;
      browserRecognition.lang = 'en-US';

      browserRecognition.onstart = () => {
        document.getElementById('btn-browser-mic').innerText = '⏹️ Stop Web-Mic (Listening...)';
        document.getElementById('hud-status').innerText = 'Listening via browser...';
      };
      browserRecognition.onresult = (evt) => {
        let transcript = '';
        for (let i = evt.resultIndex; i < evt.results.length; ++i) {
          transcript += evt.results[i][0].transcript;
        }
        document.getElementById('hud-msg').innerText = transcript;
        if (evt.results[0].isFinal) {
          sendQuickCommand(transcript);
        }
      };
      browserRecognition.onend = () => {
        document.getElementById('btn-browser-mic').innerText = '🌐 Use Browser Web-Mic';
        browserRecognition = null;
      };
      browserRecognition.start();
    }
  </script>
</body>
</html>
"""


class VoiceDashboardState:
    """Thread-safe event broadcast and state manager for Web HUD."""

    def __init__(self, assistant: OptimisticAssistant) -> None:
        self.assistant = assistant
        self.event_queues: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.voice_thread: threading.Thread | None = None
        self.is_listening = False

    def add_client(self) -> queue.Queue:
        q = queue.Queue(maxsize=100)
        with self._lock:
            self.event_queues.append(q)
        # Send initial status
        mic_name = self.assistant.mic.device_name if self.assistant.mic else "USB / System Default"
        stt_name = self.assistant.stt.active_engine if self.assistant.stt else "auto"
        q.put({"type": "status", "mic": mic_name, "engine": stt_name, "listening": self.is_listening})
        return q

    def remove_client(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self.event_queues:
                self.event_queues.remove(q)

    def broadcast(self, event: dict[str, Any]) -> None:
        with self._lock:
            for q in list(self.event_queues):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def start_voice_loop(self) -> None:
        if self.is_listening:
            return
        self.is_listening = True
        self.broadcast({"type": "status", "listening": True})

        def _loop() -> None:
            if self.assistant.stt is None:
                self.assistant.stt = SpeechRecognizer(
                    engine=self.assistant.config.stt_engine,
                    sample_rate=self.assistant.config.sample_rate,
                    vosk_model_path=self.assistant.config.vosk_model_path,
                )
            if self.assistant.mic is None:
                self.assistant.mic = MicrophoneStream(
                    sample_rate=self.assistant.config.sample_rate,
                    channels=self.assistant.config.channels,
                    chunk_samples=self.assistant.config.chunk_samples,
                    device=self.assistant.config.input_device,
                )
            self.assistant.mic.start()
            self.assistant._running = True

            last_partial = ""
            try:
                while self.is_listening and self.assistant._running:
                    chunk = self.assistant.mic.read_chunk(timeout=0.1)
                    if chunk is None:
                        continue
                    utterance = self.assistant.vad.process_chunk(chunk)
                    rms = self.assistant.vad.current_rms
                    self.broadcast({"type": "level", "rms": rms})

                    if self.assistant.vad.speech_started:
                        self.broadcast({"type": "speech_detected"})
                        self.assistant.stt.start_stream()
                        for pre in self.assistant.vad._utterance_chunks:
                            self.assistant.stt.feed_stream_chunk(pre)

                    if self.assistant.vad.is_speaking and not self.assistant.vad.speech_started:
                        partial = self.assistant.stt.feed_stream_chunk(chunk)
                        if partial and partial != last_partial:
                            self.broadcast({"type": "live_partial", "text": partial})
                            last_partial = partial

                    if utterance is not None:
                        final_text = self.assistant.stt.finish_stream(utterance)
                        last_partial = ""
                        if final_text:
                            self.broadcast({"type": "heard", "text": final_text})
                            res = self.assistant.process_utterance(final_text)
                            if res:
                                self.broadcast({"type": "executed", "message": res.message, "success": res.success})
            except Exception as e:
                print(f"[Web Dashboard Error] {e}")
            finally:
                self.is_listening = False
                self.broadcast({"type": "status", "listening": False})

        self.voice_thread = threading.Thread(target=_loop, name="WebVoiceLoop", daemon=True)
        self.voice_thread.start()

    def stop_voice_loop(self) -> None:
        self.is_listening = False
        if self.assistant.mic is not None:
            self.assistant.mic.stop()
        self.broadcast({"type": "status", "listening": False})


class DashboardHandler(BaseHTTPRequestHandler):
    state: VoiceDashboardState

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = self.state.add_client()
            try:
                while True:
                    event = q.get(timeout=20.0)
                    msg = f"data: {json.dumps(event)}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
            except (queue.Empty, ConnectionResetError, BrokenPipeError):
                pass
            finally:
                self.state.remove_client(q)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else ""

        if parsed.path == "/api/command":
            data = json.loads(body or "{}")
            text = data.get("text", "")
            cmd = self.state.assistant.parser.parse(text)
            self.state.broadcast({"type": "intent", "intent": cmd.intent.name, "confidence": cmd.confidence})
            result = self.state.assistant.process_utterance(text)
            res_dict = {
                "success": result.success if result else False,
                "message": result.message if result else "No action",
                "intent": cmd.intent.name,
                "confidence": cmd.confidence,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res_dict).encode("utf-8"))
        elif parsed.path == "/api/voice/start":
            self.state.start_voice_loop()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "started"}')
        elif parsed.path == "/api/voice/stop":
            self.state.stop_voice_loop()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "stopped"}')
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy SSE logging
        pass


def run_web_dashboard(
    host: str = "127.0.0.1",
    port: int = 8000,
    config: OptimisticConfig | None = None,
    open_browser: bool = True,
) -> None:
    cfg = config or OptimisticConfig()
    cfg.display_mode = "console"
    assistant = OptimisticAssistant(config=cfg)

    state = VoiceDashboardState(assistant)
    DashboardHandler.state = state

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    url = f"http://{host}:{port}"
    print(f"\n=======================================================")
    print(f"  👓 OPTIMISTIC VOICE WEB DASHBOARD ACTIVE!")
    print(f"  👉 URL: {url}")
    print(f"  (Press Ctrl+C to stop the dashboard server)")
    print(f"=======================================================\n", flush=True)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_voice_loop()
        server.server_close()
        assistant.shutdown()


if __name__ == "__main__":
    run_web_dashboard()
