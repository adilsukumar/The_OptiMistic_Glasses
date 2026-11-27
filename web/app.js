const apiStatus = document.getElementById("api-status");
const camStatus = document.getElementById("cam-status");
const startCamBtn = document.getElementById("start-cam-btn");
const stopCamBtn = document.getElementById("stop-cam-btn");
const resetBufBtn = document.getElementById("reset-buf-btn");
const cameraStream = document.getElementById("camera-stream");
const cameraPlaceholder = document.getElementById("camera-placeholder");
const handsIndicator = document.getElementById("hands-indicator");
const fpsIndicator = document.getElementById("fps-indicator");
const latencyIndicator = document.getElementById("latency-indicator");
const bufferIndicator = document.getElementById("buffer-indicator");
const thresholdSlider = document.getElementById("threshold-slider");
const thresholdVal = document.getElementById("threshold-val");
const autoSpeak = document.getElementById("auto-speak");

const hudClock = document.getElementById("hud-clock");
const hudMode = document.getElementById("hud-mode");
const hudSymbol = document.getElementById("hud-symbol");
const hudPrimary = document.getElementById("hud-primary");
const hudSecondary = document.getElementById("hud-secondary");
const hudConfidence = document.getElementById("hud-confidence");
const hudCategory = document.getElementById("hud-category");
const top3Bars = document.getElementById("top3-bars");

const conversationSentenceEl = document.getElementById("conversation-sentence");
const speakSentenceBtn = document.getElementById("speak-sentence-btn");
const undoWordBtn = document.getElementById("undo-word-btn");
const clearTranscriptBtn = document.getElementById("clear-transcript-btn");

const conversationFlowsEl = document.getElementById("conversation-flows");
const gestureGrid = document.getElementById("gesture-grid");
const metricsSummary = document.getElementById("isl-metrics-summary");

const ISL_EXACT_GUIDE = {
  "NAMASTE": {
    symbol: "🙏",
    hands: "2 Hands (or 1 hand at chest), open palms pressed flat together.",
    start: "In front of chest center.",
    motion: "Hold flat palms pressed together in prayer position.",
    avoid: "Do not curl fingers or separate palms."
  },
  "HOW ARE YOU": {
    symbol: "🤲",
    hands: "2 Hands, open palms facing upward / forward (or 1 hand cupped forward).",
    start: "Hold hands in front of chest.",
    motion: "Hold open palms steady or move slightly forward in polite question.",
    avoid: "Keep hands visible inside the camera frame."
  },
  "I AM FINE": {
    symbol: "👌",
    hands: "1 Hand, OK sign (Thumb & Index tips touch, Middle finger UP).",
    start: "At chest or shoulder height.",
    motion: "Hold OK circle upright facing camera.",
    avoid: "Do not fold middle finger."
  },
  "WHAT ARE YOU DOING": {
    symbol: "❓",
    hands: "1 Hand ('W' sign: Index, Middle, Ring UP) OR 2 Index fingers pointing UP.",
    start: "At chest level.",
    motion: "Hold upright facing camera.",
    avoid: "Do not extend pinky for 'W'."
  },
  "I AM A BOY": {
    symbol: "👦",
    hands: "1 Hand, cap visor gesture at forehead / temple.",
    start: "Hand raised to temple / forehead height.",
    motion: "Touch or hold index and thumb near temple (boy cap brim sign).",
    avoid: "Do not lower hand below chin."
  },
  "I AM A GIRL": {
    symbol: "👧",
    hands: "1 Hand, index finger pointing along cheek or jawline.",
    start: "Hand beside cheek near earlobe.",
    motion: "Touch or trace index finger along cheek (earring / bindi sign).",
    avoid: "Do not block your face."
  },
  "HELP ME": {
    symbol: "🆘",
    hands: "2 Hands (fist on flat palm) OR 1 fist held against chest.",
    start: "At chest level.",
    motion: "Place fist/thumbs-up on flat open palm, or hold fist against chest.",
    avoid: "Ensure fist is clearly resting on palm or chest."
  },
  "I LOVE YOU": {
    symbol: "🤟",
    hands: "1 Hand, Thumb OUT, Index UP, Pinky UP (Middle & Ring curled).",
    start: "Chest / shoulder height.",
    motion: "Hold steady facing camera.",
    avoid: "Keep middle and ring fingers curled into palm."
  },
  "HELLO": {
    symbol: "👋",
    hands: "1 Hand, open flat palm facing camera beside head.",
    start: "Raise hand beside head / ear level.",
    motion: "Wave palm side-to-side gently.",
    avoid: "Do not drop hand below shoulder."
  },
  "YES": {
    symbol: "✊",
    hands: "1 Hand, closed fist held upright in mid-air.",
    start: "Hold fist at chest / mid level.",
    motion: "Nod fist slightly or hold firm.",
    avoid: "Do not extend thumb upward (which means Alright)."
  },
  "NO": {
    symbol: "🤏",
    hands: "1 Hand, authentic ISL beak pinch (Thumb, Index & Middle tips touch).",
    start: "Chest height.",
    motion: "Pinch tips together like closing a beak.",
    avoid: "Keep other fingers back."
  },
  "ALRIGHT": {
    symbol: "👍",
    hands: "1 Hand, closed fist with thumb pointing straight up.",
    start: "Hold hand at chest / shoulder height.",
    motion: "Hold thumb-up steady facing camera.",
    avoid: "Do not open other 4 fingers."
  },
  "ONE": {
    symbol: "1️⃣",
    hands: "1 Hand, Index finger pointing UP only.",
    start: "Chest height.",
    motion: "Hold index finger upright.",
    avoid: "Keep other 3 fingers and thumb folded."
  },
  "TWO": {
    symbol: "2️⃣",
    hands: "1 Hand, Index and Middle fingers UP in 'V' shape.",
    start: "Chest height.",
    motion: "Hold 'V' shape upright.",
    avoid: "Keep ring and pinky fingers folded."
  },
  "THREE": {
    symbol: "3️⃣",
    hands: "1 Hand, Thumb OUT, Index UP, Middle UP.",
    start: "Chest height.",
    motion: "Hold 3 fingers extended.",
    avoid: "Keep ring and pinky fingers folded."
  },
  "FOUR": {
    symbol: "4️⃣",
    hands: "1 Hand, 4 fingers UP (Index, Middle, Ring, Pinky), Thumb in.",
    start: "Chest height.",
    motion: "Hold 4 fingers upright with thumb tucked.",
    avoid: "Do not extend thumb."
  },
  "FIVE": {
    symbol: "5️⃣",
    hands: "1 Hand, all 5 fingers extended and spread open at chest level.",
    start: "At chest level.",
    motion: "Hold 5 open fingers facing camera.",
    avoid: "Do not raise to ear level (which means Hello)."
  },
  "SIX": {
    symbol: "6️⃣",
    hands: "1 Hand, Pinky finger pointing UP only.",
    start: "Chest height.",
    motion: "Hold pinky upright clearly.",
    avoid: "Keep index, middle, and ring fingers curled."
  },
  "SEVEN": {
    symbol: "7️⃣",
    hands: "1 Hand, Thumb OUT and Index UP forming an 'L' shape.",
    start: "Chest height.",
    motion: "Hold 'L' shape upright.",
    avoid: "Do not raise middle finger."
  },
  "TEN": {
    symbol: "🔟",
    hands: "2 Hands, both hands showing 5 + 5 spread fingers.",
    start: "Both hands at chest level.",
    motion: "Hold both open palms side-by-side (5+5=10).",
    avoid: "Keep both hands inside frame."
  },
  "GOOD MORNING": {
    symbol: "🌅",
    hands: "2 Hands (1 at chin, then both arms).",
    start: "Touch fingertips to chin and move forward ('Good').",
    motion: "Place left hand under right elbow and raise right forearm upward like the rising sun.",
    avoid: "Do not skip the rising forearm motion."
  },
  "GOOD AFTERNOON": {
    symbol: "☀️",
    hands: "2 Hands.",
    start: "Touch chin and move hand forward ('Good').",
    motion: "Hold left hand flat under right elbow with right forearm angled at mid-height.",
    avoid: "Keep forearm steady at mid-height."
  },
  "GOOD EVENING": {
    symbol: "🌇",
    hands: "2 Hands.",
    start: "Touch chin and move hand forward ('Good').",
    motion: "Lower right forearm smoothly downward toward horizontal left hand (setting sun).",
    avoid: "Ensure downward arc is clearly visible."
  },
  "GOOD NIGHT": {
    symbol: "🌙",
    hands: "2 Hands.",
    start: "Touch chin and move hand forward ('Good').",
    motion: "Bring right open hand downward over left wrist/hand.",
    avoid: "Do not drop hands below the camera frame."
  },
  "THANK YOU": {
    symbol: "🙏",
    hands: "1 Hand, flat open palm, fingers together.",
    start: "Touch fingertips lightly to chin/lips.",
    motion: "Move open hand smoothly forward toward camera.",
    avoid: "Do not wave; use one clean forward motion."
  },
  "PLEASED": {
    symbol: "😊",
    hands: "1 Hand, flat open palm.",
    start: "Place open palm flat against center of chest.",
    motion: "Rub in a smooth circle on your chest 2 times.",
    avoid: "Keep palm open and facing chest."
  },
  "SUMMER": {
    symbol: "🌞",
    hands: "1 Hand, index finger extended.",
    start: "Place hand near your forehead.",
    motion: "Swipe across forehead left-to-right (wiping sweat).",
    avoid: "Keep hand below the top edge of frame."
  },
  "SPRING": {
    symbol: "🌸",
    hands: "2 Hands.",
    start: "Cup left hand into an 'O'; tuck right fingertips inside.",
    motion: "Push right fingers upward and spread them open like a blooming flower.",
    avoid: "Make the finger-opening motion clear."
  },
  "WINTER": {
    symbol: "❄️",
    hands: "2 Hands, closed fists.",
    start: "Hold both fists in front of chest.",
    motion: "Shiver both fists slightly inward toward chest.",
    avoid: "Keep fists slightly apart so both hands are tracked."
  },
  "FALL": {
    symbol: "🍂",
    hands: "2 Hands.",
    start: "Hold left forearm horizontal; right open hand above it.",
    motion: "Brush right open hand downward past left forearm.",
    avoid: "Keep both hands in frame during the drop."
  },
  "SEASON": {
    symbol: "🔄",
    hands: "2 Hands.",
    start: "Hold left hand in a loose fist; right hand beside it.",
    motion: "Circle right hand around the left fist.",
    avoid: "Keep circular path large and clear."
  },
  "MONSOON": {
    symbol: "🌧️",
    hands: "2 Hands, all 5 fingers spread.",
    start: "Raise both hands to shoulder height, palms forward/down.",
    motion: "Lower both hands while wiggling fingers like falling rain.",
    avoid: "Keep fingers spread while moving downward."
  }
};

const CONVERSATION_FLOWS = [
  {
    title: "1. Morning Greeting Conversation",
    signs: ["HELLO", "GOOD MORNING", "HOW ARE YOU"],
    meaning: "\"Hello! Good morning, how are you?\""
  },
  {
    title: "2. Polite Reply Conversation",
    signs: ["ALRIGHT", "THANK YOU", "PLEASED"],
    meaning: "\"I am alright, thank you! Pleased to meet you.\""
  },
  {
    title: "3. Weather & Season Small Talk",
    signs: ["HELLO", "MONSOON", "SEASON"],
    meaning: "\"Hello! It is monsoon season.\""
  },
  {
    title: "4. Evening / Farewell Conversation",
    signs: ["GOOD EVENING", "THANK YOU", "GOOD NIGHT"],
    meaning: "\"Good evening, thank you, good night!\""
  }
];

let pollTimer = null;
let lastSpokenGesture = "";
let lastSpokenAt = 0;
let labelCategoryMap = {};
let sentenceWords = [];

function updateClock() {
  hudClock.textContent = new Date().toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

function updateHud({ mode, symbol, primary, secondary, confidence, category }) {
  hudMode.textContent = mode || "ACTIVE";
  hudSymbol.textContent = symbol || "🖐️";
  hudPrimary.textContent = primary || "";
  hudSecondary.textContent = secondary || "";
  hudConfidence.textContent = `Confidence: ${confidence ?? "--"}`;
  hudCategory.textContent = `Category: ${category ?? "--"}`;
}

let _browserUtterance = null;

function speakText(text) {
  if (!text) return;

  // 1. Native Windows speaker output through server
  fetch("/api/camera/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: text }),
  }).catch(() => {});

  // 2. Browser Web Speech API fallback
  if ("speechSynthesis" in window) {
    try {
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1.0;
      _browserUtterance = utter;
      window.speechSynthesis.speak(utter);
    } catch (e) {}
  }
}

function formatConversationSentence(words) {
  if (!words.length) return "";
  const formatted = words
    .map((w) => {
      const lower = w.toLowerCase();
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(", ");
  return formatted.replace(/How are you(,|$)/i, "How are you?$1");
}

function renderSentence() {
  if (!sentenceWords.length) {
    conversationSentenceEl.innerHTML =
      '<span class="muted">Your signed conversation will appear here as a full sentence...</span>';
    return;
  }
  const symbols = sentenceWords
    .map((w) => ISL_EXACT_GUIDE[w]?.symbol || "🖐️")
    .join(" ");
  conversationSentenceEl.textContent = `${symbols}  ${formatConversationSentence(sentenceWords)}`;
}

function addWordToConversation(gesture, english) {
  const displayWord = english || gesture;
  if (!sentenceWords.length || sentenceWords[sentenceWords.length - 1] !== displayWord) {
    sentenceWords.push(displayWord);
    renderSentence();
  }

  // Non-repeating translation: speak ONCE when gesture is shown, NEVER repeat on interval
  if (autoSpeak.checked) {
    if (gesture !== lastSpokenGesture) {
      lastSpokenGesture = gesture;
      lastSpokenAt = Date.now();
      speakText(displayWord);
    }
  }
}

speakSentenceBtn.addEventListener("click", () => {
  const text = formatConversationSentence(sentenceWords);
  if (text) speakText(text);
});

undoWordBtn.addEventListener("click", () => {
  sentenceWords.pop();
  renderSentence();
});

clearTranscriptBtn.addEventListener("click", () => {
  sentenceWords = [];
  renderSentence();
});

function renderTop3(topPreds) {
  if (!topPreds || topPreds.length === 0) {
    top3Bars.innerHTML = '<span class="muted">Waiting for 30 frames with hands visible...</span>';
    return;
  }
  top3Bars.innerHTML = topPreds
    .map((p) => {
      const pct = Math.round((p.confidence || 0) * 100);
      const sym = ISL_EXACT_GUIDE[p.label]?.symbol || "🖐️";
      return `
        <div class="bar-row">
          <span class="bar-label">${sym} ${p.label}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
          <span class="bar-pct">${pct}%</span>
        </div>
      `;
    })
    .join("");
}

thresholdSlider.addEventListener("input", async () => {
  const pct = Number(thresholdSlider.value);
  thresholdVal.textContent = `${pct}%`;
  await fetch("/api/camera/threshold", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ threshold: pct / 100 }),
  });
});

resetBufBtn.addEventListener("click", async () => {
  await fetch("/api/camera/reset", { method: "POST" });
});

async function pollCameraPrediction() {
  try {
    const res = await fetch("/api/camera/prediction");
    const data = await res.json();
    if (!data.running) {
      setCameraUI(false);
      if (data.error) {
        updateHud({
          mode: "CAMERA ERROR",
          symbol: "⚠️",
          primary: "Camera Unavailable",
          secondary: data.error,
          confidence: "--",
          category: "Error",
        });
      }
      return;
    }

    handsIndicator.textContent = `Hands: ${data.hands_detected ? "DETECTED" : "NONE"}`;
    handsIndicator.style.color = data.hands_detected ? "#4ade80" : "#f87171";
    fpsIndicator.textContent = `FPS: ${data.fps ?? "--"}`;
    latencyIndicator.textContent = `AI: ${data.latency_ms ?? "--"} ms`;
    bufferIndicator.textContent = `Buffer: ${data.buffer_fill}/${data.buffer_capacity}`;

    renderTop3(data.top_predictions);

    const confPct = `${((data.confidence || 0) * 100).toFixed(1)}%`;
    if (data.recognized) {
      const guide = ISL_EXACT_GUIDE[data.gesture];
      updateHud({
        mode: "LIVE TRANSLATION",
        symbol: guide ? guide.symbol : "🖐️",
        primary: data.gesture,
        secondary: guide ? guide.motion : "Recognized Indian Sign Language gesture",
        confidence: confPct,
        category: labelCategoryMap[data.gesture] || "ISL Sign",
      });
      addWordToConversation(data.gesture, data.english);
    } else {
      if (!data.hands_detected && Date.now() - lastSpokenAt > 1000) {
        lastSpokenGesture = "";
      }
      updateHud({
        mode: "TRACKING",
        symbol: data.hands_detected ? "✋" : "📷",
        primary: data.hands_detected ? "Analyzing Hand Symbol..." : "Show Hand Symbol",
        secondary: data.hands_detected
          ? "Hold the motion smoothly for ~1s (watch Top-3 candidates below)"
          : "Place your hand(s) clearly inside the camera frame",
        confidence: confPct,
        category: data.hands_detected ? "Hands Visible" : "No Hands",
      });
    }
  } catch (err) {
    console.error(err);
  }
}

function setCameraUI(running) {
  startCamBtn.disabled = running;
  stopCamBtn.disabled = !running;
  resetBufBtn.disabled = !running;
  if (running) {
    camStatus.textContent = "Camera: Live";
    camStatus.className = "pill pill-ok";
    cameraStream.src = `/api/camera/stream?t=${Date.now()}`;
    cameraStream.classList.remove("hidden");
    cameraPlaceholder.classList.add("hidden");
    if (!pollTimer) {
      pollTimer = setInterval(pollCameraPrediction, 160);
    }
  } else {
    camStatus.textContent = "Camera: Stopped";
    camStatus.className = "pill pill-muted";
    cameraStream.removeAttribute("src");
    cameraStream.classList.add("hidden");
    cameraPlaceholder.classList.remove("hidden");
    handsIndicator.textContent = "Hands: --";
    fpsIndicator.textContent = "FPS: --";
    latencyIndicator.textContent = "AI: -- ms";
    bufferIndicator.textContent = "Buffer: 0/30";
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }
}

startCamBtn.addEventListener("click", async () => {
  startCamBtn.disabled = true;
  await fetch("/api/camera/start", { method: "POST" });
  setCameraUI(true);
});

stopCamBtn.addEventListener("click", async () => {
  await fetch("/api/camera/stop", { method: "POST" });
  setCameraUI(false);
});

function renderConversationFlows() {
  conversationFlowsEl.innerHTML = "";
  CONVERSATION_FLOWS.forEach((flow) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "flow-card";
    const seqText = flow.signs
      .map((s) => `${ISL_EXACT_GUIDE[s]?.symbol || "🖐️"} ${s}`)
      .join("  →  ");
    btn.innerHTML = `
      <div class="flow-title">${flow.title}</div>
      <div class="flow-seq">${seqText}</div>
      <div class="flow-meaning">${flow.meaning}</div>
    `;
    btn.addEventListener("click", () => {
      sentenceWords = [...flow.signs];
      renderSentence();
      const first = flow.signs[0];
      updateHud({
        mode: "CONVERSATION PREVIEW",
        symbol: ISL_EXACT_GUIDE[first]?.symbol || "🖐️",
        primary: flow.signs.join(" → "),
        secondary: flow.meaning,
        confidence: "Guide",
        category: "Conversation",
      });
      if (autoSpeak.checked) {
        speakText(formatConversationSentence(sentenceWords));
      }
    });
    conversationFlowsEl.appendChild(btn);
  });
}

async function loadProjectInfo() {
  try {
    const [infoRes, camRes] = await Promise.all([
      fetch("/api/info"),
      fetch("/api/camera/prediction"),
    ]);
    const info = await infoRes.json();
    const camData = await camRes.json();

    apiStatus.textContent = "API: Ready";
    apiStatus.className = "pill pill-ok";

    const testPerf = info.metrics?.test_performance;
    if (testPerf) {
      metricsSummary.textContent = `Model Accuracy: ${(testPerf.accuracy * 100).toFixed(1)}% | Macro F1: ${(testPerf.macro_f1 * 100).toFixed(1)}%`;
    }

    renderConversationFlows();

    const idToLabel = info.labels?.id_to_label || {};
    gestureGrid.innerHTML = "";
    Object.values(idToLabel).forEach((item) => {
      labelCategoryMap[item.display_label] = item.parent_label;
      const g = ISL_EXACT_GUIDE[item.display_label] || {
        symbol: "🖐️",
        hands: "1–2 Hands",
        start: "In front of chest",
        motion: "Perform sign clearly",
        avoid: "Do not leave camera frame",
      };
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "guide-card";
      btn.innerHTML = `
        <div class="guide-card-top">
          <div class="guide-card-left">
            <span class="guide-symbol">${g.symbol}</span>
            <span class="title">${item.display_label}</span>
          </div>
          <span class="cat">${item.parent_label}</span>
        </div>
        <ul class="step-list">
          <li><strong>Hands:</strong> ${g.hands}</li>
          <li><strong>Start:</strong> ${g.start}</li>
          <li><strong>Motion:</strong> ${g.motion}</li>
        </ul>
        <div class="avoid"><strong>Avoid:</strong> ${g.avoid}</div>
      `;
      btn.addEventListener("click", () => {
        updateHud({
          mode: "GUIDE PREVIEW",
          symbol: g.symbol,
          primary: item.display_label,
          secondary: g.motion,
          confidence: "Guide",
          category: item.parent_label,
        });
        addWordToConversation(item.display_label);
      });
      gestureGrid.appendChild(btn);
    });

    if (camData.running) {
      setCameraUI(true);
    }
  } catch (err) {
    apiStatus.textContent = "API: Error";
    console.error(err);
  }
}

loadProjectInfo();
