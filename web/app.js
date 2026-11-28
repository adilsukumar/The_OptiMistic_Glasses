// The OptiMistic Glasses - ISL Connected Translator & 500+ Dictionary Frontend
const byId = (id) => document.getElementById(id);

const startBtn = byId("start-cam-btn");
const stopBtn = byId("stop-cam-btn");
const resetBtn = byId("reset-buf-btn");
const modeSelect = byId("recognition-mode");
const autoSpeakCheck = byId("auto-speak");
const sensitivitySlider = byId("threshold-slider");
const speakTranscriptBtn = byId("speak-sentence-btn");
const undoWordBtn = byId("undo-word-btn");
const clearTranscriptBtn = byId("clear-transcript-btn");
const streamImg = byId("camera-stream");
const sentenceOutput = byId("conversation-sentence");

// Live Drafting elements
const draftContainer = byId("live-draft-container");
const draftText = byId("hud-draft-text");
const tokenChipsContainer = byId("hud-token-chips");

// Dictionary elements
const searchInput = byId("dict-search-input");
const categoryPillsContainer = byId("category-pills");
const dictGrid = byId("dictionary-grid");
const dictCountBadge = byId("dict-count-badge");

let running = false;
let requestPending = false;
let revision = 0;
let lastEventId = 0;
let transcript = [];
let latestEvent = null;

// Full Dictionary State
let allVocabulary = [];
let activeCategory = "ALL";
let searchQuery = "";

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || "Request failed.");
  return data;
}

function renderTranscript() {
  if (!transcript.length) {
    sentenceOutput.innerHTML = '<p class="placeholder-text">Completed connected sentences will appear here as they are spoken.</p>';
    speakTranscriptBtn.disabled = true;
    undoWordBtn.disabled = true;
    return;
  }

  speakTranscriptBtn.disabled = modeSelect.value === "counting";
  undoWordBtn.disabled = false;

  sentenceOutput.innerHTML = "";
  transcript.forEach((item, index) => {
    const bubble = document.createElement("div");
    bubble.className = "sentence-bubble";
    
    const header = document.createElement("div");
    header.className = "bubble-header";
    
    const numBadge = document.createElement("span");
    numBadge.className = "bubble-number";
    numBadge.textContent = `#${index + 1}`;
    
    const catBadge = document.createElement("span");
    catBadge.className = "bubble-category";
    catBadge.textContent = item.category || "Conversation";

    const playBtn = document.createElement("button");
    playBtn.className = "bubble-play-btn";
    playBtn.innerHTML = "🔊 Speak";
    playBtn.onclick = () => {
      api("/api/camera/speak", { event_ids: [item.id] });
    };

    header.append(numBadge, catBadge, playBtn);

    const engText = document.createElement("div");
    engText.className = "bubble-english";
    engText.textContent = item.english;

    bubble.append(header, engText);

    if (item.hindi) {
      const hinText = document.createElement("div");
      hinText.className = "bubble-hindi";
      hinText.textContent = item.hindi;
      bubble.append(hinText);
    }

    if (item.tokens && item.tokens.length > 1) {
      const tokensRow = document.createElement("div");
      tokensRow.className = "bubble-tokens";
      tokensRow.innerHTML = `<small>Signed:</small> ${item.tokens.map(t => `<span class="chip-mini">${t}</span>`).join(" ")}`;
      bubble.append(tokensRow);
    }

    sentenceOutput.appendChild(bubble);
  });

  // Auto-scroll to latest sentence
  sentenceOutput.scrollTop = sentenceOutput.scrollHeight;
}

function renderMode() {
  const counting = modeSelect.value === "counting";
  autoSpeakCheck.disabled = counting;
  sensitivitySlider.disabled = counting;
  byId("hud-badge").textContent = counting ? "COUNTING (1-10)" : "CONNECTED CONVERSATION";
  renderTranscript();
}

function setCamera(nextRunning) {
  startBtn.disabled = nextRunning;
  stopBtn.disabled = !nextRunning;
  resetBtn.disabled = !nextRunning;
  const status = byId("cam-status");
  status.textContent = nextRunning ? "Camera: Live" : "Camera: Stopped";
  status.className = nextRunning ? "pill pill-ok" : "pill pill-muted";
  if (running === nextRunning) return;
  running = nextRunning;
  streamImg.classList.toggle("hidden", !running);
  byId("camera-placeholder").classList.toggle("hidden", running);
  if (running) {
    streamImg.src = "/api/camera/stream";
  } else {
    streamImg.removeAttribute("src");
  }
}

function renderPrediction(data) {
  if (modeSelect.value !== data.mode) latestEvent = null;
  modeSelect.value = data.mode;
  autoSpeakCheck.checked = data.auto_speak;
  renderMode();
  setCamera(data.running);

  // New events processing
  for (const event of data.events) {
    if (event.id <= lastEventId) continue;
    lastEventId = event.id;
    if (data.mode === "conversation") {
      if (event.category !== "Counting") transcript.push(event);
    } else {
      transcript.push(event);
    }
    latestEvent = event;
  }

  const availableIds = new Set(data.events.map((e) => e.id));
  transcript = transcript.filter((e) => availableIds.has(e.id));
  renderTranscript();

  // HUD Status
  byId("hud-mode").textContent = data.error ? "CAMERA ERROR" : (data.running ? "LIVE" : "STANDBY");
  
  // Live Sentence Drafting Display
  if (data.running && data.live_preview && data.live_preview.active && data.live_preview.tokens.length > 0) {
    draftContainer.classList.remove("hidden");
    draftText.textContent = `"${data.live_preview.sentence}"`;
    tokenChipsContainer.innerHTML = data.live_preview.tokens
      .map((t) => `<span class="token-pill">${t}</span>`)
      .join('<span class="token-arrow">➔</span>');
    byId("hud-primary").textContent = data.live_preview.sentence;
    byId("hud-secondary").textContent = `Signing in progress (${data.live_preview.tokens.length} signs accumulated)... Lower hands or pause to speak`;
  } else {
    draftContainer.classList.add("hidden");
    byId("hud-primary").textContent = data.error || (data.running ? latestEvent?.english : null) || data.status;
    byId("hud-secondary").textContent = data.running ? data.status : "Standby";
  }

  byId("hud-confidence").textContent = !data.running || latestEvent?.confidence == null
    ? "" : `Confidence: ${Math.round(latestEvent.confidence * 100)}%`;
  byId("hands-indicator").textContent = `Foreground hands: ${data.hands_detected ? "In view" : "None"}`;
  byId("fps-indicator").textContent = `FPS: ${data.fps}`;
  byId("latency-indicator").textContent = `Latency: ${data.latency_ms} ms`;
}

async function poll() {
  const currentRevision = revision;
  try {
    if (!requestPending) {
      const response = await fetch("/api/camera/prediction");
      if (!response.ok) throw new Error("Camera status unavailable.");
      const data = await response.json();
      if (currentRevision === revision && !requestPending) renderPrediction(data);
    }
  } catch (error) {
    byId("hud-secondary").textContent = error.message;
  } finally {
    setTimeout(poll, running ? 100 : 500);
  }
}

async function apply(path, body, onSuccess = () => {}) {
  if (requestPending) return;
  requestPending = true;
  revision += 1;
  try {
    const data = await api(path, body);
    onSuccess(data);
  } catch (error) {
    byId("hud-primary").textContent = error.message;
  } finally {
    requestPending = false;
  }
}

// Control buttons
startBtn.addEventListener("click", () => apply("/api/camera/start", {}, () => {
  latestEvent = null;
  setCamera(true);
}));
stopBtn.addEventListener("click", () => apply("/api/camera/stop", {}, () => setCamera(false)));
resetBtn.addEventListener("click", () => apply("/api/camera/reset", {}, () => { latestEvent = null; }));
modeSelect.addEventListener("change", () => {
  latestEvent = null;
  renderMode();
  apply("/api/camera/mode", { mode: modeSelect.value });
});
autoSpeakCheck.addEventListener("change", () => apply("/api/camera/auto-speak", { enabled: autoSpeakCheck.checked }));
sensitivitySlider.addEventListener("input", () => { byId("threshold-val").textContent = `${sensitivitySlider.value}%`; });
sensitivitySlider.addEventListener("change", () => apply("/api/camera/threshold", { threshold: Number(sensitivitySlider.value) / 100 }));
speakTranscriptBtn.addEventListener("click", () => apply("/api/camera/speak", { event_ids: transcript.map((e) => e.id) }));
undoWordBtn.addEventListener("click", () => {
  transcript.pop();
  renderTranscript();
});
clearTranscriptBtn.addEventListener("click", () => apply("/api/camera/reset", { clear: true }, () => {
  transcript = [];
  latestEvent = null;
  renderTranscript();
}));

// ==========================================
// 540+ ISL Master Dictionary UI
// ==========================================
function renderCategoryPills(categories) {
  categoryPillsContainer.innerHTML = "";
  
  const allBtn = document.createElement("button");
  allBtn.className = activeCategory === "ALL" ? "cat-pill active" : "cat-pill";
  allBtn.textContent = `All Signs (${allVocabulary.length})`;
  allBtn.onclick = () => {
    activeCategory = "ALL";
    updatePillsActiveState();
    renderFilteredDictionary();
  };
  categoryPillsContainer.appendChild(allBtn);

  Object.entries(categories).forEach(([catName, count]) => {
    const btn = document.createElement("button");
    btn.className = activeCategory === catName ? "cat-pill active" : "cat-pill";
    btn.textContent = `${catName} (${count})`;
    btn.onclick = () => {
      activeCategory = catName;
      updatePillsActiveState();
      renderFilteredDictionary();
    };
    categoryPillsContainer.appendChild(btn);
  });
}

function updatePillsActiveState() {
  Array.from(categoryPillsContainer.children).forEach((btn) => {
    const isCurrent = (activeCategory === "ALL" && btn.textContent.startsWith("All")) ||
                      btn.textContent.startsWith(activeCategory);
    btn.classList.toggle("active", isCurrent);
  });
}

function renderFilteredDictionary() {
  const query = searchQuery.trim().toLowerCase();
  
  let filtered = allVocabulary.filter((item) => {
    const matchesCategory = activeCategory === "ALL" || item.category === activeCategory;
    if (!matchesCategory) return false;
    if (!query) return true;
    
    return (
      item.english.toLowerCase().includes(query) ||
      (item.hindi && item.hindi.includes(query)) ||
      item.gloss.toLowerCase().includes(query) ||
      (item.category && item.category.toLowerCase().includes(query)) ||
      (item.handshape && item.handshape.toLowerCase().includes(query)) ||
      (item.tags && item.tags.some((t) => t.includes(query)))
    );
  });

  dictCountBadge.textContent = `Showing ${filtered.length} of ${allVocabulary.length} Signs`;

  if (!filtered.length) {
    dictGrid.innerHTML = `
      <div class="empty-search">
        <p>No ISL signs found matching "<strong>${searchQuery}</strong>".</p>
        <small>Try searching another keyword, English word, Hindi term, or category.</small>
      </div>`;
    return;
  }

  // Display top 100 matching signs to maintain 60fps responsiveness
  const limit = 120;
  const displaySet = filtered.slice(0, limit);
  
  dictGrid.innerHTML = "";
  displaySet.forEach((sign) => {
    const card = document.createElement("article");
    card.className = "dict-card";

    const topRow = document.createElement("div");
    topRow.className = "dict-card-top";

    const titleBox = document.createElement("div");
    const h3 = document.createElement("h3");
    h3.textContent = sign.english;
    
    const hindiSpan = document.createElement("span");
    hindiSpan.className = "hindi-label";
    hindiSpan.textContent = sign.hindi || "";
    titleBox.append(h3, hindiSpan);

    const badgesBox = document.createElement("div");
    badgesBox.className = "badges-box";
    
    const handsBadge = document.createElement("span");
    handsBadge.className = "badge badge-hands";
    handsBadge.textContent = sign.hands || "1 Hand";

    const catBadge = document.createElement("span");
    catBadge.className = "badge badge-category";
    catBadge.textContent = sign.category || "General";

    badgesBox.append(handsBadge, catBadge);
    topRow.append(titleBox, badgesBox);

    const desc = document.createElement("p");
    desc.className = "dict-desc";
    desc.textContent = sign.handshape || (sign.instructions ? sign.instructions[0] : "");

    const instructionsList = document.createElement("ul");
    instructionsList.className = "dict-steps";
    if (sign.instructions && sign.instructions.length) {
      sign.instructions.forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        instructionsList.appendChild(li);
      });
    }

    const footRow = document.createElement("div");
    footRow.className = "dict-card-footer";
    
    const glossBadge = document.createElement("span");
    glossBadge.className = "gloss-tag";
    glossBadge.textContent = `GLOSS: ${sign.gloss}`;

    const videoLink = document.createElement("a");
    videoLink.className = "video-link";
    videoLink.href = sign.videoUrl || `https://www.youtube.com/results?search_query=ISLRTC+${sign.gloss}+Indian+Sign+Language`;
    videoLink.target = "_blank";
    videoLink.rel = "noopener noreferrer";
    videoLink.innerHTML = "▶ Watch Demo";

    footRow.append(glossBadge, videoLink);
    card.append(topRow, desc, instructionsList, footRow);
    dictGrid.appendChild(card);
  });

  if (filtered.length > limit) {
    const moreNotice = document.createElement("div");
    moreNotice.className = "more-notice";
    moreNotice.textContent = `+ ${filtered.length - limit} more signs matching. Refine your search above to narrow down.`;
    dictGrid.appendChild(moreNotice);
  }
}

// Search debounce
let debounceTimer;
searchInput.addEventListener("input", (e) => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    searchQuery = e.target.value;
    renderFilteredDictionary();
  }, 150);
});

async function loadDictionary() {
  try {
    const response = await fetch("/api/vocabulary");
    if (response.ok) {
      const data = await response.json();
      allVocabulary = data.vocabulary || [];
      renderCategoryPills(data.categories || {});
      renderFilteredDictionary();
      byId("vocab-status").textContent = `Dictionary: ${allVocabulary.length} Signs`;
    }
  } catch (err) {
    dictGrid.innerHTML = `<p class="error-msg">Could not load vocabulary: ${err.message}</p>`;
  }
}

async function initialize() {
  try {
    const [info, camera] = await Promise.all([
      api("/api/info"),
      fetch("/api/camera/prediction").then((res) => res.json()),
    ]);
    renderPrediction(camera);
    sensitivitySlider.value = Math.round(camera.threshold * 100);
    byId("threshold-val").textContent = `${sensitivitySlider.value}%`;
    byId("api-status").textContent = "API: Connected";
    byId("api-status").className = "pill pill-ok";
  } catch (error) {
    byId("api-status").textContent = "API: Offline";
    byId("hud-primary").textContent = error.message;
  } finally {
    loadDictionary();
    poll();
  }
}

initialize();
