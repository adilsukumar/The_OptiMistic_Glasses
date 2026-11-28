# The OptiMistic Glasses — Indian Sign Language AI Platform

Assistive AI smart glasses platform for continuous Indian Sign Language (ISL) recognition, fluent connected speech synthesis, and natural conversational dialogue.

---

## 🌟 Major Capabilities

1. **Natural Connected Sentence Formation**:
   - When a mute or deaf person signs in Indian Sign Language, the system connects sequential gestures into fluid, grammatically correct English and Hindi sentences (e.g. `[I] ➔ [HUNGRY] ➔ [WANT] ➔ [FOOD]` is synthesized into `"I am hungry and I want some food."`).
   - Real-time sentence drafting HUD shows words accumulating live as you sign.
   - Automatic Sentence Boundary Detection: A natural pause (~1.8 seconds) or resting hands below the signing zone finalizes the thought.
   - Handles broken/separate sentences: Distinct thoughts signed after pauses are segmented into separate, sequential sentences with clean punctuation.

2. **Master 540+ Sign ISL Dictionary & Dataset**:
   - **541 authentic Indian Sign Language signs** mapped directly from ISLRTC national curriculum and AI4Bharat/Microsoft Research INCLUDE dataset.
   - Spans 15 comprehensive categories:
     - Greetings & Civilities (35 signs)
     - Questions & Inquiries (30 signs)
     - Pronouns, People & Identity (30 signs)
     - Family & Relationships (25 signs)
     - Food, Cooking & Dining (49 signs)
     - Medical, Health, Symptoms & Emergency (40 signs)
     - Emotions, Feelings & Descriptors (40 signs)
     - Time, Calendar, Days & Months (30 signs)
     - Numbers & Quantitative Counting (0-1000+) (30 signs)
     - Indian Sign Language Alphabet (A-Z) (26 signs)
     - Places, Travel, Transportation & Directions (34 signs)
     - Common Verbs & Everyday Actions (46 signs)
     - Education, School, Office & Work (22 signs)
     - Colors, Nature & Animals (26 signs)
     - Clothing & Accessories (20 signs)
     - Household & Gadgets (20 signs)
     - More Animals & Wildlife (18 signs)
     - Conversational Phrases & Dialogue (20 signs)

3. **Foreground Signer Priority & Zero-Buffer Engine**:
   - Automatically prioritizes the nearest foreground signer using palm depth geometry ($\ge 0.08$) and filters out people walking in the background.
   - Extended comfortable seated signing space ($0.20 \le y \le 0.82$).
   - Ultra-fast ~50ms response latency with zero buffering and no audio stutter.

4. **Speech Output (Text-to-Speech)**:
   - Automatically speaks finalized connected sentences aloud using native Windows SAPI / System.Speech synthesis.

---

## 🚀 Quick Start

Run either interface from the project root:

| Interface | Launch Command | Features |
| --- | --- | --- |
| **Interactive Web Dashboard** | `python server.py` (Open [http://localhost:8000](http://localhost:8000)) | Live video stream, connected sentence preview, spoken transcript bubbles, and searchable 540+ ISL master dictionary. |
| **Desktop OpenCV Camera HUD** | `python isl_app.py` or `.\run_isl.ps1` | Native low-latency camera HUD with live sign tokens, sentence drafting preview, and automatic voice synthesis. |

### Hotkeys in Desktop App (`isl_app.py`):
- **N**: Toggle mode between Continuous Conversation and Counting.
- **S**: Replay / speak the conversation transcript.
- **C**: Clear current tracking buffer and transcript.
- **M**: Toggle mirror mode.
- **G**: Print the complete gesture guide to terminal.
- **Q**: Exit application.

---

## 📖 How It Works

```
[Webcam Video] 
      │
      ▼
[Foreground Hand Tracker] ── (Filters background movement; locks nearest hands)
      │
      ▼
[Multi-Tier ISL Engine] ─── (Real-time handshape, landmark geometry & trajectory)
      │
      ▼
[Continuous Sentence Builder]
      │
      ├─► [Active Drafting HUD] ("I am hungry and need food...")
      │
      ▼ (Inter-sign pause or hands lowered)
[Grammar Synthesizer] ────► [Connected English & Hindi Sentence]
      │
      ▼
[Windows TTS Speaker] ────► Spoken aloud with zero delay!
```

### Connected Sentence Examples:

| Signs Signed in Sequence | Connected English Sentence | Hindi Translation |
| --- | --- | --- |
| `[I, HUNGRY, WANT, FOOD]` | **"I am hungry and I want some food."** | मुझे भूख लगी है और खाना चाहिए। |
| `[WHERE, HOSPITAL]` | **"Where is the nearest hospital?"** | अस्पताल कहाँ है? |
| `[HELP, ME, PLEASE]` | **"Please help me."** | कृपया मेरी मदद कीजिए। |
| `[DOCTOR, CALL, FAST]` | **"Please call a doctor immediately!"** | कृपया तुरंत डॉक्टर को बुलाएं! |
| `[WHAT, TIME, NOW]` | **"What is the time right now?"** | अभी क्या समय हुआ है? |
| `[WHERE, WASHROOM]` | **"Where is the washroom?"** | शौचालय कहाँ है? |
| `[THANK_YOU, VERY_MUCH]` | **"Thank you very much."** | बहुत-बहुत धन्यवाद। |
| `[TODAY, I, GO, HOME]` | **"Today I am going home."** | आज मैं घर जा रहा हूँ। |

---

## 📁 Project Structure

```
d:\optimistic\
├── data\
│   └── isl_master_vocabulary.json  # 541 curated authentic ISL signs
├── isl_vocabulary_db.py            # Fast O(1) dictionary search & query engine
├── isl_sentence_synthesizer.py     # Connected sentence builder & NLP grammar engine
├── isl_recognition_engine.py       # Multi-tier ISL recognition & foreground filter
├── isl_speech.py                   # Windows Text-To-Speech (TTS) engine
├── server.py                       # FastAPI / HTTP / MJPEG streaming server
├── isl_app.py                      # Standalone OpenCV camera application
├── web\
│   ├── index.html                  # Responsive dashboard with 540+ sign browser
│   ├── app.js                      # Live drafting controller & dictionary search
│   ├── style.css                   # Modern dark-mode UI styling
│   └── isl_vocabulary.json         # Web-optimized vocabulary catalog
├── run_isl.bat                     # 1-click Windows batch launcher
├── run_isl.ps1                     # 1-click PowerShell launcher
└── README.md                       # Documentation
```
