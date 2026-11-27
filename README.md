# The OptiMistic Glasses

Smart glasses assistive AI platform powered by Raspberry Pi and Windows Desktop.

## Indian Sign Language (ISL) Live Translator

Real-time, zero-buffer Indian Sign Language (ISL) live translator with MediaPipe 21-point hand tracking, precision gesture recognition, and instant spoken English voice output through computer speakers.

### 1. Run in PowerShell (Desktop Camera Window + Offline Voice)
Run the live translator directly from PowerShell:

```powershell
# Start live camera translator with speech output:
python isl_app.py

# Or run via runner scripts:
.\run_isl.ps1
# Or double-click:
.\run_isl.bat

# Print the complete step-by-step ISL signing guide:
python isl_app.py --guide
```

#### Controls inside OpenCV window:
- `[q]` : Quit application
- `[c]` : Clear conversation sentence and transcript
- `[s]` : Speak the current conversation sentence aloud
- `[m]` : Toggle Camera Mirror mode (on/off)
- `[g]` : Print the complete ISL Signing Guide in PowerShell
- `[h]` : Help menu

---

### 2. Supported Vocabulary & Actions

| Icon | Sign | How to Sign It | Spoken Translation |
| :---: | :--- | :--- | :--- |
| 🙏 | **NAMASTE** | Press both open palms flat together at chest center (or 1 hand in center prayer) | *"Namaste"* (नमस्ते) |
| 🤲 | **HOW ARE YOU** | Hold both open palms up/forward in front of chest (or 1 hand cupped forward) | *"How are you?"* (आप कैसे हैं?) |
| 👌 | **I AM FINE** | OK sign: Thumb tip & Index tip form circle, other 3 fingers UP | *"I am fine"* (मैं ठीक हूँ) |
| ❓ | **WHAT ARE YOU DOING** | Raise Index, Middle, Ring UP ('W' sign) OR point both Index fingers UP | *"What are you doing?"* (आप क्या कर रहे हैं?) |
| 👦 | **I AM A BOY** | Cap visor gesture: Hand horizontal at forehead / temple with index & thumb near temple | *"I am a boy"* (मैं एक लड़का हूँ) |
| 👧 | **I AM A GIRL** | Earring gesture: Index finger touching / pointing along cheek or jawline | *"I am a girl"* (मैं एक लड़की हूँ) |
| 🆘 | **HELP ME** | Thumbs-up/fist on flat base palm, OR single fist held against chest | *"Help me"* (मेरी मदद करो) |
| 🤟 | **I LOVE YOU** | Thumb OUT, Index UP, Pinky UP (Middle & Ring folded into palm) | *"I love you"* (मैं तुमसे प्यार करता हूँ) |
| 👋 | **HELLO** | Raise open palm (5 fingers spread) beside head / ear height | *"Hello"* (हैलो / नमस्ते) |
| ✊ | **YES** | Closed fist held upright in mid-air (all 4 fingers curled) | *"Yes"* (हाँ) |
| 🤏 | **NO** | Authentic ISL beak: Thumb, Index & Middle tips pinching together | *"No"* (नहीं) |
| 👍 | **ALRIGHT** | Thumbs-Up sign (Thumb UP only, other 4 curled) | *"Alright"* (ठीक है) |
| 1️⃣ | **ONE** | Index finger pointing UP only (other fingers curled) | *"One (1)"* (एक) |
| 2️⃣ | **TWO** | Index UP and Middle UP in a 'V' shape (Counting 2) | *"Two (2)"* (दो) |
| 3️⃣ | **THREE** | Thumb OUT, Index UP, Middle UP (Counting 3) | *"Three (3)"* (तीन) |
| 4️⃣ | **FOUR** | 4 fingers (Index, Middle, Ring, Pinky) UP, Thumb folded in | *"Four (4)"* (चार) |
| 5️⃣ | **FIVE** | All 5 fingers extended and spread open at chest level | *"Five (5)"* (पाँच) |
| 6️⃣ | **SIX** | Pinky finger pointing UP only (other 4 fingers curled) | *"Six (6)"* (छह) |
| 7️⃣ | **SEVEN** | Thumb OUT and Index UP forming an 'L' shape (like a 7) | *"Seven (7)"* (सात) |
| 8️⃣ | **EIGHT** | Thumb OUT, Index UP, Middle UP, Ring UP (Counting 8) | *"Eight (8)"* (आठ) |
| 🔟 | **TEN** | Both open hands showing 5 + 5 spread fingers | *"Ten (10)"* (दस) |

---

### 3. Run on Localhost Web HUD (Optional Browser Mode)
```powershell
python server.py
```
Open **http://localhost:8000** in your browser.
