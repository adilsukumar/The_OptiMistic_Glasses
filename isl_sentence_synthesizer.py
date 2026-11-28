"""Natural Connected Sentence Synthesizer for Indian Sign Language (ISL).

Converts raw sequential ISL sign glosses into fluent, grammatically connected,
properly punctuated English and Hindi sentences. Manages inter-sign cadence,
cadence-based sentence boundaries, live preview accumulation, and clean TTS generation.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import isl_vocabulary_db as db


# Explicit template patterns for common ISL gloss sequences
_PATTERNS: List[Tuple[List[str], str, str, str]] = [
    # Pattern: [Tokens], English Translation, Hindi Translation, Category
    (["HELLO"], "Hello!", "नमस्ते!", "Greetings"),
    (["NAMASTE"], "Namaste.", "नमस्ते।", "Greetings"),
    (["GOOD", "MORNING"], "Good morning.", "शुभ प्रभात।", "Greetings"),
    (["GOOD", "AFTERNOON"], "Good afternoon.", "शुभ दोपहर।", "Greetings"),
    (["GOOD", "EVENING"], "Good evening.", "शुभ संध्या।", "Greetings"),
    (["GOOD", "NIGHT"], "Good night.", "शुभ रात्रि।", "Greetings"),
    (["HOW", "ARE", "YOU"], "How are you?", "आप कैसे हैं?", "Conversation"),
    (["HOW", "YOU"], "How are you doing?", "आप कैसे हैं?", "Conversation"),
    (["I", "FINE"], "I am fine.", "मैं ठीक हूँ।", "Conversation"),
    (["I", "AM", "FINE"], "I am fine, thank you.", "मैं ठीक हूँ, धन्यवाद।", "Conversation"),
    (["I", "FINE", "THANK", "YOU"], "I am fine, thank you.", "मैं ठीक हूँ, धन्यवाद।", "Conversation"),
    (["NICE", "MEET", "YOU"], "Nice to meet you.", "आपसे मिलकर अच्छा लगा।", "Conversation"),
    (["NICE", "TO", "MEET", "YOU"], "Nice to meet you.", "आपसे मिलकर अच्छा लगा।", "Conversation"),
    (["SEE", "YOU", "LATER"], "See you later.", "बाद में मिलते हैं।", "Conversation"),
    (["SEE", "YOU", "TOMORROW"], "See you tomorrow.", "कल मिलते हैं।", "Conversation"),
    (["THANK", "YOU"], "Thank you.", "धन्यवाद।", "Polite"),
    (["THANK", "YOU", "VERY", "MUCH"], "Thank you very much.", "बहुत-बहुत धन्यवाद।", "Polite"),
    (["PLEASE"], "Please.", "कृपया।", "Polite"),
    (["SORRY"], "I am sorry.", "मुझे माफ़ कीजिए।", "Polite"),
    (["EXCUSE", "ME"], "Excuse me.", "क्षमा करें।", "Polite"),
    (["WELCOME"], "You are welcome.", "आपका स्वागत है।", "Polite"),
    (["ALRIGHT"], "Alright, sounds good.", "ठीक है।", "Conversation"),
    (["YES"], "Yes.", "हाँ।", "Conversation"),
    (["NO"], "No.", "नहीं।", "Conversation"),
    (["YES", "I", "AGREE"], "Yes, I agree completely.", "हाँ, मैं पूरी तरह सहमत हूँ।", "Conversation"),
    (["NO", "I", "DISAGREE"], "No, I disagree.", "नहीं, मैं असहमत हूँ।", "Conversation"),

    # Needs & Dining
    (["I", "HUNGRY"], "I am hungry.", "मुझे भूख लगी है।", "Daily Needs"),
    (["I", "HUNGRY", "FOOD"], "I am hungry and need some food.", "मुझे भूख लगी है, खाना चाहिए।", "Daily Needs"),
    (["I", "HUNGRY", "WANT", "FOOD"], "I am hungry and I want some food.", "मुझे भूख लगी है और खाना चाहिए।", "Daily Needs"),
    (["WANT", "FOOD", "PLEASE"], "Please give me some food.", "कृपया मुझे खाना दीजिए।", "Daily Needs"),
    (["I", "THIRSTY"], "I am thirsty.", "मुझे प्यास लगी है।", "Daily Needs"),
    (["I", "THIRSTY", "WATER"], "I am thirsty and need water.", "मुझे प्यास लगी है, पानी चाहिए।", "Daily Needs"),
    (["I", "WANT", "WATER"], "I would like some water, please.", "मुझे पानी चाहिए।", "Daily Needs"),
    (["WANT", "WATER"], "I want some water, please.", "मुझे पानी चाहिए।", "Daily Needs"),
    (["WANT", "WATER", "PLEASE"], "Could I please have some water?", "कृपया मुझे थोड़ा पानी मिलेगा?", "Daily Needs"),
    (["WATER", "PLEASE"], "Water, please.", "कृपया पानी दीजिए।", "Daily Needs"),
    (["EAT", "FOOD"], "Let us eat food.", "खाना खाते हैं।", "Daily Needs"),
    (["DRINK", "WATER"], "I want to drink water.", "मुझे पानी पीना है।", "Daily Needs"),
    (["DRINK", "TEA"], "I would like some tea.", "मुझे चाय चाहिए।", "Daily Needs"),
    (["DRINK", "COFFEE"], "I would like some coffee.", "मुझे कॉफ़ी चाहिए।", "Daily Needs"),
    (["FOOD", "TASTE", "GOOD"], "The food is delicious.", "खाना बहुत स्वादिष्ट है।", "Daily Needs"),
    (["I", "FULL"], "I am full and satisfied.", "मेरा पेट भर गया है।", "Daily Needs"),

    # Inquiries & Questions
    (["WHAT", "YOUR", "NAME"], "What is your name?", "आपका नाम क्या है?", "Questions"),
    (["WHAT", "IS", "YOUR", "NAME"], "What is your name?", "आपका नाम क्या है?", "Questions"),
    (["MY", "NAME"], "My name is...", "मेरा नाम है...", "Conversation"),
    (["WHAT", "ARE", "YOU", "DOING"], "What are you doing?", "आप क्या कर रहे हैं?", "Questions"),
    (["WHAT", "YOU", "DOING"], "What are you doing?", "आप क्या कर रहे हैं?", "Questions"),
    (["WHERE", "ARE", "YOU", "GOING"], "Where are you going?", "आप कहाँ जा रहे हैं?", "Questions"),
    (["WHERE", "YOU", "GO"], "Where are you going?", "आप कहाँ जा रहे हैं?", "Questions"),
    (["WHERE", "IS", "HOSPITAL"], "Where is the hospital?", "अस्पताल कहाँ है?", "Medical"),
    (["WHERE", "HOSPITAL"], "Where is the nearest hospital?", "अस्पताल कहाँ है?", "Medical"),
    (["WHERE", "IS", "WASHROOM"], "Where is the washroom?", "शौचालय कहाँ है?", "Questions"),
    (["WHERE", "WASHROOM"], "Where is the washroom?", "शौचालय कहाँ है?", "Questions"),
    (["WHERE", "TOILET"], "Where is the toilet?", "शौचालय कहाँ है?", "Questions"),
    (["WHERE", "POLICE"], "Where is the police station?", "पुलिस स्टेशन कहाँ है?", "Emergency"),
    (["WHERE", "BUS", "STOP"], "Where is the bus stop?", "बस स्टॉप कहाँ है?", "Travel"),
    (["WHERE", "RAILWAY", "STATION"], "Where is the railway station?", "रेलवे स्टेशन कहाँ है?", "Travel"),
    (["WHAT", "TIME"], "What time is it?", "क्या समय हुआ है?", "Questions"),
    (["WHAT", "TIME", "NOW"], "What is the time right now?", "अभी क्या समय हुआ है?", "Questions"),
    (["HOW", "MUCH", "COST"], "How much does this cost?", "इसकी कीमत क्या है?", "Questions"),
    (["HOW", "MUCH"], "How much is this?", "यह कितना है?", "Questions"),
    (["CAN", "YOU", "HELP", "ME"], "Can you please help me?", "क्या आप मेरी मदद कर सकते हैं?", "Requests"),
    (["HELP", "ME", "PLEASE"], "Please help me.", "कृपया मेरी मदद कीजिए।", "Emergency"),
    (["HELP", "ME"], "Help me, please.", "मेरी मदद करो।", "Emergency"),

    # Emergency & Health
    (["EMERGENCY"], "This is an emergency!", "यह आपातकाल है!", "Emergency"),
    (["CALL", "DOCTOR"], "Please call a doctor quickly.", "कृपया डॉक्टर को बुलाएं।", "Medical"),
    (["DOCTOR", "CALL", "FAST"], "Please call a doctor immediately!", "कृपया तुरंत डॉक्टर को बुलाएं!", "Medical"),
    (["CALL", "AMBULANCE"], "Please call an ambulance right away!", "तुरंत एम्बुलेंस बुलाइए!", "Emergency"),
    (["CALL", "POLICE"], "Please call the police immediately!", "तुरंत पुलिस को बुलाइए!", "Emergency"),
    (["I", "SICK"], "I am feeling sick.", "मैं बीमार महसूस कर रहा हूँ।", "Medical"),
    (["I", "SICK", "FEVER"], "I am sick with a fever.", "मुझे बुखार है और मैं बीमार हूँ।", "Medical"),
    (["I", "HAVE", "HEADACHE"], "I have a severe headache.", "मेरे सिर में बहुत तेज़ दर्द है।", "Medical"),
    (["HEADACHE", "PAIN"], "I have a bad headache.", "मेरे सिर में दर्द है।", "Medical"),
    (["STOMACH", "PAIN"], "I have severe stomach pain.", "मेरे पेट में तेज़ दर्द है।", "Medical"),
    (["CHEST", "PAIN"], "I am having chest pain, please help!", "मेरे सीने में दर्द हो रहा है, मदद करें!", "Emergency"),
    (["I", "NEED", "MEDICINE"], "I need my medicine.", "मुझे अपनी दवा चाहिए।", "Medical"),
    (["ACCIDENT", "HELP"], "There has been an accident, please help!", "दुर्घटना हो गई है, मदद करें!", "Emergency"),
    (["FIRE", "HELP"], "Fire! Please help and evacuate!", "आग लग गई है! मदद करें!", "Emergency"),
    (["DANGER"], "Warning, this is dangerous!", "खतरा है, सावधान रहें!", "Emergency"),

    # Feelings & Identity
    (["I", "BOY"], "I am a boy.", "मैं एक लड़का हूँ।", "Identity"),
    (["I", "AM", "A", "BOY"], "I am a boy.", "मैं एक लड़का हूँ।", "Identity"),
    (["I", "GIRL"], "I am a girl.", "मैं एक लड़की हूँ।", "Identity"),
    (["I", "AM", "A", "GIRL"], "I am a girl.", "मैं एक लड़की हूँ।", "Identity"),
    (["I", "DEAF"], "I am deaf.", "मैं बधिर हूँ।", "Identity"),
    (["I", "MUTE"], "I am mute and communicate through sign language.", "मैं मूक हूँ और सांकेतिक भाषा से बात करता हूँ।", "Identity"),
    (["I", "LOVE", "YOU"], "I love you.", "मैं तुमसे प्यार करता हूँ।", "Emotions"),
    (["I", "LOVE", "MY", "FAMILY"], "I love my family.", "मुझे अपने परिवार से प्यार है।", "Family"),
    (["I", "HAPPY"], "I am very happy today.", "मैं आज बहुत खुश हूँ।", "Emotions"),
    (["I", "SAD"], "I am feeling sad.", "मैं उदास महसूस कर रहा हूँ।", "Emotions"),
    (["I", "TIRED"], "I am very tired and need rest.", "मैं बहुत थक गया हूँ और आराम चाहिए।", "Emotions"),
    (["I", "TIRED", "SLEEP"], "I am tired and want to sleep.", "मैं थक गया हूँ और सोना चाहता हूँ।", "Daily Needs"),
    (["LET", "US", "GO"], "Let us go now.", "चलो अब चलें।", "Conversation"),
    (["WAIT", "A", "MINUTE"], "Wait a minute, please.", "कृपया एक मिनट रुकिए।", "Conversation"),
    (["I", "DONT", "UNDERSTAND"], "I do not understand.", "मुझे समझ नहीं आया।", "Conversation"),
    (["PLEASE", "REPEAT"], "Could you please repeat that?", "कृपया दोबारा दोहराएं।", "Polite"),
    (["TODAY", "I", "GO", "HOME"], "Today I am going home.", "आज मैं घर जा रहा हूँ।", "Travel"),
    (["TODAY", "I", "GO", "SCHOOL"], "Today I am going to school.", "आज मैं स्कूल जा रहा हूँ।", "Education"),
    (["TODAY", "I", "GO", "HOSPITAL"], "Today I am going to the hospital.", "आज मैं अस्पताल जा रहा हूँ।", "Medical"),
]


def _normalize_token(token: str) -> str:
    """Standardize sign string to clean uppercase token."""
    t = str(token).strip().upper()
    t = re.sub(r"[^A-Z0-9_]", "_", t)
    return re.sub(r"_+", "_", t).strip("_")


def synthesize_sentence(tokens: List[str]) -> Dict[str, Any]:
    """Convert an accumulated list of ISL sign tokens into a fluent sentence.
    
    Returns dict with keys:
      - sentence: Natural English sentence
      - hindi: Hindi translation
      - category: Dominant theme
      - tokens: Clean token list
    """
    if not tokens:
        return {"sentence": "", "hindi": "", "category": "General", "tokens": []}

    cleaned = [_normalize_token(t) for t in tokens if t]
    if not cleaned:
        return {"sentence": "", "hindi": "", "category": "General", "tokens": []}

    # 1. Check exact pattern matches
    for pattern, eng, hin, cat in _PATTERNS:
        if cleaned == pattern:
            return {"sentence": eng, "hindi": hin, "category": cat, "tokens": cleaned}

    # 2. Check sub-sequence / fuzzy pattern match if signs match closely
    cleaned_set = set(cleaned)
    best_match = None
    best_overlap = 0.0
    for pattern, eng, hin, cat in _PATTERNS:
        pat_set = set(pattern)
        if pat_set.issubset(cleaned_set) or cleaned_set.issubset(pat_set):
            overlap = len(pat_set.intersection(cleaned_set)) / max(len(pat_set), len(cleaned_set))
            if overlap > best_overlap and overlap >= 0.70:
                best_overlap = overlap
                best_match = (eng, hin, cat)

    if best_match and best_overlap >= 0.75:
        return {"sentence": best_match[0], "hindi": best_match[1], "category": best_match[2], "tokens": cleaned}

    # 3. Rule-Based Natural English & Hindi Grammar Assembly
    # Resolve metadata for each token from ISL database
    resolved_signs = []
    for t in cleaned:
        sign = db.get_sign(t)
        if sign:
            resolved_signs.append(sign)
        else:
            resolved_signs.append({
                "gloss": t,
                "english": t.title().replace("_", " "),
                "hindi": t,
                "grammar_role": "noun",
                "category": "Conversation"
            })

    # Check question indicators
    is_question = any(s.get("grammar_role") == "question" or s["gloss"] in (
        "WHAT", "WHERE", "WHO", "WHEN", "WHY", "HOW", "WHICH", "CAN_YOU"
    ) for s in resolved_signs)

    # Check emergency indicators
    is_emergency = any(s.get("grammar_role") == "emergency" or s["gloss"] in (
        "HELP", "EMERGENCY", "DANGER", "FIRE", "ACCIDENT", "AMBULANCE", "POLICE"
    ) for s in resolved_signs)

    # Build fluent English phrase
    words = [s["english"] for s in resolved_signs]
    
    # Handle consecutive duplicates (e.g. [WATER, WATER] -> [WATER])
    deduped_words = []
    for w in words:
        if not deduped_words or deduped_words[-1].lower() != w.lower():
            deduped_words.append(w)
            
    # Grammar connector heuristics
    raw_sentence = " ".join(deduped_words)
    
    # Common grammar transformations
    transforms = [
        (r"\bI Hungry\b", "I am hungry"),
        (r"\bI Thirsty\b", "I am thirsty"),
        (r"\bI Sick\b", "I am sick"),
        (r"\bI Fine\b", "I am fine"),
        (r"\bI Tired\b", "I am tired"),
        (r"\bI Happy\b", "I am happy"),
        (r"\bI Sad\b", "I am sad"),
        (r"\bI Boy\b", "I am a boy"),
        (r"\bI Girl\b", "I am a girl"),
        (r"\bI Deaf\b", "I am deaf"),
        (r"\bI Mute\b", "I am mute"),
        (r"\bWhere Hospital\b", "Where is the hospital"),
        (r"\bWhere Washroom\b", "Where is the washroom"),
        (r"\bWhere Toilet\b", "Where is the washroom"),
        (r"\bWhere Police\b", "Where is the police station"),
        (r"\bWhat Time\b", "What time is it"),
        (r"\bHelp Me\b", "Please help me"),
        (r"\bThank You Very Much\b", "Thank you very much"),
    ]
    for pattern, repl in transforms:
        raw_sentence = re.sub(pattern, repl, raw_sentence, flags=re.IGNORECASE)

    # Clean punctuation and capitalization
    raw_sentence = raw_sentence.strip()
    if raw_sentence:
        raw_sentence = raw_sentence[0].upper() + raw_sentence[1:]
        if not raw_sentence.endswith((".", "?", "!")):
            if is_question:
                raw_sentence += "?"
            elif is_emergency:
                raw_sentence += "!"
            else:
                raw_sentence += "."

    # Hindi synthesis
    hindi_words = [s.get("hindi", "") for s in resolved_signs if s.get("hindi")]
    deduped_hindi = []
    for hw in hindi_words:
        if not deduped_hindi or deduped_hindi[-1] != hw:
            deduped_hindi.append(hw)
    hindi_sentence = " ".join(deduped_hindi)
    if hindi_sentence and not hindi_sentence.endswith(("।", "?", "!")):
        hindi_sentence += "।"

    category = "Emergency" if is_emergency else ("Questions" if is_question else resolved_signs[0].get("category", "Conversation"))

    return {
        "sentence": raw_sentence,
        "hindi": hindi_sentence,
        "category": category,
        "tokens": cleaned
    }


class ContinuousSentenceBuilder:
    """Manages continuous sign accumulation, cadence, and sentence boundaries.
    
    When a person signs:
    1. Words signed within `cadence_timeout` (~1.8s) are chained into the current active sentence.
    2. A rest/pause (>1.8s) or hands lowered concludes the sentence.
    3. The completed sentence is polished, spoken, and saved.
    4. Subsequent signs naturally form a brand new, separate sentence.
    """

    def __init__(self, cadence_timeout: float = 1.8, cooldown_timeout: float = 0.7):
        self.cadence_timeout = cadence_timeout
        self.cooldown_timeout = cooldown_timeout
        self.active_tokens: List[str] = []
        self.last_sign_time: Optional[float] = None
        self.last_sign_name: Optional[str] = None
        self.completed_sentences: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Clear all active tokens and buffer."""
        self.active_tokens.clear()
        self.last_sign_time = None
        self.last_sign_name = None

    def add_sign(self, sign_name: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Add a recognized sign to the active sentence buffer.
        
        If a new sign arrives after cadence_timeout, the PREVIOUS sentence is finalized
        and emitted, and the new sign starts a fresh sentence.
        """
        now = time.monotonic() if now is None else now
        norm_name = _normalize_token(sign_name)
        if not norm_name:
            return None

        finalized = None
        # Check if previous sentence expired due to pause
        if self.last_sign_time is not None and now - self.last_sign_time > self.cadence_timeout:
            if self.active_tokens:
                finalized = self.finalize_sentence(now=now)

        # Check duplicate cooldown
        if norm_name == self.last_sign_name and self.last_sign_time is not None:
            if now - self.last_sign_time < self.cooldown_timeout:
                return finalized

        self.active_tokens.append(norm_name)
        self.last_sign_name = norm_name
        self.last_sign_time = now
        return finalized

    def check_pause(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Check if inter-sign pause indicates sentence completion."""
        now = time.monotonic() if now is None else now
        if not self.active_tokens or self.last_sign_time is None:
            return None

        if now - self.last_sign_time >= self.cadence_timeout:
            return self.finalize_sentence(now=now)
        return None

    def get_preview(self) -> Dict[str, Any]:
        """Return live draft preview of sentence currently accumulating."""
        if not self.active_tokens:
            return {"active": False, "sentence": "", "tokens": [], "token_count": 0}
        preview = synthesize_sentence(self.active_tokens)
        return {
            "active": True,
            "sentence": preview["sentence"],
            "hindi": preview["hindi"],
            "tokens": list(self.active_tokens),
            "token_count": len(self.active_tokens),
            "category": preview["category"]
        }

    def finalize_sentence(self, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Finalize the active tokens into a completed sentence and reset active buffer."""
        if not self.active_tokens:
            return None
        now = time.monotonic() if now is None else now
        res = synthesize_sentence(self.active_tokens)
        res["timestamp"] = now
        res["id"] = len(self.completed_sentences) + 1
        self.completed_sentences.append(res)
        self.reset()
        return res
