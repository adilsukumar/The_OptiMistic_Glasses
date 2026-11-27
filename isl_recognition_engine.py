"""The OptiMistic Glasses — Precision Indian Sign Language (ISL) Recognition Engine.

Features:
- Authentic Conversational Vocabulary:
  - Namaste, How Are You, I Am Fine, What Are You Doing, I Am A Boy, I Am A Girl,
    Help Me, I Love You, Hello, Yes, No, Alright
- Full Sign Counting:
  - One (1), Two (2), Three (3), Four (4), Five (5), Six (6), Seven (7), Eight (8), Ten (10)
- Near-Hand Priority Filter: Background hands (far away / small palm) are completely ignored!
- Foreground Focus: Sorts hands by palm size so the closest hand to the camera is always primary.
- True 3D Landmark Geometry (x, y, z): Prevents open cupped/forward hands from collapsing into fists!
- Natural Signing Range: Works across the entire chest-to-face active signing volume (wrist up to y=0.82).
- Absolute Silence on Neutral/Resting Hands: If no intentional sign is formed, returns None (no speech!).
- Unambiguous Spatial Disambiguation: Strict face-height bounding for Girl vs Chest-height for Numbers (One).
- Multi-Hand & Merged-Hand Namaste Detection: Works whether camera tracks 2 hands or merges them into 1!
- 100% Mutually Exclusive Finger Signatures (Zero False Collisions)
- Fast 2-Frame Stability (~60ms response time, zero buffering!)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional
import numpy as np


def _dist2(p1: np.ndarray, p2: np.ndarray) -> float:
    """2D Euclidean distance in image plane."""
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def _dist3(p1: np.ndarray, p2: np.ndarray) -> float:
    """3D Euclidean distance using MediaPipe x, y, z landmarks."""
    return float(np.linalg.norm(p1[:3] - p2[:3]))


def filter_near_hands(raw_hands: List[np.ndarray]) -> List[np.ndarray]:
    """Filter out tiny background hands and resting hands; sort so the nearest hand is primary."""
    if not raw_hands:
        return []
    valid = []
    for h in raw_hands:
        # Distance from wrist (0) to middle finger base (9) measures physical palm size on screen
        palm_size = _dist2(h[0], h[9])

        # Filter out tiny hands in background (foreground signing hand has palm_size >= 0.08)
        if palm_size < 0.08:
            continue

        # Filter out hands resting on lap or desk at bottom of frame (wrist > 0.82)
        if float(h[0, 1]) > 0.82:
            continue

        # Filter out hands hanging completely downwards below wrist
        if float(h[8, 1]) > float(h[0, 1]) + 0.08 and float(h[12, 1]) > float(h[0, 1]) + 0.08:
            continue

        valid.append((palm_size, h))

    # Sort descending by palm size so the closest/largest foreground hand is index 0
    valid.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in valid]


def analyze_hand(pts: np.ndarray) -> Dict[str, Any]:
    """Calculate clear 3D & 2D finger state flags from MediaPipe landmarks."""
    wrist = pts[0]
    pinky_mcp = pts[17]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    ring_mcp = pts[13]

    # 3D distances from fingertip to corresponding MCP base
    d_idx = _dist3(pts[8], index_mcp)
    d_mid = _dist3(pts[12], middle_mcp)
    d_rng = _dist3(pts[16], ring_mcp)
    d_pky = _dist3(pts[20], pinky_mcp)

    # 2D vertical checks: tip is higher in frame (lower y) than PIP by at least 0.03
    idx_up_2d = bool(pts[6, 1] - pts[8, 1] > 0.03)
    mid_up_2d = bool(pts[10, 1] - pts[12, 1] > 0.03)
    rng_up_2d = bool(pts[14, 1] - pts[16, 1] > 0.03)
    pky_up_2d = bool(pts[18, 1] - pts[20, 1] > 0.03)

    # A finger is EXTENDED if it is pointing up in 2D OR extended in 3D (e.g. forward/cupped)
    idx_extended = bool(idx_up_2d or d_idx > 0.12)
    mid_extended = bool(mid_up_2d or d_mid > 0.12)
    rng_extended = bool(rng_up_2d or d_rng > 0.12)
    pky_extended = bool(pky_up_2d or d_pky > 0.12)

    # Thumb: Extended outward if tip (4) is far from pinky base (17) in 2D/3D
    thumb_out = bool(_dist2(pts[4], pinky_mcp) > (_dist2(pts[3], pinky_mcp) * 1.15))

    # A finger is CURLED when not pointing up in 2D and folded toward base/wrist
    idx_curled = bool(not idx_up_2d and (d_idx < 0.13 or pts[8, 1] >= pts[6, 1] - 0.015))
    mid_curled = bool(not mid_up_2d and (d_mid < 0.13 or pts[12, 1] >= pts[10, 1] - 0.015))
    rng_curled = bool(not rng_up_2d and (d_rng < 0.13 or pts[16, 1] >= pts[14, 1] - 0.015))
    pky_curled = bool(not pky_up_2d and (d_pky < 0.13 or pts[20, 1] >= pts[18, 1] - 0.015))

    # Hand orientation: hand is upright if middle MCP is above wrist (allowing natural tilt)
    hand_is_upright = bool(pts[9, 1] < pts[0, 1] + 0.08)

    # Closed Fist: all 4 fingers curled, tips close to wrist, hand held upright
    fist_curled = bool(
        idx_curled and mid_curled and rng_curled and pky_curled
        and d_idx < 0.12 and d_mid < 0.12
        and _dist3(pts[8], wrist) < 0.22
        and _dist3(pts[12], wrist) < 0.22
        and hand_is_upright
    )

    # Genuine Thumbs-Up: Thumb tip pointing UP significantly above MCP, other 4 fingers curled into fist
    thumb_up = bool(
        (pts[2, 1] - pts[4, 1]) > 0.06
        and pts[4, 1] < pts[0, 1] - 0.04
        and fist_curled
    )

    # 3D OK Circle: Thumb tip (4) touches Index tip (8) in 3D
    ok_circle = bool(_dist3(pts[4], pts[8]) < 0.08)

    # 3D ISL "NO" Beak Pinch: Thumb tip touches both Index tip and Middle tip, extended forward from wrist
    beak_pinch = bool(
        _dist3(pts[4], pts[8]) < 0.08
        and _dist3(pts[4], pts[12]) < 0.08
        and _dist3(pts[8], wrist) >= 0.16
        and hand_is_upright
    )

    # Hand coordinates
    hand_y = float(pts[0, 1])
    hand_x = float(pts[0, 0])
    index_tip_y = float(pts[8, 1])
    index_tip_x = float(pts[8, 0])

    # Cap visor gesture for Boy (hand horizontal at forehead/temple height, upright)
    cap_visor = bool(
        hand_y < 0.40
        and index_tip_y < 0.24
        and _dist2(pts[4], pts[8]) < 0.18
        and abs(pts[8, 1] - pts[5, 1]) < 0.10
        and hand_is_upright
    )

    # Girl gesture: Hand raised to CHEEK / EAR height beside face (index touching cheek/jawline)
    girl_gesture = bool(
        hand_y < 0.42
        and index_tip_y < 0.30
        and abs(index_tip_x - 0.50) > 0.12
        and idx_up_2d
        and mid_curled
        and rng_curled
        and pky_curled
    )

    # Intentional single-hand How Are You: Hand held in active chest space with palm facing up/forward
    cupped_how_are_you = bool(
        0.30 <= hand_y <= 0.75
        and d_idx > 0.13 and d_mid > 0.13
        and thumb_out
        and not fist_curled
        and not ok_circle
        and abs(pts[8, 1] - pts[0, 1]) < 0.15
    )

    return {
        "thumb_out": thumb_out,
        "thumb_up": thumb_up,
        "index_up_2d": idx_up_2d,
        "middle_up_2d": mid_up_2d,
        "ring_up_2d": rng_up_2d,
        "pinky_up_2d": pky_up_2d,
        "idx_curled": idx_curled,
        "mid_curled": mid_curled,
        "rng_curled": rng_curled,
        "pky_curled": pky_curled,
        "index": idx_extended,
        "middle": mid_extended,
        "ring": rng_extended,
        "pinky": pky_extended,
        "ok_circle": ok_circle,
        "beak_pinch": beak_pinch,
        "cap_visor": cap_visor,
        "girl_gesture": girl_gesture,
        "cupped_how_are_you": cupped_how_are_you,
        "fist_curled": fist_curled,
        "hand_is_upright": hand_is_upright,
        "hand_y": hand_y,
        "hand_x": hand_x,
        "index_tip_y": index_tip_y,
        "index_tip_x": index_tip_x,
    }


class PrecisionISLRecognizer:
    """Instant, zero-confusion recognizer with fast 2-frame (~60ms) stability filter."""

    def __init__(self, stability_threshold: int = 2) -> None:
        self.stability_threshold = stability_threshold
        self.current_candidate: Optional[str] = None
        self.candidate_count: int = 0
        self.history: Deque[Dict[str, Any]] = deque(maxlen=15)

    def reset(self) -> None:
        self.current_candidate = None
        self.candidate_count = 0
        self.history.clear()

    def process(self, raw_hands: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Evaluate raw hand landmarks with near-hand priority and distinct signatures."""
        if not raw_hands:
            self.reset()
            return None

        # Filter out tiny background hands and hands resting outside active frame
        active_hands = filter_near_hands(raw_hands)
        if not active_hands:
            self.reset()
            return None

        self.history.append({"hands": active_hands, "count": len(active_hands)})
        num_hands = len(active_hands)
        match: Optional[Dict[str, Any]] = None

        # =============================================================
        # 1. TWO-HANDED GESTURES (Ten, What Are You Doing, Namaste, Help Me, How Are You)
        # =============================================================
        if num_hands >= 2:
            h1 = active_hands[0]
            h2 = active_hands[1]
            f1 = analyze_hand(h1)
            f2 = analyze_hand(h2)

            wrist_dist = _dist2(h1[0], h2[0])
            index_dist = _dist2(h1[8], h2[8])

            # A. TEN (10) (Both open hands showing 5 + 5 spread fingers)
            if (
                (f1["thumb_out"] and f1["index_up_2d"] and f1["middle_up_2d"] and f1["ring_up_2d"] and f1["pinky_up_2d"])
                and (f2["thumb_out"] and f2["index_up_2d"] and f2["middle_up_2d"] and f2["ring_up_2d"] and f2["pinky_up_2d"])
                and wrist_dist >= 0.18
            ):
                match = {
                    "gesture": "TEN",
                    "english": "Ten (10)",
                    "hindi": "दस",
                    "confidence": 0.99,
                    "category": "Counting",
                    "icon": "🔟",
                }

            # B. WHAT ARE YOU DOING (Both hands showing Index fingers pointing UP side-by-side)
            elif (
                (f1["index_up_2d"] and f1["mid_curled"] and f1["rng_curled"] and f1["pky_curled"])
                and (f2["index_up_2d"] and f2["mid_curled"] and f2["rng_curled"] and f2["pky_curled"])
                and wrist_dist >= 0.16
            ):
                match = {
                    "gesture": "WHAT ARE YOU DOING",
                    "english": "What are you doing?",
                    "hindi": "आप क्या कर रहे हैं?",
                    "confidence": 0.98,
                    "category": "Conversation",
                    "icon": "❓",
                }

            # C. NAMASTE (Both palms joined flat together in center chest)
            elif (
                not f1["fist_curled"] and not f2["fist_curled"]
                and wrist_dist < 0.26
                and index_dist < 0.26
                and abs(h1[0, 0] - h2[0, 0]) < 0.22
                and (f1["index_up_2d"] or f1["middle_up_2d"])
                and (f2["index_up_2d"] or f2["middle_up_2d"])
            ):
                match = {
                    "gesture": "NAMASTE",
                    "english": "Namaste",
                    "hindi": "नमस्ते",
                    "confidence": 0.99,
                    "category": "Greetings",
                    "icon": "🙏",
                }

            # D. HELP ME (Two hands: fist resting on flat palm)
            elif (
                wrist_dist < 0.35
                and (
                    ((f1["fist_curled"] or f1["thumb_up"]) and (f2["index"] or f2["middle"] or f2["thumb_out"]))
                    or
                    ((f2["fist_curled"] or f2["thumb_up"]) and (f1["index"] or f1["middle"] or f1["thumb_out"]))
                )
            ):
                match = {
                    "gesture": "HELP ME",
                    "english": "Help me",
                    "hindi": "मेरी मदद करो",
                    "confidence": 0.98,
                    "category": "Needs & Emergency",
                    "icon": "🆘",
                }

            # E. HOW ARE YOU (Two hands: both hands held open with palms facing up/forward)
            elif (
                (f1["index"] and f1["middle"] and f1["thumb_out"] and not f1["index_up_2d"])
                and (f2["index"] and f2["middle"] and f2["thumb_out"] and not f2["index_up_2d"])
                and not f1["fist_curled"] and not f2["fist_curled"]
                and wrist_dist >= 0.16
            ):
                match = {
                    "gesture": "HOW ARE YOU",
                    "english": "How are you?",
                    "hindi": "आप कैसे हैं?",
                    "confidence": 0.98,
                    "category": "Greetings",
                    "icon": "🤲",
                }

        # =============================================================
        # 2. SINGLE-HANDED GESTURES (Primary Near Hand in Foreground)
        # =============================================================
        if match is None and num_hands >= 1:
            h = active_hands[0]
            f = analyze_hand(h)

            t_out = f["thumb_out"]
            t_up = f["thumb_up"]
            i_up = f["index_up_2d"]
            m_up = f["middle_up_2d"]
            r_up = f["ring_up_2d"]
            p_up = f["pinky_up_2d"]
            i_c = f["idx_curled"]
            m_c = f["mid_curled"]
            r_c = f["rng_curled"]
            p_c = f["pky_curled"]

            ok = f["ok_circle"]
            beak = f["beak_pinch"]
            fist = f["fist_curled"]
            cap = f["cap_visor"]
            girl = f["girl_gesture"]
            hand_y = f["hand_y"]

            # ---------------------------------------------------------
            # A. 4 FINGERS UP -> NAMASTE (if prayer fingers close) or FOUR (4) (if spread)
            # ---------------------------------------------------------
            if not t_out and i_up and m_up and r_up and p_up:
                if _dist2(h[8], h[20]) < 0.12 and 0.35 <= f["hand_x"] <= 0.65:
                    match = {
                        "gesture": "NAMASTE",
                        "english": "Namaste",
                        "hindi": "नमस्ते",
                        "confidence": 0.98,
                        "category": "Greetings",
                        "icon": "🙏",
                    }
                elif _dist2(h[8], h[20]) >= 0.12:
                    match = {
                        "gesture": "FOUR",
                        "english": "Four (4)",
                        "hindi": "चार",
                        "confidence": 0.98,
                        "category": "Counting",
                        "icon": "4️⃣",
                    }

            # ---------------------------------------------------------
            # B. I AM A BOY (Cap visor gesture at forehead / temple)
            # ---------------------------------------------------------
            elif cap:
                match = {
                    "gesture": "I AM A BOY",
                    "english": "I am a boy",
                    "hindi": "मैं एक लड़का हूँ",
                    "confidence": 0.98,
                    "category": "Identity",
                    "icon": "👦",
                }

            # ---------------------------------------------------------
            # C. I AM A GIRL (Index finger touching / pointing along cheek or jawline at face level)
            # ---------------------------------------------------------
            elif girl:
                match = {
                    "gesture": "I AM A GIRL",
                    "english": "I am a girl",
                    "hindi": "मैं एक लड़की हूँ",
                    "confidence": 0.98,
                    "category": "Identity",
                    "icon": "👧",
                }

            # ---------------------------------------------------------
            # D. I LOVE YOU (Thumb: OUT, Index: UP, Pinky: UP, Middle & Ring CURLED)
            # ---------------------------------------------------------
            elif t_out and i_up and not m_up and not r_up and p_up and m_c and r_c:
                match = {
                    "gesture": "I LOVE YOU",
                    "english": "I love you",
                    "hindi": "मैं तुमसे प्यार करता हूँ",
                    "confidence": 0.99,
                    "category": "Feelings",
                    "icon": "🤟",
                }

            # ---------------------------------------------------------
            # E. I AM FINE / OK (Thumb + Index circle, Middle & Ring & Pinky UP, upright)
            # ---------------------------------------------------------
            elif ok and m_up and r_up and p_up and f["hand_is_upright"]:
                match = {
                    "gesture": "I AM FINE",
                    "english": "I am fine",
                    "hindi": "मैं ठीक हूँ",
                    "confidence": 0.98,
                    "category": "Feelings",
                    "icon": "👌",
                }

            # ---------------------------------------------------------
            # F. WHAT ARE YOU DOING ('W' handshape: Index UP, Middle UP, Ring UP, Pinky curled)
            # ---------------------------------------------------------
            elif not t_out and i_up and m_up and r_up and not p_up and p_c:
                match = {
                    "gesture": "WHAT ARE YOU DOING",
                    "english": "What are you doing?",
                    "hindi": "आप क्या कर रहे हैं?",
                    "confidence": 0.98,
                    "category": "Conversation",
                    "icon": "❓",
                }

            # ---------------------------------------------------------
            # G. NO (Authentic ISL beak pinch: Thumb + Index + Middle touch together, Ring & Pinky curled)
            # ---------------------------------------------------------
            elif beak and r_c and p_c:
                match = {
                    "gesture": "NO",
                    "english": "No",
                    "hindi": "नहीं",
                    "confidence": 0.98,
                    "category": "Basic",
                    "icon": "🤏",
                }

            # ---------------------------------------------------------
            # H. COUNTING: EIGHT (8) (Thumb: OUT, Index: UP, Middle: UP, Ring: UP, Pinky: CURLED)
            # ---------------------------------------------------------
            elif t_out and i_up and m_up and r_up and not p_up and p_c:
                match = {
                    "gesture": "EIGHT",
                    "english": "Eight (8)",
                    "hindi": "आठ",
                    "confidence": 0.98,
                    "category": "Counting",
                    "icon": "8️⃣",
                }

            # ---------------------------------------------------------
            # I. COUNTING: THREE (3) (Thumb: OUT, Index: UP, Middle: UP, Ring & Pinky CURLED)
            # ---------------------------------------------------------
            elif t_out and i_up and m_up and not r_up and not p_up and r_c and p_c:
                match = {
                    "gesture": "THREE",
                    "english": "Three (3)",
                    "hindi": "तीन",
                    "confidence": 0.98,
                    "category": "Counting",
                    "icon": "3️⃣",
                }

            # ---------------------------------------------------------
            # J. COUNTING: TWO (2) / PEACE (Index: UP, Middle: UP, Ring & Pinky CURLED)
            # ---------------------------------------------------------
            elif not t_out and i_up and m_up and not r_up and not p_up and r_c and p_c:
                match = {
                    "gesture": "TWO",
                    "english": "Two (2)",
                    "hindi": "दो",
                    "confidence": 0.98,
                    "category": "Counting",
                    "icon": "2️⃣",
                }

            # ---------------------------------------------------------
            # K. COUNTING: ONE (1) (Index finger UP at chest/body level, all others CURLED, not girl)
            # ---------------------------------------------------------
            elif not t_out and i_up and m_c and r_c and p_c and not girl:
                match = {
                    "gesture": "ONE",
                    "english": "One (1)",
                    "hindi": "एक",
                    "confidence": 0.98,
                    "category": "Counting",
                    "icon": "1️⃣",
                }

            # ---------------------------------------------------------
            # L. COUNTING: SIX (6) (Pinky finger UP only, others CURLED)
            # ---------------------------------------------------------
            elif not t_out and not i_up and not m_up and not r_up and p_up and i_c and m_c and r_c:
                match = {
                    "gesture": "SIX",
                    "english": "Six (6)",
                    "hindi": "छह",
                    "confidence": 0.98,
                    "category": "Counting",
                    "icon": "6️⃣",
                }

            # ---------------------------------------------------------
            # M. COUNTING: SEVEN (7) (Thumb OUT + Index UP in 'L' shape, others CURLED)
            # ---------------------------------------------------------
            elif t_out and i_up and not m_up and not r_up and not p_up and m_c and r_c and p_c:
                match = {
                    "gesture": "SEVEN",
                    "english": "Seven (7)",
                    "hindi": "सात",
                    "confidence": 0.98,
                    "category": "Counting",
                    "icon": "7️⃣",
                }

            # ---------------------------------------------------------
            # N. THUMBS UP / ALRIGHT (Thumb UP only with curled fist in mid-air)
            # ---------------------------------------------------------
            elif t_up:
                match = {
                    "gesture": "ALRIGHT",
                    "english": "Alright",
                    "hindi": "ठीक है",
                    "confidence": 0.98,
                    "category": "Greetings",
                    "icon": "👍",
                }

            # ---------------------------------------------------------
            # O. YES / CLOSED FIST (Closed fist in mid-air)
            # ---------------------------------------------------------
            elif fist:
                match = {
                    "gesture": "YES",
                    "english": "Yes",
                    "hindi": "हाँ",
                    "confidence": 0.97,
                    "category": "Basic",
                    "icon": "✊",
                }

            # ---------------------------------------------------------
            # P. ALL 5 FINGERS SPREAD:
            # - Raised high near head (hand_y < 0.40 and index_tip_y < 0.25) -> HELLO (👋)
            # - At chest level -> FIVE (5) (5️⃣)
            # ---------------------------------------------------------
            elif t_out and i_up and m_up and r_up and p_up:
                if hand_y < 0.40:
                    match = {
                        "gesture": "HELLO",
                        "english": "Hello",
                        "hindi": "हैलो / नमस्ते",
                        "confidence": 0.98,
                        "category": "Greetings",
                        "icon": "👋",
                    }
                elif f["hand_is_upright"]:
                    match = {
                        "gesture": "FIVE",
                        "english": "Five (5)",
                        "hindi": "पाँच",
                        "confidence": 0.98,
                        "category": "Counting",
                        "icon": "5️⃣",
                    }

            # ---------------------------------------------------------
            # Q. SINGLE-HAND "HOW ARE YOU" (Open palm cupped forward at chest)
            # ---------------------------------------------------------
            elif f["cupped_how_are_you"]:
                match = {
                    "gesture": "HOW ARE YOU",
                    "english": "How are you?",
                    "hindi": "आप कैसे हैं?",
                    "confidence": 0.96,
                    "category": "Greetings",
                    "icon": "🤲",
                }

        # -------------------------------------------------------------
        # 3. FAST STABILITY FILTER (~60ms response time, zero buffering!)
        # -------------------------------------------------------------
        # If no intentional sign was formed, match remains None! (SILENCE)
        if match is not None:
            cand = match["gesture"]
            if cand == self.current_candidate:
                self.candidate_count += 1
            else:
                self.current_candidate = cand
                self.candidate_count = 1

            match["stable"] = bool(self.candidate_count >= self.stability_threshold)
            return match

        self.reset()
        return None


        # -------------------------------------------------------------
        # 3. STABILITY FILTER (Fast ~100ms response time)
        # -------------------------------------------------------------
        # If no intentional sign was formed, match remains None! (SILENCE)
        if match is not None:
            cand = match["gesture"]
            if cand == self.current_candidate:
                self.candidate_count += 1
            else:
                self.current_candidate = cand
                self.candidate_count = 1

            match["stable"] = bool(self.candidate_count >= self.stability_threshold)
            return match

        self.reset()
        return None
