"""Foreground hand selection and local ISL phrase decoding."""

from __future__ import annotations

from collections import deque
from time import monotonic

import numpy as np

from isl_sentence_synthesizer import ContinuousSentenceBuilder
import isl_vocabulary_db as db

ISL_LESSON = "https://www.youtube.com/watch?v=5vHmvYA8Z6Q"
INCLUDE_VIDEOS = "https://huggingface.co/datasets/ai4bharat/INCLUDE"
GREETINGS = {
    "HELLO": ("Hello", "48. Hello"),
    "HOW ARE YOU": ("How are you?", "49. How are you"),
    "ALRIGHT": ("Alright", "50. Alright"),
    "GOOD MORNING": ("Good morning.", "51. Good Morning"),
    "GOOD AFTERNOON": ("Good afternoon.", "52. Good afternoon"),
    "GOOD EVENING": ("Good evening.", "53. Good evening"),
    "GOOD NIGHT": ("Good night.", "54. Good night"),
    "THANK YOU": ("Thank you.", "55. Thank you"),
    "PLEASED": ("Pleased to meet you.", "56. Pleased"),
}

GREETING_INSTRUCTIONS = {
    "HELLO": "Raise one open palm beside your ear with all five fingers spread. A short salute from the temple also works.",
    "HOW ARE YOU": "Raise one or both open, slightly cupped palms in front of your chest. Face them up or forward. You can also point toward the person and then raise both thumbs.",
    "ALRIGHT": "Curl four fingers into a fist and point your thumb up. Hold the hand in front of your chest.",
    "GOOD MORNING": "Lift your hand from low beside you and spread your fingers beside your head, palm forward.",
    "GOOD AFTERNOON": "Follow the afternoon movement in the video, bringing the sun handshape up over your head.",
    "GOOD EVENING": "Keep the sun handshape lower and to the side as in the video. Copy the whole movement.",
    "GOOD NIGHT": "Lift one hand from low in front of you to beside your face. Bend the fingers and turn your palm toward your face.",
    "THANK YOU": "Bring the fingertips of a flat hand to your chin, then move the hand out and slightly down.",
    "PLEASED": "Lift one open hand with your fingers extended. Turn the hand as it comes up, following the Pleased clip for the exact motion.",
}

LANDMARK_PHRASES = {
    "NAMASTE": ("Namaste", "Press both open palms together at chest center, fingers up. With one hand, show its edge to the camera at the same position.", "https://usesignify.org/signs/namaste"),
    "I AM FINE": ("I am fine", "Touch thumb tip to index tip in a circle. Hold your middle, ring and little fingers straight.", None),
    "WHAT ARE YOU DOING": ("What are you doing?", "Raise index, middle and ring fingers in a W with thumb and little finger tucked. Or hold both index fingers upright and apart.", None),
    "I AM A BOY": ("I am a boy", "Hold a flat hand sideways across the forehead, like a cap visor. Keep your fingers pointed across your forehead.", None),
    "I AM A GIRL": ("I am a girl", "Curl the other fingers and point your index finger along your cheek or toward your earlobe.", None),
    "HELP ME": ("Help me", "Place a fist or thumbs-up on the open palm of your other hand at chest height. Hold the stack or lift both hands together.", "https://usesignify.org/signs/help"),
    "I LOVE YOU": ("I love you", "Extend the thumb, index and little fingers. Fold the middle and ring fingers. You can also point to your chest, cross your wrists, then point outward.", None),
    "YES": ("Yes", "Hold a closed upright fist in front of your chest with your thumb tucked against the fingers.", None),
    "NO": ("No", "Bring your thumb, index and middle fingertips together in a forward pinch. Keep the ring and little fingers folded.", None),
    "BYE": ("Bye!", "Raise an open palm beside your head and wave side to side twice.", "https://www.youtube.com/watch?v=O6YJgD20e2g"),
    "SORRY": ("Sorry", "Pinch each earlobe lightly between your thumb and index finger, then release.", "https://usesignify.org/signs/sorry"),
    "WELCOME": ("Welcome", "Hold two open palms up in front of your chest. Sweep them inward together toward your body.", "https://usesignify.org/signs/welcome"),
    "WATER": ("Water", "Form a W-handshape with index, middle, and ring fingers upright near chin.", None),
    "FOOD": ("Food", "Bring bunched fingertips to touch lips twice as if eating.", None),
    "HUNGRY": ("Hungry", "Draw curved hand downward from throat to stomach.", None),
    "DOCTOR": ("Doctor", "Tap index and middle fingertips against underside of opposite wrist (pulse).", None),
    "TIME": ("Time", "Tap index finger on opposite wrist where watch is worn.", None),
}

HINDI_TRANSLATIONS = {
    "NAMASTE": "नमस्ते", "HOW ARE YOU": "आप कैसे हैं?", "I AM FINE": "मैं ठीक हूँ",
    "WHAT ARE YOU DOING": "आप क्या कर रहे हैं?", "I AM A BOY": "मैं एक लड़का हूँ",
    "I AM A GIRL": "मैं एक लड़की हूँ", "HELP ME": "मेरी मदद करो",
    "I LOVE YOU": "मैं तुमसे प्यार करता हूँ", "HELLO": "हैलो / नमस्ते", "YES": "हाँ",
    "NO": "नहीं", "ALRIGHT": "ठीक है", "BYE": "अलविदा", "SORRY": "माफ़ कीजिए",
    "WELCOME": "स्वागत है", "GOOD MORNING": "शुभ प्रभात", "GOOD AFTERNOON": "शुभ दोपहर",
    "GOOD EVENING": "शुभ संध्या", "GOOD NIGHT": "शुभ रात्रि", "THANK YOU": "धन्यवाद",
    "PLEASED": "आपसे मिलकर अच्छा लगा", "WATER": "पानी", "FOOD": "खाना",
    "HUNGRY": "भूख लगी है", "DOCTOR": "डॉक्टर", "TIME": "समय",
    "ONE": "एक", "TWO": "दो", "THREE": "तीन", "FOUR": "चार", "FIVE": "पाँच",
    "SIX": "छह", "SEVEN": "सात", "EIGHT": "आठ", "TEN": "दस",
}

# Finger order: thumb, index, middle, ring, little.
COUNTS = {
    (False, True, False, False, False): (1, "At chest height, raise the index finger and curl the other fingers."),
    (False, True, True, False, False): (2, "Raise the index and middle fingers."),
    (True, True, True, False, False): (3, "Extend the thumb, index and middle fingers."),
    (False, True, True, True, True): (4, "Raise four fingers and tuck the thumb."),
    (True, True, True, True, True): (5, "Spread all five fingers at chest height."),
    (False, False, False, False, True): (6, "Raise the little finger."),
    (True, True, False, False, False): (7, "Extend the thumb and index finger in an L."),
    (True, True, True, True, False): (8, "Extend the thumb and first three fingers."),
}
COUNT_NAMES = {1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE",
               6: "SIX", 7: "SEVEN", 8: "EIGHT", 10: "TEN"}
SIGN_VIDEOS = {
    "HELLO": ISL_LESSON,
    "HOW ARE YOU": "https://www.youtube.com/watch?v=3fS1QTOaUyk",
    "GOOD MORNING": "https://www.youtube.com/watch?v=L37McuHckSI",
    "GOOD AFTERNOON": "https://www.youtube.com/watch?v=swQX2Ca4Otg",
    "GOOD EVENING": "https://www.youtube.com/watch?v=s85rsXqYSqU",
    "GOOD NIGHT": "https://usesignify.org/signs/goodnight",
    "THANK YOU": "https://www.youtube.com/watch?v=wK8amEJAK1Y",
}
RELEASE_SECONDS = 0.18
MAX_SIGN_SECONDS = 5.0


def gesture_guide():
    entries = [
        {"gesture": label, "english": english, "mode": "conversation",
         "instruction": GREETING_INSTRUCTIONS[label], "example": f"Greetings / {folder}",
         "video": SIGN_VIDEOS.get(label)}
        for label, (english, folder) in GREETINGS.items()
    ] + [
        {"gesture": label, "english": english, "mode": "conversation",
         "instruction": instruction, "video": video}
        for label, (english, instruction, video) in LANDMARK_PHRASES.items()
    ] + [
        {"gesture": COUNT_NAMES[number], "english": _count_translation(number), "mode": "counting",
         "instruction": instruction}
        for number, instruction in [
            *COUNTS.values(), (10, "At chest height, hold up two open hands with all fingers spread."),
        ]
    ]
    for entry in entries:
        entry["hindi"] = HINDI_TRANSLATIONS[entry["gesture"]]
    return entries


def _count_translation(number):
    return f"{COUNT_NAMES[number].title()} ({number})"


def _palm_size(hand):
    # Use length and width so an edge-on or palm-up sign can remain in view.
    return float(max(np.linalg.norm(hand[9, :2] - hand[0, :2]),
                     np.linalg.norm(hand[5, :2] - hand[17, :2])))


class ForegroundHands:
    """Pick at most one hand per MediaPipe label in the foreground signing area."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._previous = {}
        self._scale = 0.0
        self._last_seen = None

    def select(self, detections, now=None):
        now = monotonic() if now is None else now
        if self._last_seen is not None and now - self._last_seen > 0.75:
            self._previous.clear()

        candidates = []
        for label, hand in detections:
            size = _palm_size(hand)
            x, y = np.mean(hand[[0, 5, 9, 13, 17], :2], axis=0)
            if label in ("left", "right") and size >= 0.055 and 0.12 <= x <= 0.88 and 0.06 <= y <= 0.80:
                candidates.append((size, label, hand))

        largest = max((size for size, _, _ in candidates), default=0.0)
        minimum = max(0.66 * largest, 0.62 * self._scale)
        active = {}
        for size, label, hand in sorted(candidates, key=lambda item: item[0], reverse=True):
            if size < minimum or label in active:
                continue
            previous = self._previous.get(label)
            if previous is not None and np.linalg.norm(previous[0, :2] - hand[0, :2]) > 0.25:
                continue
            active[label] = hand

        if active:
            self._previous = active
            self._last_seen = now
            scale = max(_palm_size(hand) for hand in active.values())
            self._scale = max(scale, 0.94 * self._scale)
        return active.get("left"), active.get("right")


class SigningFace:
    """Find the foreground signer's face periodically for relative gesture positions."""

    def __init__(self):
        import cv2

        self._cv2 = cv2
        self._detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.reset()

    def reset(self):
        self._last_check = -float("inf")
        self._last_found = -float("inf")
        self._face = None

    def locate(self, frame):
        now = monotonic()
        if now - self._last_check >= 0.30 and not self._detector.empty():
            self._last_check = now
            cv2 = self._cv2
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (160, max(1, round(160 * h / w))), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            largest = max(
                (face for face in self._detector.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=4, minSize=(22, 22)
                ) if 0.18 * gray.shape[1] < face[0] + face[2] / 2 < 0.82 * gray.shape[1]),
                key=lambda face: face[2] * face[3], default=None,
            )
            if largest is not None:
                x, y, width, height = largest
                fresh = np.array([(x + width / 2) / gray.shape[1], (y + height / 2) / gray.shape[0],
                                  width / gray.shape[1], height / gray.shape[0]])
                self._face = fresh if self._face is None else 0.7 * fresh + 0.3 * self._face
                self._last_found = now
        if now - self._last_found > 1.2:
            self._face = None
        return self._face


def _finger_states(points, size):
    bases = np.array([5, 9, 13, 17])
    reach = points[bases + 3] - points[bases]
    first_bones = points[bases + 1] - points[bases]
    tips = points[bases + 3] - points[bases + 1]
    alignment = np.sum(first_bones * tips, axis=1) / np.maximum(
        np.linalg.norm(first_bones, axis=1) * np.linalg.norm(tips, axis=1), 1e-6
    )
    straight = (alignment > 0.65) & (np.linalg.norm(reach, axis=1) > 0.36 * size)
    folded = (alignment < 0.35) | (np.linalg.norm(reach, axis=1) < 0.40 * size)
    return straight, folded, reach


def _finger_pattern(hand):
    # MediaPipe z uses the x scale. Camera input is 4:3.
    points = hand * np.array([1.0, 0.75, 1.0], dtype=np.float32)
    size = np.linalg.norm(points[9] - points[0])
    if size < 1e-5 or points[0, 1] - points[9, 1] < 0.30 * size:
        return None
    straight, folded, reach = _finger_states(points, size)
    extended = straight & (reach[:, 1] < -0.23 * size)
    if not np.all(extended | folded):
        return None
    thumb = np.linalg.norm(points[4] - points[5]) / size
    if 0.42 <= thumb <= 0.60:
        return None
    return (bool(thumb > 0.60), *extended.tolist())


def _describe_hand(hand):
    points = hand * np.array([1.0, 0.75, 1.0], dtype=np.float32)
    axis = points[9] - points[0]
    size = max(float(np.linalg.norm(axis)), 1e-5)
    straight, folded, reach = _finger_states(points, size)
    return points, axis, size, straight, folded, reach


def _thumbs_up(shape):
    points, _, size, straight, folded, _ = shape
    return (np.all(folded & ~straight)
            and points[3, 1] - points[4, 1] > 0.23 * size
            and points[4, 1] < points[9, 1] - 0.12 * size)


def _observe_gestures(hands, face):
    # Face-relative positions separate a greeting near the ear from a count at the chest.
    cx, cy, width, height = face if face is not None else (0.50, 0.29, 0.22, 0.24)
    shapes = [_describe_hand(hand) for hand in hands]

    def chest(point):
        return (cy + 0.67 * height < point[1] / 0.75 < cy + 2.40 * height
                and abs(point[0] - cx) < 1.65 * width)

    def ear(point):
        return (cy - 0.20 * height < point[1] / 0.75 < cy + 0.67 * height
                and 0.32 * width < abs(point[0] - cx) < 1.15 * width)

    def upright(shape):
        return shape[1][1] < -0.52 * shape[2]

    def open_upright(shape):
        return np.all(shape[3]) and upright(shape)

    def near_tips(shape):
        points, _, size, *_ = shape
        return np.linalg.norm(points[4] - points[8]) < 0.27 * size

    def cupped(shape):
        pts, axis, size, straight, folded, reach = shape
        # Palm pitched forward or horizontal, with open fingers or slightly flexed tips.
        normal = np.cross(pts[5] - pts[0], pts[17] - pts[0])
        horizontal = abs(normal[1]) > 0.62 * np.linalg.norm(normal)
        fingers_open = np.count_nonzero(straight) >= 3 or (
            not np.any(folded[:3]) and np.count_nonzero(np.linalg.norm(reach, axis=1) > 0.55 * size) >= 3
        )
        return chest(pts[0]) and fingers_open and not near_tips(shape) and (
            axis[2] < -0.30 * size or (horizontal and axis[1] < 0.20 * size)
        )

    def index_only(shape):
        straight, folded = shape[3:5]
        return straight[0] and np.all(folded[1:] & ~straight[1:])

    if len(shapes) == 1:
        shape = shapes[0]
        pts, axis, size, straight, folded, reach = shape
        pinched = near_tips(shape)
        if upright(shape) and pinched and not straight[0] and np.all(straight[1:]):
            return "I AM FINE", None
        if (pinched and np.all(folded[2:]) and not np.any(straight[2:])
                and np.linalg.norm(pts[4] - pts[9]) > 0.44 * size
                and np.linalg.norm(pts[4] - pts[12]) < 0.30 * size):
            return "NO", None

        if index_only(shape):
            if (ear(pts[8]) and abs(pts[8, 0] - cx) < 0.76 * width
                    and pts[0, 1] / 0.75 < cy + height
                    and (abs(reach[0, 1]) > 0.22 * size or abs(reach[0, 0]) > 0.34 * size)):
                return "I AM A GIRL", None
            toward_center = abs(pts[8, 0] - cx) < abs(pts[5, 0] - cx) - 0.28 * size
            toward_camera = reach[0, 2] < -0.55 * size and np.linalg.norm(reach[0, :2]) < 0.70 * size
            if toward_camera:
                return "YOU", None
            if toward_center and abs(reach[0, 0]) > abs(reach[0, 1]) and chest(pts[0]):
                return "SELF", None

        if upright(shape):
            if tuple(straight) == (True, False, False, True) and np.all(folded[1:3]):
                if np.linalg.norm(pts[4] - pts[5]) > 0.64 * size and not pinched:
                    return "I LOVE YOU", None
            if chest(pts[0]) and tuple(straight) == (True, True, True, False) and folded[3]:
                if np.linalg.norm(pts[4] - pts[5]) < 0.60 * size:
                    if pts[9, 1] / 0.75 < cy + 1.15 * height and abs(pts[9, 0] - cx) < 1.10 * width:
                        return "WATER", None
                    return "WHAT ARE YOU DOING", None
            if chest(pts[0]) and np.all(folded & ~straight):
                if _thumbs_up(shape):
                    return "ALRIGHT", None
                if np.linalg.norm(pts[4] - pts[5]) < 0.76 * size:
                    return "YES", None

        if pinched and np.all(folded[1:]) and abs(pts[4, 1] / 0.75 - (cy + 0.35 * height)) < 0.40 * height:
            return "FOOD", None

        if (chest(pts[0]) and np.count_nonzero(folded) >= 2 and np.count_nonzero(straight) <= 2
                and abs(pts[0, 0] - cx) < 0.75 * width and pts[0, 1] / 0.75 > cy + 0.85 * height):
            return "HUNGRY", None

        if open_upright(shape):
            if (chest(pts[0]) and abs(pts[9, 0] - cx) < 0.36 * width
                    and np.linalg.norm((pts[5] - pts[17])[:2]) < 0.46 * size
                    and abs(axis[0]) < 0.35 * size):
                return "NAMASTE", None
            beside_head = (cy - 0.37 * height <= pts[9, 1] / 0.75 <= cy + 0.65 * height
                           and pts[0, 1] / 0.75 < cy + 1.02 * height
                           and 0.53 * width < abs(pts[9, 0] - cx) < 1.78 * width)
            if not beside_head and cupped(shape):
                return "HOW ARE YOU", None
            thumb_open = np.linalg.norm(pts[4] - pts[5]) > 0.60 * size
            return "WAVE", (float(axis[0] / size), float(pts[0, 0]), size, beside_head and thumb_open)
        if (np.count_nonzero(straight) >= 3 and abs(axis[0]) > 0.62 * size
                and abs(pts[9, 1] / 0.75 - (cy - 0.31 * height)) < 0.30 * height
                and abs(pts[9, 0] - cx) < 0.70 * width):
            return "I AM A BOY", None
        if cupped(shape):
            return "HOW ARE YOU", None

    elif len(shapes) == 2:
        if all(_thumbs_up(shape) for shape in shapes):
            return "TWO THUMBS", None
        (p, ax, size, straight, folded, _), (q, bx, other_size, other_straight, other_folded, _) = sorted(
            shapes, key=lambda shape: shape[0][9, 0]
        )
        scale = max(size, other_size)
        spread = q[9, 0] - p[9, 0]
        if (p[8, 0] < cx < q[8, 0] and all(near_tips(shape) for shape in shapes)
                and ear(p[8]) and ear(q[8]) and abs(p[8, 1] - q[8, 1]) < 0.35 * height):
            return "SORRY", None
        if (all(open_upright(shape) for shape in shapes)
                and chest((p[0] + q[0]) / 2) and abs((p[9, 0] + q[9, 0]) / 2 - cx) < 0.42 * width
                and np.linalg.norm(p[9, :2] - q[9, :2]) < 0.54 * scale
                and np.linalg.norm(p[12, :2] - q[12, :2]) < 0.60 * scale):
            return "NAMASTE", None
        if (all(index_only(shape) and upright(shape) for shape in shapes)
                and all(chest(shape[0][0]) for shape in shapes)
                and spread > 0.85 * scale):
            return "WHAT ARE YOU DOING", None

        # Pulse check for DOCTOR or tap wrist for TIME
        if (np.linalg.norm(p[8, :2] - q[0, :2]) < 0.40 * scale or np.linalg.norm(q[8, :2] - p[0, :2]) < 0.40 * scale):
            if any(index_only(shape) for shape in shapes):
                return "TIME", None
            return "DOCTOR", None

        for fist, support in (shapes, shapes[::-1]):
            pts, fist_axis, fist_size, fingers, bent, _ = fist
            base, base_axis, base_size, open_fingers, _, _ = support
            stacked = (base[9, 1] - pts[9, 1] > 0.13 * fist_size
                       and np.linalg.norm(base[9, :2] - pts[9, :2]) < 1.30 * max(fist_size, base_size))
            if (chest(base[0]) and np.all(bent & ~fingers) and fist_axis[1] < -0.10 * fist_size
                    and stacked and np.all(open_fingers) and abs(base_axis[1]) < 0.60 * base_size):
                return "HELP ME", None

        paired = (0.5 * scale < spread < 3.3 * scale
                  and abs(p[0, 1] - q[0, 1]) < scale
                  and np.linalg.norm(p[0, :2] - q[0, :2]) < 1.65 * scale)
        fingers_match = ((np.all(straight) and np.all(other_straight))
                         or (np.all(folded) and np.all(other_folded)))
        if paired and fingers_match and ax[0] < -0.37 * size and bx[0] > 0.37 * other_size:
            return "CROSS", None
        if all(cupped(shape) for shape in shapes):
            return "PALMS UP", (float(spread), float((p[9, 1] + q[9, 1]) / 2), scale)
        if spread > 0.85 * scale and all(
            open_upright(shape) and chest(shape[0][0]) and shape[0][9, 1] / 0.75 > cy + 0.35 * height
            for shape in shapes
        ):
            return "HOW ARE YOU", None
    return None, None


def _stable_runs(observations):
    runs = []
    for now, label, value in observations:
        if (not runs or runs[-1][-1][1] != label
                or now - runs[-1][-1][0] > 0.22):
            runs.append([])
        runs[-1].append((now, label, value))
    return [run for run in runs if len(run) >= 3 and run[-1][0] - run[0][0] >= 0.08]


def _wave_reversals(values, minimum):
    anchor = values[0]
    changes = []
    for value in values[1:]:
        delta = value - anchor
        if abs(delta) < minimum:
            continue
        direction = 1 if delta > 0 else -1
        if not changes or changes[-1] != direction:
            changes.append(direction)
        anchor = value
    return len(changes)


def _match_landmark_phrase(observations):
    runs = _stable_runs(observations)
    labels = [run[0][1] for run in runs]
    stage = 0
    sequence = ("SELF", "CROSS", "YOU")
    for label in labels:
        if label == sequence[stage]:
            stage += 1
            if stage == len(sequence):
                return "I LOVE YOU"

    if "YOU" in labels and "TWO THUMBS" in labels[labels.index("YOU") + 1:]:
        return "HOW ARE YOU"

    candidates = []
    static_labels = LANDMARK_PHRASES.keys() | {"HELLO", "ALRIGHT", "HOW ARE YOU"}
    span = observations[-1][0] - observations[0][0] if observations else 0
    for run in runs:
        label = run[0][1]
        elapsed = run[-1][0] - run[0][0]
        if label == "WAVE":
            tilts, wrists, sizes, beside_head = zip(*(sample[2] for sample in run))
            if elapsed >= 0.42 and (_wave_reversals(tilts, 0.28) >= 3
                                    or _wave_reversals(wrists, 0.38 * np.median(sizes)) >= 3):
                return "BYE"
            if sum(beside_head) / len(run) >= 0.70:
                label = "HELLO"
        if label == "PALMS UP":
            spreads, heights, sizes = (np.asarray(part) for part in zip(*(sample[2] for sample in run)))
            scale = np.median(sizes)
            if elapsed >= 0.42 and np.any(
                (spreads[0] - spreads > 0.65 * scale) & (heights - heights[0] > 0.30 * scale)
            ):
                return "WELCOME"
            label = "HOW ARE YOU"
        if label == "TWO THUMBS":
            label = "ALRIGHT"
        if label in static_labels and elapsed >= max(0.28, 0.35 * span):
            candidates.append((elapsed, label))

    candidates.sort(reverse=True)
    if candidates and not any(
        label != candidates[0][1] and elapsed > 0.80 * candidates[0][0]
        for elapsed, label in candidates[1:]
    ):
        return candidates[0][1]
    return None


class PrecisionISLRecognizer:
    """Decode real-time ISL signs into connected conversational sentences or numeric poses."""

    def __init__(self, mode="conversation", threshold=0.75, predictor=None, cadence_timeout=1.8):
        self.mode = mode
        self.threshold = threshold
        self._predictor = predictor
        self.sentence_builder = ContinuousSentenceBuilder(cadence_timeout=cadence_timeout, cooldown_timeout=0.6)
        self.reset()

    def reset(self):
        # Clear raw frame queue and sentence buffer completely
        self._frames = deque(maxlen=300)
        self._observations = deque(maxlen=300)
        self._last_seen = None
        self._started = None
        self._discard = False
        self._candidate = None
        self._candidate_since = None
        self._emitted_count = None
        self._neutral_since = None
        self.sentence_builder.reset()
        self.status = "Show your next sign"

    def _clear_active_sign(self):
        """Clear single sign frame buffers while preserving the active connected sentence."""
        self._frames.clear()
        self._observations.clear()
        self._last_seen = None
        self._started = None
        self._discard = False

    def load_model(self):
        if self._predictor is None:
            from inference.gesture_predictor import GesturePredictor
            self._predictor = GesturePredictor()

    def get_live_sentence_preview(self) -> dict:
        return self.sentence_builder.get_preview()

    def finalize_current_sentence(self, now=None) -> dict | None:
        return self.sentence_builder.finalize_sentence(now=now)

    def process(self, features, hands, now=None, face=None):
        now = monotonic() if now is None else now
        if self.mode == "counting":
            return self._count(hands, now)

        # 1. Check if hands are down/resting and inter-sign pause completed a sentence
        if not hands or (self._last_seen is not None and now - self._last_seen >= 0.25):
            completed = self.sentence_builder.check_pause(now=now)
            if completed:
                self.status = f"Spoken: {completed['sentence']}"
                return {
                    "gesture": completed["tokens"][-1] if completed["tokens"] else "PAUSE",
                    "english": completed["sentence"],
                    "hindi": completed["hindi"],
                    "confidence": 1.0,
                    "category": completed["category"],
                    "is_sentence_complete": True,
                    "sentence": completed["sentence"],
                    "hindi_sentence": completed["hindi"],
                    "tokens": completed["tokens"],
                    "live_preview": self.sentence_builder.get_preview(),
                }

        # 2. Accumulate signing frames while hands are in view
        if hands:
            self._started = now if self._started is None else self._started
            self._last_seen = now
            preview = self.sentence_builder.get_preview()
            if preview["active"]:
                self.status = f"Drafting: {preview['sentence']}"
            else:
                self.status = "Finish sign and lower hands briefly"
            if now - self._started > MAX_SIGN_SECONDS:
                self._discard = True
                self._frames.clear()
                self._observations.clear()
                self.status = "Lower your hands and try one sign at a time"
            if not self._discard:
                self._frames.append(features)
                self._observations.append((now, *_observe_gestures(hands, face)))
            return None

        if self._last_seen is None or now - self._last_seen < RELEASE_SECONDS:
            return None

        # 3. Decode completed sign upon release
        frames = list(self._frames)
        observations = list(self._observations)
        duration = self._last_seen - self._started
        self._clear_active_sign()
        if duration < 0.35:
            return None

        label = _match_landmark_phrase(observations)
        if not label and len(frames) >= 6:
            match = self.decode_phrase(np.asarray(frames, dtype=np.float32))
            if match:
                label = match["gesture"]

        if label:
            english = (LANDMARK_PHRASES.get(label) or GREETINGS.get(label, (label.title(), None)))[0]
            completed = self.sentence_builder.add_sign(label, now=now)
            preview = self.sentence_builder.get_preview()
            hindi = HINDI_TRANSLATIONS.get(label, "")
            cat = LANDMARK_PHRASES.get(label, ("General", None, None))[2] or "Conversation"
            return {
                "gesture": label,
                "english": english,
                "hindi": hindi,
                "confidence": 0.95,
                "category": cat,
                "is_sentence_complete": bool(completed),
                "sentence": completed["sentence"] if completed else None,
                "hindi_sentence": completed["hindi"] if completed else None,
                "tokens": completed["tokens"] if completed else preview["tokens"],
                "live_preview": preview,
            }

        self.status = "Sign unclear. Try again."
        return None

    def decode_phrase(self, frames):
        from dataset.extract_landmarks import interpolate_short_dropouts, resample_sequence

        # The model was trained on isolated clips. Reject static poses before classification.
        moving = False
        for flag, start in ((126, 0), (127, 63)):
            hand = frames[frames[:, flag] > 0.5, start:start + 63]
            if len(hand) >= 4:
                spread = np.percentile(hand, 90, axis=0) - np.percentile(hand, 10, axis=0)
                moving |= np.linalg.norm(spread) / np.sqrt(21) >= 0.12
        if not moving:
            return None

        self.load_model()
        seq = resample_sequence(interpolate_short_dropouts(frames), self._predictor.sequence_length)
        # Average slightly shifted views so a few frames of raising or lowering
        # the hands do not determine the output by themselves.
        edge = max(1, self._predictor.sequence_length // 10)
        views = (seq, resample_sequence(seq[edge:]), resample_sequence(seq[:-edge]))
        views = [self._predictor.predict_sequence(view) for view in views]
        scores = np.mean(views, axis=0)
        first, second = np.argsort(scores)[-2:][::-1]
        label = self._predictor.class_names[first]
        confidence = float(scores[first])
        votes = sum(int(np.argmax(view)) == first for view in views)
        if (label not in GREETINGS or confidence < self.threshold
                or confidence - scores[second] < 0.20 or votes < 2):
            return None
        return {"gesture": label, "english": GREETINGS[label][0],
                "confidence": confidence, "category": "Conversation"}

    def _count(self, hands, now):
        patterns = [_finger_pattern(hand) for hand in hands]
        number = None
        if len(patterns) == 1 and patterns[0] in COUNTS:
            number = COUNTS[patterns[0]][0]
        elif len(patterns) == 2 and all(pattern == (True,) * 5 for pattern in patterns):
            number = 10

        if number is None:
            self._candidate = None
            self._neutral_since = now if self._neutral_since is None else self._neutral_since
            if now - self._neutral_since >= RELEASE_SECONDS:
                self._emitted_count = None
            return None
        self._neutral_since = None
        if number != self._candidate:
            self._candidate = number
            self._candidate_since = now
            return None
        if number == self._emitted_count or now - self._candidate_since < 0.22:
            return None
        self._emitted_count = number
        return {"gesture": COUNT_NAMES[number], "english": _count_translation(number),
                "confidence": None, "category": "Counting"}
