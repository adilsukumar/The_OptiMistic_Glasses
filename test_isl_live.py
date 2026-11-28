"""Regression checks for the live translator's recognition and speech boundaries."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

import server
from isl_recognition_engine import COUNTS, ForegroundHands, PrecisionISLRecognizer, gesture_guide


def open_hand(x=0.5, size=1.0):
    coords = np.zeros((21, 3), dtype=np.float32)
    coords[0] = [x, 0.55, 0]
    coords[4] = [x - 0.1 * size, 0.55 - 0.1 * size, 0]
    for base, offset in zip((5, 9, 13, 17), (-0.04, 0, 0.025, 0.05)):
        for joint, height in enumerate((0.12, 0.165, 0.20, 0.23)):
            coords[base + joint] = [x + offset * size, 0.55 - height * size, 0]
    return coords


def test_foreground_keeps_one_per_label_and_does_not_pick_up_background_on_release():
    selector = ForegroundHands()
    left, right = open_hand(0.32), open_hand(0.65)
    duplicate, background = open_hand(0.55, 0.85), open_hand(0.71, 0.48)
    picked_left, picked_right = selector.select([
        ("right", background), ("left", duplicate), ("right", right), ("left", left),
    ], now=0.0)
    assert picked_left is left
    assert picked_right is right
    assert selector.select([("right", background)], now=1.0) == (None, None)


def test_multi_person_scene_selects_only_front_person_one_hand():
    selector = ForegroundHands()
    # Front person is nearest (size 1.2), signing with single right hand at chest
    front_person_hand = open_hand(0.52, size=1.2)
    # Background 3rd persons with multiple hands visible
    bg_person1_l = open_hand(0.20, size=0.60)
    bg_person1_r = open_hand(0.30, size=0.55)
    bg_person2_l = open_hand(0.80, size=0.50)
    bg_person2_r = open_hand(0.88, size=0.48)

    detections = [
        ("left", bg_person1_l),
        ("right", bg_person1_r),
        ("right", front_person_hand),
        ("left", bg_person2_l),
        ("right", bg_person2_r),
    ]

    picked_left, picked_right = selector.select(detections, now=0.0)
    assert picked_left is None, "Should not attach any background 3rd person hand to the left slot"
    assert picked_right is front_person_hand, "Should strictly select only the nearest person's hand"


def test_multi_person_scene_selects_only_front_person_two_hands():
    selector = ForegroundHands()
    # Front person signing with two hands (size 1.2 and 1.15)
    front_r = open_hand(0.42, size=1.2)
    front_l = open_hand(0.58, size=1.15)
    # Background hands from 3rd persons
    bg_l = open_hand(0.20, size=0.65)
    bg_r = open_hand(0.85, size=0.60)

    detections = [
        ("left", bg_l),
        ("right", front_r),
        ("left", front_l),
        ("right", bg_r),
    ]

    picked_left, picked_right = selector.select(detections, now=0.0)
    assert picked_left is front_l, "Should select front person left hand"
    assert picked_right is front_r, "Should select front person right hand"


def test_side_by_side_rejects_further_person():
    selector = ForegroundHands()
    # Person A is in front (size 1.25)
    person_a = open_hand(0.60, size=1.25)
    # Person B is standing to the side but further away (size 0.70)
    person_b = open_hand(0.25, size=0.70)

    detections = [
        ("left", person_b),
        ("right", person_a),
    ]

    picked_left, picked_right = selector.select(detections, now=0.0)
    assert picked_left is None, "Further person B must be discarded"
    assert picked_right is person_a, "Nearest person A must be selected"


def test_conversation_decodes_once_after_release_and_rejects_static_counting_pose():
    recognizer = PrecisionISLRecognizer()
    tracked_hand = open_hand()
    features = np.zeros(128, dtype=np.float32)
    features[127] = 1
    for i in range(12):
        assert recognizer.process(features, [tracked_hand], now=i / 20) is None
    assert recognizer.process(features, [], now=0.60) is None
    assert recognizer.process(features, [], now=0.75) is None
    assert recognizer.status == "Sign unclear. Try again."

    result = {"gesture": "HELLO", "english": "Hello!", "category": "Conversation"}
    with patch.object(recognizer, "decode_phrase", return_value=result) as decode:
        for i in range(12):
            assert recognizer.process(features, [tracked_hand], now=1 + i / 20) is None
        assert recognizer.process(features, [], now=1.62) is None
        decode.assert_not_called()
        assert recognizer.process(features, [], now=1.76) == result
        assert recognizer.process(features, [], now=2.00) is None
        decode.assert_called_once()


def test_counting_stays_separate_and_a_held_pose_is_emitted_once():
    recognizer = PrecisionISLRecognizer(mode="counting")
    hands = [open_hand(0.32), open_hand(0.66)]
    with patch.object(recognizer, "decode_phrase") as decode:
        assert recognizer.process(None, hands, now=0.0) is None
        assert recognizer.process(None, hands, now=0.1) is None
        assert recognizer.process(None, hands, now=0.24)["english"] == "Ten (10)"
        assert recognizer.process(None, hands, now=0.5) is None
        decode.assert_not_called()


def test_manual_speech_only_uses_completed_conversation_events():
    controller = server.ISLController()
    controller._events.extend([
        {"id": 1, "category": "Conversation", "english": "Hello!"},
        {"id": 2, "category": "Counting", "english": "5"},
        {"id": 3, "category": "Conversation", "english": "How are you?"},
    ])
    speaker = Mock()
    with patch.object(server, "_SPEAKER", speaker):
        assert controller.speak_transcript([1, 3])
        assert not controller.speak_transcript([2])
        assert not controller.speak_transcript(["Hello!"])
        controller.set_mode("counting")
        assert not controller.speak_transcript([1])
        speaker.speak.assert_called_once_with("Hello! How are you?")

def _shape(x=0.5, y=0.55, straight=(True,) * 4, rotation=0):
    hand = open_hand(x)
    for base, extended in zip((5, 9, 13, 17), straight):
        if not extended:
            hand[base + 1:base + 4] = hand[base] + np.array([
                [0.0, -0.035, 0.0], [0.01, -0.04, -0.04], [0.01, 0.015, -0.03],
            ])
    sin, cos = np.sin(rotation), np.cos(rotation)
    pivot = np.array([x, 0.55])
    hand[:, :2] = ((hand[:, :2] - pivot) @ np.array([[cos, sin], [-sin, cos]])
                   + np.array([x, y]))
    return hand


def _finish(recognizer, poses):
    features = np.zeros(128, dtype=np.float32)
    for i, hands in enumerate(poses):
        assert recognizer.process(features, hands, now=i * 0.06) is None
    end = len(poses) * 0.06
    assert recognizer.process(features, [], now=end) is None
    result = recognizer.process(features, [], now=end + 0.18)
    assert recognizer.process(features, [], now=end + 0.35) is None
    return result


def test_i_am_fine_requires_a_closed_index_thumb_circle_and_three_straight_fingers():
    hand = _shape(straight=(False, True, True, True))
    hand[4] = hand[8]
    assert _finish(PrecisionISLRecognizer(), [[hand]] * 8)["english"] == "I am fine"
    hand[4, 0] -= 0.06
    assert _finish(PrecisionISLRecognizer(), [[hand]] * 8) is None
    assert _finish(PrecisionISLRecognizer(), [[open_hand()]] * 8) is None


def test_bye_requires_side_to_side_motion():
    wave = [[_shape(x=x, y=0.40)] for x in (0.60, 0.66, 0.72, 0.66, 0.60, 0.66, 0.72, 0.66, 0.60)]
    assert _finish(PrecisionISLRecognizer(), wave)["english"] == "Bye!"
    assert _finish(PrecisionISLRecognizer(), wave[:3] + [wave[2]] * 5)["english"] == "Hello"
    assert _finish(PrecisionISLRecognizer(), [[_shape(x=0.66, y=0.56)]] * 8) is None


def _thumbs_up_pose(x=0.51, y=0.58):
    hand = _shape(x=x, y=y, straight=(False,) * 4)
    hand[3] = [x - 0.04, y - 0.10, 0]
    hand[4] = [x - 0.04, y - 0.18, 0]
    return hand


def test_help_me_accepts_a_held_stack_or_a_joint_lift():
    fist = _thumbs_up_pose()
    support = _shape(x=0.41, y=0.53, rotation=np.pi / 2)
    offsets = np.linspace(0.0, 0.075, 8)
    lift = [[fist - [0, dy, 0], support - [0, dy, 0]] for dy in offsets]
    assert _finish(PrecisionISLRecognizer(), lift)["english"] == "Help me"
    assert _finish(PrecisionISLRecognizer(), [[fist, support]] * 8)["english"] == "Help me"
    assert _finish(PrecisionISLRecognizer(), [[fist]] * 8)["english"] == "Alright"


def test_i_love_you_requires_all_three_signs_in_order():
    toward_self = _shape(x=0.30, straight=(True, False, False, False), rotation=np.pi / 2)
    cross = [_shape(x=0.53, rotation=-np.pi / 2), _shape(x=0.47, rotation=np.pi / 2)]
    toward_camera = _shape(straight=(True, False, False, False))
    toward_camera[6:9, :2] = toward_camera[5, :2]
    toward_camera[6:9, 2] = [-0.05, -0.08, -0.11]
    assert _finish(PrecisionISLRecognizer(), [[toward_self]]*4 + [cross]*4 + [[toward_camera]]*4)["english"] == "I love you"
    assert _finish(PrecisionISLRecognizer(), [cross]*4 + [[toward_self]]*4 + [[toward_camera]]*4) is None


def test_how_are_you_supports_point_then_two_thumbs_up():
    point = _shape(straight=(True, False, False, False))
    point[6:9, :2] = point[5, :2]
    point[6:9, 2] = [-0.05, -0.08, -0.11]
    thumbs = [_thumbs_up_pose(x=0.35), _thumbs_up_pose(x=0.62)]
    assert _finish(PrecisionISLRecognizer(), [[point]]*4 + [thumbs]*4)["english"] == "How are you?"


def _tuck_thumb(hand):
    hand[4] = hand[5] + [0.015, -0.025, 0]
    return hand


def _pinch_ear(x):
    hand = _shape(x=x, y=0.48, straight=(False, True, True, True))
    hand[4] = hand[8] = [x, 0.30, -0.01]
    return hand


def _cupped_palm(x, y):
    hand = _shape(x=x, y=y)
    distance = hand[:, 1] - y
    hand[:, 1] = y + distance * 0.40
    hand[:, 2] = distance * 0.80
    return hand


def _pinch_no():
    hand = _shape(straight=(False,) * 4)
    hand[4] = hand[8] = hand[12] = [0.48, 0.39, -0.08]
    return hand


def _prayer_hand():
    hand = _shape()
    hand[:, 0] = 0.5 + 0.14 * (hand[:, 0] - 0.5)
    return hand


@pytest.mark.parametrize(("hands", "english"), [
    ([_shape(x=0.48), _shape(x=0.52)], "Namaste"),
    ([_prayer_hand()], "Namaste"),
    ([_cupped_palm(0.47, 0.58)], "How are you?"),
    ([_cupped_palm(0.34, 0.58), _cupped_palm(0.66, 0.58)], "How are you?"),
    ([_shape(0.34, 0.56), _shape(0.66, 0.56)], "How are you?"),
    ([_tuck_thumb(_shape(straight=(True, True, True, False)))], "What are you doing?"),
    ([_shape(x=0.38, straight=(True, False, False, False)), _shape(x=0.61, straight=(True, False, False, False))], "What are you doing?"),
    ([_shape(x=0.43, y=0.22, rotation=np.pi / 2)], "I am a boy"),
    ([_shape(x=0.65, y=0.49, straight=(True, False, False, False))], "I am a girl"),
    ([_tuck_thumb(_shape(straight=(False,) * 4)), _shape(x=0.41, y=0.53, rotation=np.pi / 2)], "Help me"),
    ([_shape(straight=(True, False, False, True))], "I love you"),
    ([_shape(x=0.66, y=0.40)], "Hello"),
    ([_tuck_thumb(_shape(straight=(False,) * 4))], "Yes"),
    ([_pinch_no()], "No"),
    ([_thumbs_up_pose()], "Alright"),
    ([_pinch_ear(0.38), _pinch_ear(0.62)], "Sorry"),
])
def test_static_conversation_variants(hands, english):
    result = _finish(PrecisionISLRecognizer(), [hands] * 8)
    assert result is not None
    assert result["english"] == english


def test_welcome_requires_both_palms_to_sweep_inward():
    sweep = [[_cupped_palm(0.32 + shift, 0.52 + shift),
              _cupped_palm(0.68 - shift, 0.52 + shift)]
             for shift in np.linspace(0, 0.08, 10)]
    assert _finish(PrecisionISLRecognizer(), sweep)["english"] == "Welcome"
    assert _finish(PrecisionISLRecognizer(), [sweep[0]] * 10)["english"] == "How are you?"


@pytest.mark.parametrize(("pattern", "number"), [(pattern, v[0]) for pattern, v in COUNTS.items()])
def test_counting_reports_each_supplied_number_pose(pattern, number):
    hand = _shape(straight=pattern[1:])
    if not pattern[0]:
        _tuck_thumb(hand)
    counter = PrecisionISLRecognizer(mode="counting")
    assert counter.process(None, [hand], now=0) is None
    result = counter.process(None, [hand], now=0.25)
    assert result["english"].endswith(f"({number})")
    assert result["category"] == "Counting"


def test_guide_contains_translations_for_the_entire_requested_table():
    entries = {entry["gesture"]: entry for entry in gesture_guide()}
    required = {
        "NAMASTE", "HOW ARE YOU", "I AM FINE", "WHAT ARE YOU DOING", "I AM A BOY", "I AM A GIRL",
        "HELP ME", "I LOVE YOU", "HELLO", "YES", "NO", "ALRIGHT",
        "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "TEN",
    }
    assert required <= entries.keys()
    assert {"SORRY", "WELCOME"} <= entries.keys()
    assert all(e["english"] and e["hindi"] and e["instruction"] for e in entries.values())


def test_hello_tracks_face_position_instead_of_a_fixed_side_of_the_frame():
    hand = _shape(x=0.84, y=0.54)
    features = np.zeros(128, dtype=np.float32)
    recognizer = PrecisionISLRecognizer()
    face = (0.68, 0.43, 0.22, 0.24)
    for i in range(9):
        assert recognizer.process(features, [hand], now=i * 0.06, face=face) is None
    assert recognizer.process(features, [], now=0.56) is None
    assert recognizer.process(features, [], now=0.7)["english"] == "Hello"
