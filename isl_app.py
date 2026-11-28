"""Run the local Windows camera and spoken-English ISL translator."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

ISL_DIR = Path(__file__).resolve().parent / "features" / "sign-language-interpreter" / "optimistic_isl"
sys.path.insert(0, str(ISL_DIR))

from isl_recognition_engine import (
    INCLUDE_VIDEOS, ISL_LESSON, ForegroundHands, PrecisionISLRecognizer, SigningFace, gesture_guide,
)
from isl_speech import WindowsSpeaker, format_transcript


def print_help():
    print("[n] Switch Conversation / Counting   [c] Clear   [s] Speak conversation")
    print("[m] Mirror preview   [g] Gesture guide   [h] Help   [q] Quit")


def print_guide():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Conversation: perform one complete sign, then lower your hands to finish.")
    print(f"ISLRTC greetings lesson: {ISL_LESSON}")
    print(f"INCLUDE training examples: {INCLUDE_VIDEOS}")
    for entry in gesture_guide():
        print(f"{entry['english']} ({entry['hindi']}, {entry['mode']})  {entry['instruction']}")
        if entry.get("example"):
            print(f"  INCLUDE example: {entry['example']}")
        if entry.get("video"):
            print(f"  {entry['video']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--mode", choices=("conversation", "counting"), default="conversation")
    parser.add_argument("--guide", action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    args = parser.parse_args()
    if args.guide:
        print_guide()
        return 0
    if not 0.65 <= args.threshold <= 0.95:
        parser.error("--threshold must be between 0.65 and 0.95")

    import cv2
    from inference.hand_detector import RealTimeHandDetector

    recognizer = PrecisionISLRecognizer(args.mode, args.threshold)
    recognizer.load_model()
    foreground = ForegroundHands()
    face_tracker = SigningFace()
    detector = RealTimeHandDetector(detection_width=384, foreground_selector=foreground.select)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        detector.close()
        print(f"Could not open camera {args.camera}.", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    latest_frame = deque(maxlen=1)
    stopping = threading.Event()

    def capture():
        try:
            while not stopping.is_set():
                ready, frame = cap.read()
                if ready:
                    latest_frame.append(frame)
                else:
                    stopping.wait(0.01)
        finally:
            cap.release()

    thread = threading.Thread(target=capture, daemon=True)
    speaker = None if args.no_voice else WindowsSpeaker()
    mirror = True
    previous_frame = None
    history = []
    last_result = ""
    last_frame_time = time.monotonic()
    window = "The OptiMistic Glasses - ISL Translator"
    print_help()
    thread.start()

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        while not stopping.is_set():
            frame = latest_frame[-1] if latest_frame else None
            if frame is not None and frame is not previous_frame:
                previous_frame = frame
                last_frame_time = time.monotonic()
                features, _, hands = detector.process_bgr_frame(frame)
                face = face_tracker.locate(frame) if hands and recognizer.mode == "conversation" else None
                result = recognizer.process(features, hands, face=face)
                preview = recognizer.get_live_sentence_preview()
                if result:
                    is_complete = result.get("is_sentence_complete", False)
                    if recognizer.mode == "conversation":
                        if is_complete:
                            last_result = result["english"]
                            print(f"[Sentence Spoken] > {last_result}", flush=True)
                            history.append(last_result)
                            if speaker:
                                speaker.speak(last_result)
                        else:
                            last_sign = result["english"]
                            print(f"[Sign Detected] > {last_sign} (Draft: {preview.get('sentence', '')})", flush=True)
                    else:
                        last_result = result["english"]
                        print(f"[Counting] > {last_result}", flush=True)
                        history.append(last_result)

                display = cv2.flip(frame, 1) if mirror else frame.copy()
                if mirror:
                    hands = [hand.copy() for hand in hands]
                    for hand in hands:
                        hand[:, 0] = 1.0 - hand[:, 0]
                detector.draw_landmarks(display, hands)
                if recognizer.mode == "counting":
                    hint = "Hold a number pose. Audio off."
                else:
                    hint = recognizer.status
                cv2.rectangle(display, (0, 0), (display.shape[1], 100), (18, 22, 32), -1)
                active_text = preview.get("sentence") if preview.get("active") else last_result
                display_lines = [
                    (f"{recognizer.mode.title()} | {hint}", 28),
                    (f"Sentence: {active_text}" if active_text else "Waiting for signs...", 62),
                    (f"Tokens: {' -> '.join(preview.get('tokens', []))}" if preview.get("tokens") else "", 90),
                ]
                for line, y in display_lines:
                    if line:
                        cv2.putText(display, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
                cv2.imshow(window, display)

            key = cv2.waitKey(8) & 0xFF
            if key == ord("q") or (previous_frame is not None and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1):
                break
            if time.monotonic() - last_frame_time > 3:
                print("The camera stopped delivering frames.", file=sys.stderr)
                return 1
            if key == ord("n"):
                recognizer.mode = "counting" if recognizer.mode == "conversation" else "conversation"
                recognizer.reset()
                last_result = ""
                if speaker:
                    speaker.clear()
            elif key == ord("c"):
                recognizer.reset()
                foreground.reset()
                face_tracker.reset()
                history.clear()
                last_result = ""
                if speaker:
                    speaker.clear()
            elif key == ord("s") and speaker and history and recognizer.mode == "conversation":
                speaker.speak(format_transcript(history))
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("g"):
                print_guide()
            elif key == ord("h"):
                print_help()
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        thread.join(timeout=2)
        detector.close()
        if speaker:
            speaker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
