"""Voice coaching module — async TTS feedback using macOS say command."""

from __future__ import annotations

import random
import subprocess
import threading
import time

from app.scorer import ScoreResult, PERFECT, MISS, CLOSE

# Preferred voices in priority order — first available one wins
FEMALE_VOICES = ["Ava", "Samantha", "Nicky", "Karen", "Victoria", "Alex"]
MALE_VOICES   = ["Tom", "Daniel", "Fred", "Ralph", "Alex"]

PERFECT_PHRASES = [
    "Yeah, that's it! Perfect form",
    "Nailed it, great position",
    "Perfect! Keep that up",
    "That's the one, hold it",
]

CLOSE_PHRASES = [
    "Almost there — just open up your {joint} a little more",
    "Getting closer — adjust your {joint} slightly",
    "Nearly perfect — tweak that {joint}",
]

EXTEND_PHRASES = [
    "Straighten your {joint} out more",
    "Open up that {joint}",
    "Get that {joint} fully extended",
]

BEND_PHRASES = [
    "Bring your {joint} in a bit",
    "Bend that {joint} more",
    "Tuck your {joint} in",
]

NO_BODY_PHRASES = [
    "Hey, step back so I can see your full body",
    "Step back a bit — I need to see your hips and shoulders",
    "Move back so your full body is in frame",
]


def _get_available_voice(preferred_list: list[str]) -> str:
    """Return the first voice from the list that is installed on this machine."""
    try:
        result = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True, text=True, timeout=5
        )
        installed = result.stdout.lower()
        for voice in preferred_list:
            if voice.lower() in installed:
                return voice
    except Exception:
        pass
    # fallback — let macOS use its default
    return ""


class VoiceCoach:
    """Async TTS coach using macOS say command with male/female voice selection."""

    def __init__(
        self,
        cooldown: float = 4.0,
        gender: str = "female",   # "female" or "male"
        rate: int = 155,
    ) -> None:
        self._cooldown = cooldown
        self._rate     = rate
        self._last_spoken: float = 0.0
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # pick voice based on gender preference
        voice_list = FEMALE_VOICES if gender == "female" else MALE_VOICES
        self._voice = _get_available_voice(voice_list)

        # test if say command is available
        try:
            subprocess.run(["say", "--version"], capture_output=True, timeout=2)
            self._available = True
            voice_display = self._voice if self._voice else "system default"
            print(f"VoiceCoach: ready — voice: {voice_display} ({gender})")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._available = False
            print("VoiceCoach: 'say' command not found — voice disabled")

    def _speak_async(self, text: str) -> None:
        """Speak text on a daemon thread using macOS say command."""
        voice = self._voice
        rate  = self._rate

        def run() -> None:
            try:
                cmd = ["say", "-r", str(rate)]
                if voice:
                    cmd += ["-v", voice]
                cmd.append(text)
                subprocess.run(cmd, capture_output=True, timeout=10)
            except Exception as e:
                print(f"VoiceCoach error: {e}")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=run, daemon=True)
            self._thread.start()

    def speak(self, text: str) -> None:
        """Speak text if cooldown has elapsed."""
        if not self._available:
            return
        now = time.time()
        if now - self._last_spoken < self._cooldown:
            return
        print(f"VoiceCoach: '{text}'")
        self._last_spoken = now
        self._speak_async(text)

    def set_gender(self, gender: str) -> None:
        """Switch voice gender at runtime — 'female' or 'male'."""
        voice_list   = FEMALE_VOICES if gender == "female" else MALE_VOICES
        self._voice  = _get_available_voice(voice_list)
        voice_display = self._voice if self._voice else "system default"
        print(f"VoiceCoach: switched to {gender} voice — {voice_display}")

    def feedback_from_score(self, score_result: ScoreResult) -> None:
        """Speak joint-specific feedback based on the latest ScoreResult."""
        if score_result is None:
            return

        misses   = [r for r in score_result.results if r.tier == MISS]
        closes   = [r for r in score_result.results if r.tier == CLOSE]
        perfects = [r for r in score_result.results if r.tier == PERFECT]

        if not misses and not closes and perfects:
            self.speak(random.choice(PERFECT_PHRASES))
        elif not misses and closes:
            joint = closes[0].joint_name.replace("_", " ")
            self.speak(random.choice(CLOSE_PHRASES).format(joint=joint))
        elif misses:
            joint  = misses[0].joint_name.replace("_", " ")
            actual = misses[0].actual_angle
            target = misses[0].target_angle
            if actual < target:
                self.speak(random.choice(EXTEND_PHRASES).format(joint=joint))
            else:
                self.speak(random.choice(BEND_PHRASES).format(joint=joint))

    def feedback_no_body(self) -> None:
        self.speak(random.choice(NO_BODY_PHRASES))

    def feedback_aligned(self) -> None:
        self.speak("Nice, hold that")

    def feedback_buffer_ready(self) -> None:
        self.speak("Okay, go ahead")