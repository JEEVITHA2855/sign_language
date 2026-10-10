"""State-based prediction stabilization and gesture debouncing.

Implements a formal recognition state machine:
    NO_HAND
       ↓
    HAND_DETECTED
       ↓
    STABLE_PREDICTION (stable for N consecutive frames >= threshold)
       ↓
    LETTER_COMMITTED (commit once)
       ↓
    WAIT_FOR_RELEASE (held gesture ignored while hand remains unchanged)
       ↓
    (Gesture Change -> STABLE_PREDICTION for new gesture)
    (Hand Leaves -> NO_HAND with hand_released = True)
"""

from __future__ import annotations

from typing import Optional, Tuple


class GestureState:
    NO_HAND = "NO_HAND"
    HAND_DETECTED = "HAND_DETECTED"
    STABLE_PREDICTION = "STABLE_PREDICTION"
    LETTER_COMMITTED = "LETTER_COMMITTED"
    WAIT_FOR_RELEASE = "WAIT_FOR_RELEASE"


class Stabilizer:
    """State-based gesture stabilizer preventing continuous letter repetition."""

    def __init__(
        self,
        window: int = 5,
        min_ratio: float = 0.70,
        min_conf: float = 0.70,
        debounce_frames: int = 10,
        required_stable_frames: int = 5,
    ) -> None:
        """Initialize the state-based gesture stabilizer.

        Parameters
        ----------
        window : int
            Alias/default for required stable frames.
        min_ratio : float
            Legacy compatibility parameter.
        min_conf : float
            Minimum confidence threshold (e.g. 0.70).
        debounce_frames : int
            Legacy parameter.
        required_stable_frames : int
            Number of consecutive frames a gesture must remain stable to commit.
        """
        self.required_stable_frames = max(1, required_stable_frames or window)
        self.min_conf = float(min_conf)
        self.min_ratio = float(min_ratio)
        self.debounce_frames = debounce_frames

        # State machine variables
        self.state: str = GestureState.NO_HAND
        self.hand_detected: bool = False
        self.hand_released: bool = True  # True if hand was removed since last commit
        self.current_candidate: Optional[str] = None
        self.stable_count: int = 0
        self.last_committed: Optional[str] = None
        self.current_confidence: float = 0.0

    def update(self, label: str, confidence: float) -> Optional[str]:
        """Update the state machine with the current frame's prediction.

        Parameters
        ----------
        label : str
            Predicted class label (e.g. 'A', 'B', 'SPACE', 'DEL').
        confidence : float
            Prediction confidence score in [0.0, 1.0].

        Returns
        -------
        Optional[str]
            Committed label on the exact frame it is accepted, else None.
        """
        self.current_confidence = float(confidence)

        # Check for invalid or empty predictions
        if not label or label in ("NONE", "NO_HAND", "UNCERTAIN", "UNKNOWN", "MODEL_UNAVAILABLE"):
            return self.mark_no_hand()

        self.hand_detected = True

        # Confidence threshold check
        if confidence < self.min_conf:
            # Below confidence cutoff: if waiting for release, stay there;
            # otherwise decay stable count so noise doesn't trigger a commit.
            if self.state == GestureState.WAIT_FOR_RELEASE:
                return None
            self.stable_count = max(0, self.stable_count - 1)
            if self.stable_count == 0:
                self.state = GestureState.HAND_DETECTED
            return None

        # Clean label string
        norm_label = label.strip().upper()

        # If user is still holding the exact same gesture that was already committed
        # without having released their hand:
        if norm_label == self.last_committed and not self.hand_released:
            self.state = GestureState.WAIT_FOR_RELEASE
            self.current_candidate = norm_label
            self.stable_count = self.required_stable_frames
            return None

        # Gesture change or new gesture detected
        if norm_label != self.current_candidate:
            # Transition to a new candidate gesture
            self.current_candidate = norm_label
            self.stable_count = 1
            self.state = GestureState.STABLE_PREDICTION
        else:
            # Same candidate sustained for another consecutive frame
            self.stable_count += 1

        # Check if stability threshold is reached
        if self.stable_count >= self.required_stable_frames:
            # Check once more if this candidate matches last_committed without release
            if norm_label == self.last_committed and not self.hand_released:
                self.state = GestureState.WAIT_FOR_RELEASE
                return None

            # COMMIT the letter!
            committed_token = norm_label
            self.last_committed = committed_token
            self.hand_released = False
            self.state = GestureState.LETTER_COMMITTED
            return committed_token

        return None

    def mark_no_hand(self) -> None:
        """Inform the stabilizer that no hand is detected in this frame."""
        self.hand_detected = False
        self.hand_released = True  # Hand was lifted/released
        self.current_candidate = None
        self.stable_count = 0
        self.state = GestureState.NO_HAND
        return None

    def reset(self) -> None:
        """Hard reset of stabilizer state."""
        self.mark_no_hand()
        self.last_committed = None

    def get_progress(self) -> Tuple[Optional[str], float, float]:
        """Return candidate label, progress ratio (0.0 to 1.0), and confidence."""
        ratio = min(1.0, self.stable_count / self.required_stable_frames) if self.required_stable_frames > 0 else 0.0
        return self.current_candidate, ratio, self.current_confidence
