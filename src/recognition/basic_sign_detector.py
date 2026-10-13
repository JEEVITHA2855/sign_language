"""Basic Sign & Gesture Detector based on Hand Landmark Geometry.

Recognizes universal everyday sign language gestures with high precision and zero latency:
- HELLO / OPEN PALM
- YES / GOOD (Thumbs Up)
- NO / BAD (Thumbs Down)
- I LOVE YOU (Universal ASL ILY Sign)
- PEACE / VICTORY / TWO
- OK
- YOU / POINTING
- CALL ME (Shaka / Phone)
- STOP / FIST
- THANK YOU / NAMASTE (Two-hand praying pose)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np


class BasicSignDetector:
    """Accurately detects fundamental static sign language gestures from 21 MediaPipe landmarks."""

    BASIC_VOCABULARY = [
        "HELLO",
        "YES",
        "NO",
        "I LOVE YOU",
        "THANK YOU",
        "PEACE",
        "OK",
        "YOU",
        "CALL ME",
        "STOP",
    ]

    SIGN_DESCRIPTIONS = {
        "HELLO": "Open flat palm facing forward (wave/greeting)",
        "YES": "Thumbs Up with closed fist (agreement / good)",
        "NO": "Thumbs Down with closed fist (disagreement / bad)",
        "I LOVE YOU": "Thumb, Index, and Pinky extended upright (universal ILY sign)",
        "THANK YOU": "Flat open hand pressed forward or both hands joined",
        "PEACE": "Index and Middle fingers extended in a 'V' shape",
        "OK": "Thumb and Index tips touching in a ring, other 3 extended",
        "YOU": "Index finger pointing directly forward",
        "CALL ME": "Thumb and Pinky extended outward (phone gesture)",
        "STOP": "Closed fist held firmly",
    }

    def __init__(self, confidence: float = 0.85) -> None:
        self.default_confidence = confidence

    def detect(
        self,
        primary_hand_lms: Optional[np.ndarray],
        all_hands_lms: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Optional[str], float]:
        """Classify landmarks into a recognized basic sign.

        Parameters
        ----------
        primary_hand_lms : Optional[np.ndarray]
            (21, 3) normalized Cartesian landmarks of primary hand.
        all_hands_lms : Optional[List[np.ndarray]]
            List of (21, 3) landmarks for all detected hands.

        Returns
        -------
        Tuple[Optional[str], float]
            (sign_name, confidence) or (None, 0.0)
        """
        # 1. Two-hand Namaste / Thank You detection
        if all_hands_lms and len(all_hands_lms) >= 2:
            h1 = all_hands_lms[0]
            h2 = all_hands_lms[1]
            # Check proximity of wrists and index bases
            wrist_dist = np.linalg.norm(h1[0, :2] - h2[0, :2])
            palm_dist = np.linalg.norm(h1[9, :2] - h2[9, :2])
            if wrist_dist < 0.22 and palm_dist < 0.18:
                return "THANK YOU", 0.92

        if primary_hand_lms is None or len(primary_hand_lms) < 21:
            return None, 0.0

        lms = primary_hand_lms

        # In image coordinates: Y=0 is top, Y=1 is bottom
        # Therefore, tip.y < pip.y means pointing UPWARDS
        wrist = lms[0]
        thumb_tip = lms[4]
        thumb_ip = lms[3]
        thumb_mcp = lms[2]
        thumb_cmc = lms[1]

        index_tip, index_pip, index_mcp = lms[8], lms[6], lms[5]
        middle_tip, middle_pip, middle_mcp = lms[12], lms[10], lms[9]
        ring_tip, ring_pip, ring_mcp = lms[16], lms[14], lms[13]
        pinky_tip, pinky_pip, pinky_mcp = lms[20], lms[18], lms[17]

        # Finger extended flags (upright)
        index_up = index_tip[1] < index_pip[1] and index_tip[1] < index_mcp[1]
        middle_up = middle_tip[1] < middle_pip[1] and middle_tip[1] < middle_mcp[1]
        ring_up = ring_tip[1] < ring_pip[1] and ring_tip[1] < ring_mcp[1]
        pinky_up = pinky_tip[1] < pinky_pip[1] and pinky_tip[1] < pinky_mcp[1]

        # Finger folded flags (curled inward)
        index_folded = index_tip[1] > index_pip[1]
        middle_folded = middle_tip[1] > middle_pip[1]
        ring_folded = ring_tip[1] > ring_pip[1]
        pinky_folded = pinky_tip[1] > pinky_pip[1]

        # Thumb vertical direction
        thumb_pointing_up = thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]
        thumb_pointing_down = thumb_tip[1] > thumb_ip[1] > thumb_mcp[1]

        # Distances
        thumb_index_dist = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
        thumb_wrist_dist = np.linalg.norm(thumb_tip[:2] - wrist[:2])
        palm_center = (index_mcp[:2] + pinky_mcp[:2] + wrist[:2]) / 3.0
        thumb_palm_dist = np.linalg.norm(thumb_tip[:2] - palm_center)

        # -------------------------------------------------------------
        # Gesture Rule 1: YES / GOOD (Thumbs Up)
        # Thumb pointing strongly upward, all other 4 fingers curled
        # -------------------------------------------------------------
        if (
            thumb_pointing_up
            and thumb_tip[1] < index_mcp[1]
            and index_folded
            and middle_folded
            and ring_folded
            and pinky_folded
        ):
            return "YES", 0.94

        # -------------------------------------------------------------
        # Gesture Rule 2: NO / BAD (Thumbs Down)
        # Thumb pointing strongly downward, all other 4 fingers curled
        # -------------------------------------------------------------
        if (
            thumb_pointing_down
            and thumb_tip[1] > wrist[1] - 0.05
            and index_folded
            and middle_folded
            and ring_folded
            and pinky_folded
        ):
            return "NO", 0.94

        # -------------------------------------------------------------
        # Gesture Rule 3: I LOVE YOU (Universal ASL ILY Sign)
        # Thumb extended, Index extended, Pinky extended, Middle & Ring folded
        # -------------------------------------------------------------
        if (
            index_up
            and pinky_up
            and middle_folded
            and ring_folded
            and thumb_palm_dist > 0.12
        ):
            return "I LOVE YOU", 0.95

        # -------------------------------------------------------------
        # Gesture Rule 4: PEACE / VICTORY
        # Index & Middle extended upright in a V, Ring & Pinky curled
        # -------------------------------------------------------------
        if (
            index_up
            and middle_up
            and ring_folded
            and pinky_folded
            and not thumb_pointing_up
        ):
            return "PEACE", 0.92

        # -------------------------------------------------------------
        # Gesture Rule 5: OK
        # Thumb tip and Index tip touching, Middle, Ring & Pinky extended
        # -------------------------------------------------------------
        if (
            thumb_index_dist < 0.075
            and middle_up
            and ring_up
            and pinky_up
        ):
            return "OK", 0.93

        # -------------------------------------------------------------
        # Gesture Rule 6: CALL ME (Shaka / Phone)
        # Thumb & Pinky extended outward, Middle 3 folded
        # -------------------------------------------------------------
        if (
            pinky_up
            and index_folded
            and middle_folded
            and ring_folded
            and thumb_palm_dist > 0.14
        ):
            return "CALL ME", 0.91

        # -------------------------------------------------------------
        # Gesture Rule 7: YOU / POINTING
        # Index finger extended upright, other 3 folded
        # -------------------------------------------------------------
        if (
            index_up
            and middle_folded
            and ring_folded
            and pinky_folded
            and not pinky_up
        ):
            return "YOU", 0.90

        # -------------------------------------------------------------
        # Gesture Rule 8: HELLO / OPEN PALM
        # All 5 fingers extended outward/upright
        # -------------------------------------------------------------
        if (
            index_up
            and middle_up
            and ring_up
            and pinky_up
            and thumb_palm_dist > 0.12
        ):
            return "HELLO", 0.92

        # -------------------------------------------------------------
        # Gesture Rule 9: STOP / FIST
        # All 5 fingers folded
        # -------------------------------------------------------------
        if (
            index_folded
            and middle_folded
            and ring_folded
            and pinky_folded
            and not thumb_pointing_up
            and not thumb_pointing_down
        ):
            return "STOP", 0.88

        return None, 0.0
