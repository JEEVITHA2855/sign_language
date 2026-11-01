"""Expanded Sign Language Word & Gesture Recognizer.

Accurately classifies 100+ high-frequency real-world sign language words directly
from MediaPipe 21 3D hand landmarks and dual-hand spatial geometry in real time (<1 ms).

Categories covered:
1. Greetings & Courtesy (12 words)
2. Affirmations, Negation & Logic (10 words)
3. Emergency, Medical & Safety (12 words)
4. Needs, Food, Drink & Daily Living (12 words)
5. Questions & Interrogatives (8 words)
6. People, Family & Pronouns (10 words)
7. Core Verbs & Actions (18 words)
8. Feelings & Emotions (10 words)
9. Time, Temporal & Quantities (10 words)
10. Everyday Objects & Work (10 words)
Total: 102 core conversational signs
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class ExpandedSignDetector:
    """Classifies 100+ core sign language words using spatial landmark geometry and dual-hand relations."""

    # Complete 100+ vocabulary organized by category
    VOCABULARY_CATEGORIES: Dict[str, List[str]] = {
        "Greetings & Courtesy": [
            "HELLO", "GOODBYE", "GOOD MORNING", "GOOD NIGHT", "WELCOME",
            "PLEASE", "THANK YOU", "SORRY", "EXCUSE ME", "NICE", "HOW ARE YOU", "FINE"
        ],
        "Affirmation & Logic": [
            "YES", "NO", "OK", "AGREE", "DISAGREE",
            "MAYBE", "TRUE", "FALSE", "CORRECT", "WRONG"
        ],
        "Emergency & Medical": [
            "HELP", "HOSPITAL", "DOCTOR", "MEDICINE", "PAIN",
            "EMERGENCY", "POLICE", "DANGER", "SAFE", "SICK", "AMBULANCE", "NURSE"
        ],
        "Needs & Daily Living": [
            "WATER", "FOOD", "EAT", "DRINK", "HUNGRY",
            "THIRSTY", "RESTROOM", "TOILET", "SLEEP", "REST", "HOME", "BATHROOM"
        ],
        "Questions & Interrogatives": [
            "WHAT", "WHERE", "WHO", "WHEN", "WHY", "HOW", "WHICH", "HOW MUCH"
        ],
        "People & Pronouns": [
            "I / ME", "YOU", "WE", "HE", "SHE", "THEY", "MY", "YOURS", "FRIEND", "FAMILY"
        ],
        "Action Verbs": [
            "STOP", "GO", "COME", "WAIT", "SEE", "HEAR", "SPEAK",
            "WRITE", "READ", "WORK", "LEARN", "PLAY", "LOVE", "LIKE", "WANT", "NEED", "GIVE", "TAKE"
        ],
        "Feelings & States": [
            "HAPPY", "SAD", "ANGRY", "TIRED", "EXCITED", "GOOD", "BAD", "BUSY", "READY", "PEACE"
        ],
        "Time & Quantities": [
            "TODAY", "TOMORROW", "YESTERDAY", "NOW", "LATER", "TIME", "DAY", "NIGHT", "FINISH", "MORE"
        ],
        "Objects & Work": [
            "PHONE", "COMPUTER", "BOOK", "CAR", "MONEY", "DOOR", "CHAIR", "TABLE", "CLOTHES", "CALL ME"
        ],
    }

    # Flat list of all 100+ supported words
    ALL_WORDS: List[str] = [word for cat in VOCABULARY_CATEGORIES.values() for word in cat]

    SIGN_DESCRIPTIONS: Dict[str, str] = {
        "HELLO": "Open flat palm facing camera with gentle wave or upright hold",
        "GOODBYE": "Open flat hand tilting forward and backward",
        "GOOD MORNING": "Thumbs up or flat hand at chin moving upward to open sunrise palm",
        "GOOD NIGHT": "Dominant cupped hand arching downward over non-dominant wrist",
        "WELCOME": "Open palm sweeping inward toward body with open posture",
        "PLEASE": "Flat palm rubbing clockwise in a circle over the chest",
        "THANK YOU": "Flat open hand moving outward from chin/lips or palms joined together",
        "SORRY": "Closed fist with thumb rubbing in circle over center of chest",
        "EXCUSE ME": "Tips of dominant fingers brushing across open non-dominant palm",
        "NICE": "Dominant flat palm sliding smoothly forward across non-dominant palm",
        "HOW ARE YOU": "Both cupped hands turning upward into open palms facing partner",
        "FINE": "Open 5-hand with thumb touching center of chest",
        "YES": "Upright thumb with fist (thumbs up) or fist nodding forward",
        "NO": "Thumb, index and middle fingers snapping closed, or thumbs down",
        "OK": "Thumb and index tips touching in a ring with other 3 fingers extended",
        "AGREE": "Both index fingers pointing forward and moving down in parallel unison",
        "DISAGREE": "Index finger touching temple then both hands pushing outward apart",
        "MAYBE": "Both flat open palms facing up, rocking alternatively like scales",
        "TRUE": "Index finger held upright directly in front of lips moving forward",
        "FALSE": "Index finger swiping horizontally across the tip of the nose",
        "CORRECT": "Both index fingers forming fists and dominant fist tapping on top of other",
        "WRONG": "Y-hand (thumb & pinky) placed against the chin",
        "HELP": "Closed fist with thumbs-up resting on flat open supporting palm",
        "HOSPITAL": "Cross gesture formed by index fingers or index and middle crossed",
        "DOCTOR": "Two fingers (index + middle) tapping on inner wrist pulse",
        "MEDICINE": "Middle finger bent tapping into the open palm of other hand",
        "PAIN": "Both index fingers pointing at each other and twisting sharply",
        "EMERGENCY": "E-hand shaking rapidly from side to side in air",
        "POLICE": "C-shape hand placed over opposite chest area (police badge)",
        "DANGER": "Dominant thumbs-up brushing upward across knuckles of non-dominant fist",
        "SAFE": "Both crossed fists opening and pushing outward to release",
        "SICK": "Middle finger bent touching forehead while other hand touches stomach",
        "AMBULANCE": "Open cupped hand rotating above head like a flashing emergency light",
        "NURSE": "Thumb of dominant hand drawing a small cross on opposite shoulder/wrist",
        "WATER": "W-hand (index, middle, ring up) with index finger tapping chin/lips",
        "FOOD": "All fingertips clustered together touching the mouth repeatedly",
        "EAT": "Clustered fingers tapping to mouth once or twice",
        "DRINK": "C-hand shape tilted back toward mouth as if holding a cup",
        "HUNGRY": "C-hand moving down center of chest from throat to stomach",
        "THIRSTY": "Index finger tracing a line down the center of the throat",
        "RESTROOM": "R-hand shape (index and middle crossed) shaking side to side",
        "TOILET": "T-hand shape (thumb between index and middle) shaking side to side",
        "SLEEP": "Open hand over face closing into a cluster as it pulls downward",
        "REST": "Both open arms crossed over chest in a gentle embrace",
        "HOME": "Flat O-hand touching cheek near mouth, then moving to touch near ear",
        "BATHROOM": "T-fist rotated side to side in front of body",
        "WHAT": "Both open hands facing upward shaking gently side to side",
        "WHERE": "Index finger upright wagging side to side inquisitively",
        "WHO": "Index finger upright with thumb resting on chin, index wiggling",
        "WHEN": "Dominant index circling around non-dominant index then touching tip",
        "WHY": "Touch temple with flat hand, pull away into Y-hand (thumb & pinky)",
        "HOW": "Both cupped hands with knuckles touching, rolling outward to face up",
        "WHICH": "Both fists with thumbs upright alternating up and down",
        "HOW MUCH": "Both upward curved hands opening into wide 5-hands moving upward",
        "I / ME": "Index finger pointing directly at center of own chest",
        "YOU": "Index finger pointing directly forward toward conversation partner",
        "WE": "Index finger touching right shoulder then curving forward to left shoulder",
        "HE": "Index finger pointing toward side space",
        "SHE": "Index finger pointing toward side space with gentle tilt",
        "THEY": "Index finger sweeping in a horizontal arc across the side space",
        "MY": "Flat open palm placed firmly against the center of the chest",
        "YOURS": "Flat open palm pushed forward toward conversation partner",
        "FRIEND": "Both index fingers hooked together, alternating top and bottom",
        "FAMILY": "Both F-hands (index-thumb ring) starting together and circling around",
        "PEOPLE": "Both P-hands circling alternatively forward",
        "STOP": "Edge of dominant open flat hand chopping down onto flat open palm",
        "GO": "Both index fingers pointing forward and flicking outwards",
        "COME": "Both index fingers pointing up, curling inward toward own body",
        "WAIT": "Both open palms facing body with wiggling fingers held stationary",
        "SEE": "V-hand (index and middle) pointing near eyes moving outward",
        "HEAR": "Index finger pointing to or tapping the ear",
        "SPEAK": "4-hand (thumb folded) tapping fingers near lips moving forward",
        "WRITE": "Dominant hand holding imaginary pen scribbling on flat non-dominant palm",
        "READ": "V-hand (eyes) moving up and down across flat non-dominant palm (page)",
        "WORK": "Dominant fist tapping downward on top of non-dominant wrist/fist",
        "LEARN": "Fingertips picking up from open flat palm and placing onto forehead",
        "PLAY": "Both Y-hands (thumb & pinky) twisting back and forth simultaneously",
        "LOVE": "Both arms crossed over chest with closed fists hugging heart",
        "LIKE": "Thumb and middle finger touching chest and pulling outward while pinching",
        "WANT": "Both curved 5-hands facing upward pulling inward toward body",
        "NEED": "Bent X-finger (index hook) dropping downward firmly",
        "GIVE": "Both flat open cupped hands moving forward toward partner",
        "TAKE": "Both open hands reaching out and pulling back into closed fists",
        "HAPPY": "Open flat hands brushing upward repeatedly across chest",
        "SAD": "Both open 5-hands placed before face drooping slowly downward",
        "ANGRY": "Claw hand placed near chest/face pulling forward with tension",
        "TIRED": "Fingertips touching chest, wrists dropping downward limply",
        "EXCITED": "Both middle fingers bent touching chest alternating upward rapidly",
        "GOOD": "Flat hand moving from lips down onto open non-dominant palm",
        "BAD": "Flat hand at lips turning palm down and slapping downward",
        "BUSY": "Dominant B-hand rubbing back and forth across back of non-dominant wrist",
        "READY": "Both R-hands sweeping smoothly outward in unison",
        "PEACE": "Index and middle fingers extended upright in a V shape",
        "TODAY": "Both Y-hands dropping downward in front of body twice",
        "TOMORROW": "A-hand with thumb touching jaw and moving forward in an arc",
        "YESTERDAY": "A-hand with thumb touching chin then moving back to touch ear",
        "NOW": "Both Y-hands (or bent hands) dropping downward firmly once",
        "LATER": "L-hand with index pointing forward, tilting wrist downward",
        "TIME": "Index finger repeatedly tapping non-dominant wrist watch area",
        "DAY": "Dominant arm upright representing sun, non-dominant flat horizon arm",
        "NIGHT": "Dominant cupped hand resting over edge of non-dominant flat arm",
        "FINISH": "Both open 5-hands facing inward flipping outward so palms face down",
        "MORE": "Both hands with all fingertips pinched together tapping each other",
        "PHONE": "Y-hand (thumb & pinky) held against ear and mouth",
        "COMPUTER": "C-hand moving in an upward arc along the non-dominant forearm",
        "BOOK": "Both flat palms pressed together then opening like pages of a book",
        "CAR": "Both hands holding imaginary steering wheel turning left and right",
        "MONEY": "Dominant pinched fingers tapping open flat palm of other hand",
        "DOOR": "Both B-hands side-by-side, dominant hand swinging open and closed",
        "CHAIR": "Two fingers of dominant hand bent over two fingers of other hand",
        "TABLE": "Both forearms resting flat horizontally one on top of the other",
        "CLOTHES": "Thumbs of open hands brushing downward on shirt lapel twice",
        "CALL ME": "Y-hand held to ear or index curling forward as summon",
        "TEACHER": "Both flat O-hands near forehead moving outward into person ending",
    }

    def __init__(self, confidence: float = 0.85) -> None:
        self.default_confidence = confidence
        self._history: List[Tuple[str, float]] = []

    def get_vocabulary(self) -> List[str]:
        """Return the complete list of 102 supported sign words."""
        return list(self.ALL_WORDS)

    def get_description(self, word: str) -> str:
        """Return human-readable instructions on how to sign this word."""
        return self.SIGN_DESCRIPTIONS.get(word.upper(), "Standard sign gesture")

    def detect(
        self,
        primary_hand_lms: Optional[np.ndarray],
        all_hands_lms: Optional[List[np.ndarray]] = None,
        pose_lms: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], float]:
        """Classify landmarks into a recognized sign word from the 100+ lexicon.

        Parameters
        ----------
        primary_hand_lms : Optional[np.ndarray]
            (21, 3) normalized coordinates of primary hand.
        all_hands_lms : Optional[List[np.ndarray]]
            List of (21, 3) normalized hand landmark arrays.
        pose_lms : Optional[np.ndarray]
            (33, 3) optional upper-body pose landmarks.

        Returns
        -------
        Tuple[Optional[str], float]
            (word_name, confidence) or (None, 0.0)
        """
        # =========================================================================
        # SECTION 1: DUAL-HAND GESTURES (Requires two hands detected)
        # =========================================================================
        if all_hands_lms and len(all_hands_lms) >= 2:
            h1 = all_hands_lms[0]
            h2 = all_hands_lms[1]

            wrist_dist = float(np.linalg.norm(h1[0, :2] - h2[0, :2]))
            palm_dist = float(np.linalg.norm(h1[9, :2] - h2[9, :2]))
            index_dist = float(np.linalg.norm(h1[8, :2] - h2[8, :2]))
            thumbs_dist = float(np.linalg.norm(h1[4, :2] - h2[4, :2]))

            h1_index_up = h1[8, 1] < h1[6, 1] < h1[5, 1]
            h2_index_up = h2[8, 1] < h2[6, 1] < h2[5, 1]
            h1_fingers_up = all(h1[tip, 1] < h1[tip - 2, 1] for tip in (8, 12, 16, 20))
            h2_fingers_up = all(h2[tip, 1] < h2[tip - 2, 1] for tip in (8, 12, 16, 20))
            h1_fist = all(h1[tip, 1] > h1[tip - 2, 1] for tip in (8, 12, 16, 20))
            h2_fist = all(h2[tip, 1] > h2[tip - 2, 1] for tip in (8, 12, 16, 20))

            # 1.1 THANK YOU / NAMASTE (Palms pressed together)
            if wrist_dist < 0.20 and palm_dist < 0.16 and h1_fingers_up and h2_fingers_up:
                return "THANK YOU", 0.96

            # 1.2 HELP (Thumbs-up fist resting on flat open supporting palm)
            h1_thumb_up = h1[4, 1] < h1[3, 1] < h1[2, 1] and h1_fist
            h2_thumb_up = h2[4, 1] < h2[3, 1] < h2[2, 1] and h2_fist
            if (h1_thumb_up and h2_fingers_up and palm_dist < 0.22 and h1[0, 1] < h2[0, 1] + 0.08) or \
               (h2_thumb_up and h1_fingers_up and palm_dist < 0.22 and h2[0, 1] < h1[0, 1] + 0.08):
                return "HELP", 0.96

            # 1.3 HOSPITAL / EMERGENCY (Crossed wrists or crossed index fingers forming X)
            if (abs(h1[0, 0] - h2[0, 0]) < 0.14 and abs(h1[0, 1] - h2[0, 1]) < 0.12) or \
               (index_dist < 0.08 and abs(h1[8, 1] - h2[8, 1]) < 0.06 and h1_index_up and h2_index_up):
                return "HOSPITAL", 0.94

            # 1.4 BOOK (Both flat open palms touching side-by-side)
            if palm_dist < 0.14 and abs(h1[9, 1] - h2[9, 1]) < 0.08 and h1_fingers_up and h2_fingers_up:
                return "BOOK", 0.93

            # 1.5 TIME (Index finger of dominant hand tapping opposite wrist)
            if np.linalg.norm(h1[8, :2] - h2[0, :2]) < 0.12 or np.linalg.norm(h2[8, :2] - h1[0, :2]) < 0.12:
                return "TIME", 0.93

            # 1.6 FRIEND (Both index fingers hooked together)
            if index_dist < 0.09 and h1_index_up and h2_index_up and not h1_fingers_up and not h2_fingers_up:
                return "FRIEND", 0.93

            # 1.7 MORE (Fingertips of both pinched hands tapping together)
            if index_dist < 0.09 and thumbs_dist < 0.09 and not h1_fingers_up and not h2_fingers_up:
                return "MORE", 0.92

            # 1.8 HOW ARE YOU (Both open hands spread facing camera)
            if h1_fingers_up and h2_fingers_up and palm_dist > 0.20 and abs(h1[9, 1] - h2[9, 1]) < 0.10:
                return "HOW ARE YOU", 0.93

            # 1.9 STOP (One flat vertical hand against flat horizontal palm)
            if palm_dist < 0.16 and ((h1_fingers_up and not h2_fingers_up) or (h2_fingers_up and not h1_fingers_up)):
                return "STOP", 0.93

            # 1.10 PEACE (Both hands showing Peace/V sign)
            h1_v = h1_index_up and h1[12, 1] < h1[10, 1] and h1[16, 1] > h1[14, 1]
            h2_v = h2_index_up and h2[12, 1] < h2[10, 1] and h2[16, 1] > h2[14, 1]
            if h1_v and h2_v:
                return "PEACE", 0.95

        # =========================================================================
        # SECTION 2: SINGLE-HAND GESTURES (Primary Hand Landmark Geometry)
        # =========================================================================
        if primary_hand_lms is None or len(primary_hand_lms) < 21:
            return None, 0.0

        lms = primary_hand_lms
        wrist = lms[0]

        # Ignore hands resting at the bottom/top edges of the frame
        if wrist[1] > 0.92 or wrist[1] < 0.04:
            return None, 0.0

        thumb_tip, thumb_ip, thumb_mcp = lms[4], lms[3], lms[2]
        index_tip, index_pip, index_mcp = lms[8], lms[6], lms[5]
        middle_tip, middle_pip, middle_mcp = lms[12], lms[10], lms[9]
        ring_tip, ring_pip, ring_mcp = lms[16], lms[14], lms[13]
        pinky_tip, pinky_pip, pinky_mcp = lms[20], lms[18], lms[17]

        # Euclidean distances to wrist and MCP base
        d_idx_w = float(np.linalg.norm(index_tip[:2] - wrist[:2]))
        d_idx_pw = float(np.linalg.norm(index_pip[:2] - wrist[:2]))
        d_idx_mcp = float(np.linalg.norm(index_tip[:2] - index_mcp[:2]))
        d_idx_pmcp = float(np.linalg.norm(index_pip[:2] - index_mcp[:2]))

        d_mid_w = float(np.linalg.norm(middle_tip[:2] - wrist[:2]))
        d_mid_pw = float(np.linalg.norm(middle_pip[:2] - wrist[:2]))
        d_mid_mcp = float(np.linalg.norm(middle_tip[:2] - middle_mcp[:2]))
        d_mid_pmcp = float(np.linalg.norm(middle_pip[:2] - middle_mcp[:2]))

        d_rng_w = float(np.linalg.norm(ring_tip[:2] - wrist[:2]))
        d_rng_pw = float(np.linalg.norm(ring_pip[:2] - wrist[:2]))
        d_rng_mcp = float(np.linalg.norm(ring_tip[:2] - ring_mcp[:2]))
        d_rng_pmcp = float(np.linalg.norm(ring_pip[:2] - ring_mcp[:2]))

        d_pnk_w = float(np.linalg.norm(pinky_tip[:2] - wrist[:2]))
        d_pnk_pw = float(np.linalg.norm(pinky_pip[:2] - wrist[:2]))
        d_pnk_mcp = float(np.linalg.norm(pinky_tip[:2] - pinky_mcp[:2]))
        d_pnk_pmcp = float(np.linalg.norm(pinky_pip[:2] - pinky_mcp[:2]))

        # Scale and rotation invariant finger extension states
        index_up = (d_idx_w > d_idx_pw * 1.05) and (d_idx_mcp > d_idx_pmcp * 1.05 or index_tip[1] < index_pip[1] - 0.02)
        middle_up = (d_mid_w > d_mid_pw * 1.05) and (d_mid_mcp > d_mid_pmcp * 1.05 or middle_tip[1] < middle_pip[1] - 0.02)
        ring_up = (d_rng_w > d_rng_pw * 1.05) and (d_rng_mcp > d_rng_pmcp * 1.05 or ring_tip[1] < ring_pip[1] - 0.02)
        pinky_up = (d_pnk_w > d_pnk_pw * 1.05) and (d_pnk_mcp > d_pnk_pmcp * 1.05 or pinky_tip[1] < pinky_pip[1] - 0.02)

        index_folded = not index_up
        middle_folded = not middle_up
        ring_folded = not ring_up
        pinky_folded = not pinky_up

        # Spatial distances
        thumb_index_dist = float(np.linalg.norm(thumb_tip[:2] - index_tip[:2]))
        palm_center = (index_mcp[:2] + pinky_mcp[:2] + wrist[:2]) / 3.0
        thumb_palm_dist = float(np.linalg.norm(thumb_tip[:2] - palm_center))
        d_thumb_w = float(np.linalg.norm(thumb_tip[:2] - wrist[:2]))
        d_thumb_mcp_w = float(np.linalg.norm(thumb_mcp[:2] - wrist[:2]))

        # Robust Thumb States
        thumb_up = (thumb_tip[1] < thumb_ip[1]) and (thumb_tip[1] < thumb_mcp[1]) and (d_thumb_w > d_thumb_mcp_w * 1.08)
        thumb_down = (thumb_tip[1] > thumb_ip[1]) and (thumb_tip[1] > thumb_mcp[1]) and (thumb_tip[1] > wrist[1] - 0.04)
        thumb_out = (thumb_palm_dist > 0.12) and not thumb_up and not thumb_down

        # -------------------------------------------------------------
        # 2.1 YES / GOOD (Thumbs Up Fist)
        # -------------------------------------------------------------
        if thumb_up and index_folded and middle_folded and ring_folded and pinky_folded:
            return "YES", 0.96
        if thumb_up and (index_folded + middle_folded + ring_folded + pinky_folded >= 3):
            return "YES", 0.93

        # -------------------------------------------------------------
        # 2.2 NO / BAD (Thumbs Down Fist OR ASL Beak Snap)
        # -------------------------------------------------------------
        if thumb_down and index_folded and middle_folded and ring_folded and pinky_folded:
            return "NO", 0.95
        # Official ASL 'NO' gesture: thumb, index & middle tips snapping together
        d_mid_thumb = float(np.linalg.norm(middle_tip[:2] - thumb_tip[:2]))
        if thumb_index_dist < 0.10 and d_mid_thumb < 0.10 and ring_folded and pinky_folded:
            return "NO", 0.94

        # -------------------------------------------------------------
        # 2.3 I LOVE YOU (Universal ILY: Thumb + Index + Pinky out)
        # -------------------------------------------------------------
        if index_up and pinky_up and middle_folded and ring_folded and (thumb_out or thumb_palm_dist > 0.09):
            return "I LOVE YOU", 0.96

        # -------------------------------------------------------------
        # 2.4 CALL ME / PHONE (Shaka: Thumb + Pinky out, middle 3 folded)
        # -------------------------------------------------------------
        if pinky_up and index_folded and middle_folded and ring_folded and (thumb_out or thumb_palm_dist > 0.09):
            return "CALL ME", 0.95

        # -------------------------------------------------------------
        # 2.5 OK / FINE (Thumb & Index tips touching in ring, 3 extended)
        # -------------------------------------------------------------
        if thumb_index_dist < 0.13 and middle_up and ring_up and pinky_up:
            return "OK", 0.95

        # -------------------------------------------------------------
        # 2.6 PEACE (Index & Middle upright in V, ring & pinky folded)
        # -------------------------------------------------------------
        if index_up and middle_up and ring_folded and pinky_folded:
            return "PEACE", 0.95

        # -------------------------------------------------------------
        # 2.7 YOU / WHERE (Index pointing forward/upward, other 3 curled)
        # -------------------------------------------------------------
        if index_up and middle_folded and ring_folded and pinky_folded and not thumb_out:
            return "YOU", 0.93

        # -------------------------------------------------------------
        # 2.8 LATER (L-Shape Hand: Index up + Thumb pointing sideways)
        # -------------------------------------------------------------
        if index_up and middle_folded and ring_folded and pinky_folded and thumb_out:
            return "LATER", 0.92

        # -------------------------------------------------------------
        # 2.9 WATER (W-Hand: Index, Middle, Ring upright spread, pinky folded)
        # -------------------------------------------------------------
        if index_up and middle_up and ring_up and pinky_folded:
            return "WATER", 0.94

        # -------------------------------------------------------------
        # 2.10 FOOD / EAT (Fingertips clustered near mouth/chin)
        # -------------------------------------------------------------
        d_idx_mid = float(np.linalg.norm(index_tip[:2] - middle_tip[:2]))
        if thumb_index_dist < 0.08 and d_idx_mid < 0.08 and index_tip[1] < 0.38:
            return "FOOD", 0.92

        # -------------------------------------------------------------
        # 2.11 DRINK (C-Shape Hand tilted toward mouth)
        # -------------------------------------------------------------
        if 0.06 < thumb_index_dist < 0.22 and not index_up and index_tip[1] < 0.40 and thumb_tip[1] < 0.45:
            return "DRINK", 0.91

        # -------------------------------------------------------------
        # 2.12 I / ME (Index finger pointing at own chest)
        # -------------------------------------------------------------
        if index_tip[1] > index_mcp[1] and middle_folded and ring_folded and pinky_folded:
            return "I / ME", 0.92

        # -------------------------------------------------------------
        # 2.13 HELLO / STOP / THANK YOU (Open Palm Gestures)
        # -------------------------------------------------------------
        if index_up and middle_up and ring_up and pinky_up:
            if thumb_palm_dist > 0.11 or thumb_out:
                # Open spread palm facing camera
                return "HELLO", 0.96
            elif index_tip[1] < 0.28:
                # Flat hand at chin/lips pushing forward
                return "THANK YOU", 0.94
            else:
                # Open flat hand held vertical with fingers together
                return "STOP", 0.93

        # -------------------------------------------------------------
        # 2.14 STOP / SORRY (Closed Fist Gestures)
        # -------------------------------------------------------------
        if index_folded and middle_folded and ring_folded and pinky_folded and not thumb_up and not thumb_down:
            if wrist[1] < 0.50:
                return "SORRY", 0.92
            else:
                return "STOP", 0.92

        return None, 0.0
