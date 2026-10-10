"""Optional fuzzy matching used to suggest likely words from an external vocabulary."""

from __future__ import annotations

from typing import Iterable, List


class FuzzyMatcher:
    def __init__(self, vocabulary: Iterable[str] | None = None) -> None:
        self.vocabulary = sorted({(word or "").strip().upper() for word in (vocabulary or []) if (word or "").strip()})

    def _distance(self, a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            prev = dp[0]
            dp[0] = i
            for j, cb in enumerate(b, start=1):
                curr = min(
                    dp[j] + 1,
                    dp[j - 1] + 1,
                    prev + (ca != cb),
                )
                prev = dp[j]
                dp[j] = curr
        return dp[-1]

    def suggest(self, detected: str, limit: int = 5, min_similarity: float = 0.6) -> List[str]:
        target = (detected or "").strip().upper()
        if not target:
            return []

        candidates = []
        for item in self.vocabulary:
            if not item:
                continue
            dist = self._distance(target, item)
            max_len = max(len(target), len(item))
            similarity = 1.0 - (dist / max_len) if max_len else 1.0
            if similarity >= min_similarity:
                candidates.append((similarity, item))

        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        return [item for _, item in candidates[:limit]]
