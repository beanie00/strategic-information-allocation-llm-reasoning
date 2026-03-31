"""Utilities for reproducing Section 3.4.3 test-time controls.

This module centralizes the paper-aligned epistemic token list, the
few-shot prompting examples from Appendix G.2.1, token counting, and
test-time control helpers used by the dedicated evaluation script.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple


EPISTEMIC_TOKEN_STRINGS: List[str] = [
    "wait",
    "hmm",
    "perhaps",
    "maybe",
    "actually",
    "alternatively",
    "seems",
    "might",
    "likely",
    "guess",
    "sure",
    "correct",
    "check",
]

_SURFACE_PREFIXES: Sequence[str] = ("", " ", "\n", "\n ", "\t")
_SURFACE_SUFFIXES: Sequence[str] = ("", ".", ",", ":", ";", "!", "?")

_EXAMPLE_1 = """[Example 1] Okay, so I need to figure out what value of x (which is a two-digit number) will make m = n4 + x a composite number, where n is a natural number.
Hmm, let's break this down step by step.

First, let me recall that a composite number is a positive integer that has at least one positive divisor other than 1 and itself. So, m has to be divisible by some number other than 1 and m. My goal is to find an x such that no matter what natural number n we choose, m will always be composite? Wait, actually, the problem says "what value of x will make m a composite number." Hmm, maybe I need to interpret it as for any n, m is composite? Or is there a specific n? Wait, the problem says "where n is a natural number and x is a two-digit positive integer." Maybe it's asking for an x such that for all natural numbers n, m is composite? Or maybe just that there exists some n where m is composite? The wording is a bit ambiguous.

Wait, let me check the problem again: "Given m = n4 + x, where n is a natural number and x is a two-digit positive integer, what value of x will make m a composite number?" Hmm, maybe it's asking for an x such that for all natural numbers n, m is composite? That would make sense, but the problem might not specify "for all n," so maybe it's asking for an x such that there exists some n where m is composite. But since x is fixed and n can be any natural number, perhaps the question is to find an x such that for every n, m is composite. Otherwise, if it's just for some n, then any x would work because you can choose n such that m is composite. For example, if x is 10, then pick n = 2: 16 + 10 = 26, which is composite. But maybe the question wants an x such that no matter what n you choose, m is composite. That's a more challenging problem.

Given that x is a two-digit number, the problem might be looking for an x such that n4 + x is always composite, regardless of n. That seems more likely. Let me confirm. The problem says "what value of x will make m a composite number" - so perhaps for any n, m must be composite. So, the x has to be such that adding it to any fourth power of a natural number results in a composite number. That would be the case. Let me think how to approach this.

First, note that n4 is a perfect fourth power. For n >= 1, n4 is 1, 16, 81, 256, 625, .... Adding x (a two-digit number between 10 and 99 inclusive) to these numbers must result in a composite number. So, the challenge is to choose x such that for every n, n4 + x is composite.

Alternatively, maybe the question is asking for an x such that there exists some n where m is composite. But since x is two-digit, and m = n4 + x, unless n is very small, m will be large, so it's likely composite. But the problem might want a specific x that works for all n. Let me see.

Wait, the problem is a bit ambiguous. Since the user is probably expecting a specific answer, maybe from an Olympiad problem, perhaps there is a standard answer here. Let me think of possible x's. Maybe x is 6? But x has to be two-digit. Wait, two-digit, so between 10 and 99. Hmm.

Alternatively, maybe x is such that n4 + x is always even? Then it would be composite unless it's 2. But n4 is even when n is even, and odd when n is odd. So if x is even, then n even: even + even = even, which is composite (as long as it's not 2). n odd: odd + even = odd. So that might not help. Alternatively, if x is odd, then n even: even + odd = odd, and n odd: odd + odd = even. Hmm.

Alternatively, maybe x is a multiple of some number. Let me think. Suppose we can choose x such that n4 + x is divisible by some prime p for all n. For instance, if x is congruent to 0 (mod p) when n4 is congruent to -x (mod p). Wait, maybe that's too vague. ......"""

_EXAMPLE_2 = """[Example 2]
Okay, so I have this geometry problem here. Let me try to parse it step by step. The problem says:

In triangle ABC, side AC is the largest. Points M and N on side AC are such that AM = AB and CN = CB. It is known that angle NBM is three times smaller than angle ABC. Find angle ABC.

Hmm, first, let me draw a rough sketch of triangle ABC. Since AC is the largest side, that means the angle opposite to AC, which is angle B, should be the largest angle in the triangle. Wait, actually, in a triangle, the largest side is opposite the largest angle. So if AC is the largest side, then angle B is the largest angle. That might be useful later.

Now, points M and N are on side AC. Let me note that. So AC is the base here, and M and N are somewhere along that base. The problem states that AM = AB and CN = CB. So starting from point A, moving along AC, we mark point M such that the length from A to M is equal to the length of AB. Similarly, starting from point C, moving towards A along AC, we mark point N such that the length from C to N is equal to CB.

Also, angle NBM is one third of angle ABC. The question is to find the measure of angle ABC.

Hmm, okay. Let me try to label all this on my mental diagram. Let me think of triangle ABC with AC as the base. Let me denote AB as c, BC as a, and AC as b. Wait, but since AC is the largest side, so b > a and b > c.

Wait, actually, standard notation is usually: in triangle ABC, side a is BC, side b is AC, and side c is AB. Wait, maybe that's conflicting here. Let me confirm: in standard triangle notation, side a is opposite angle A, side b opposite angle B, and side c opposite angle C. Wait, so if AC is the largest side, then AC is side b, which is opposite angle B. Wait, that would mean angle B is opposite the largest side, so angle B is indeed the largest angle. So that's consistent.

But perhaps to avoid confusion, let me just assign variables differently. Let me denote the sides as follows: Let AB = x, BC = y, and AC = z. Since AC is the largest, z > x and z > y. ......"""


def get_induction_few_shot_prompt(num_shots: int) -> str:
    if num_shots not in (1, 2):
        raise ValueError(f"num_shots must be 1 or 2, got {num_shots}")

    intro = (
        'Below are examples of a solution. In this way, you can express '
        'uncertainty by using phrases such as "hmm" or "wait."\n\n'
    )
    if num_shots == 1:
        return intro + _EXAMPLE_1 + "\n\n"
    return intro + _EXAMPLE_1 + "\n\n" + _EXAMPLE_2 + "\n\n"


def build_epistemic_count_pattern(token: str) -> re.Pattern[str]:
    return re.compile(rf"(?i)(?<![A-Za-z]){re.escape(token)}(?![A-Za-z])")


def count_epistemic_tokens(text: str, token_strings: Iterable[str] | None = None) -> Tuple[int, Dict[str, int]]:
    tokens = list(token_strings or EPISTEMIC_TOKEN_STRINGS)
    counts: Dict[str, int] = {}
    total = 0
    for token in tokens:
        count = len(build_epistemic_count_pattern(token).findall(text))
        counts[token] = count
        total += count
    return total, counts


def expand_token_surfaces(token: str) -> Set[str]:
    base_forms = {token.lower(), token.capitalize(), token.upper()}
    surfaces: Set[str] = set()
    for prefix in _SURFACE_PREFIXES:
        for form in base_forms:
            surfaces.add(prefix + form)
            for suffix in _SURFACE_SUFFIXES:
                surfaces.add(prefix + form + suffix)
    return surfaces


def build_epistemic_logit_bias(tokenizer, token_strings: Iterable[str] | None = None, bias: float = -100.0):
    token_strings = list(token_strings or EPISTEMIC_TOKEN_STRINGS)
    token_ids: Set[int] = set()
    matched_surfaces: Dict[str, List[str]] = {}

    for token in token_strings:
        matched_surfaces[token] = []
        for surface in sorted(expand_token_surfaces(token)):
            tokenized = tokenizer.encode(surface, add_special_tokens=False)
            if len(tokenized) == 1:
                token_ids.add(tokenized[0])
                matched_surfaces[token].append(surface)

    return {token_id: bias for token_id in sorted(token_ids)}, matched_surfaces


def build_wait_intervention_message(answer: str) -> str:
    cleaned = (answer or "").strip()
    if cleaned:
        return f"Wait {cleaned} is not the answer. Please think again.\n"
    return "Wait that is not the answer. Please think again.\n"


def resolve_fixed_prefix(model_name_or_path: str, control_mode: str, user_prefix: str = "", disable_auto_prefix: bool = False) -> str:
    if user_prefix:
        return user_prefix
    if disable_auto_prefix:
        return ""
    if control_mode == "induce" and "Qwen3-14B-Base" in model_name_or_path:
        return "Okay, so I "
    return ""
