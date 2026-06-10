"""Post-process the Judge feed's scenario descriptions into a fixed template.

The local 7B synthesis model does not reliably obey a strict output template
(it rambles into verbose markdown). This standalone pass reformats each surfaced
scenario into a uniform five-field structure using a stronger HOSTED instruct
model — text-only, so it needs no local GPU. Run it as the last step after any
`verity_hardnovel.py` run; it rewrites `outputs/waymo/judge_scored.json` in place.

    set -a && . ./.env && set +a
    .venv/bin/python -m drivers.verity_format_descriptions
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

FEED = Path("outputs/waymo/judge_scored.json")
HOSTED_URL = "https://integrate.api.nvidia.com/v1"
HOSTED_MODEL = os.environ.get("FORMAT_MODEL_ID", "meta/llama-3.1-70b-instruct")

TEMPLATE = (
    "Reformat the autonomous-vehicle scenario description below into EXACTLY this "
    "five-line template. Preserve the content; only restructure it. Output the five "
    "labeled lines and nothing else — no markdown, no bold, no extra headers, no text "
    "before or after, English only:\n\n"
    "Scenario: <a short title, 5-8 words>\n"
    "Setting: <road type and lane layout, intersection/signals, location, time of day, "
    "weather, lighting — one sentence>\n"
    "Agents: <the vehicles, pedestrians, and cyclists present, each with position, motion, "
    "and intent — one to two sentences>\n"
    "Sequence: <what unfolds over the scene — two to three sentences>\n"
    "Challenge: <one sentence on what makes this operationally hard for the automated driver>\n\n"
    "Description to reformat:\n{desc}"
)


def _reformat(client: OpenAI, desc: str) -> str:
    r = client.chat.completions.create(
        model=HOSTED_MODEL, max_tokens=400, temperature=0.2,
        messages=[{"role": "user", "content": TEMPLATE.format(desc=desc)}])
    return (r.choices[0].message.content or "").strip()


def main() -> None:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        sys.exit("NVIDIA_API_KEY not set (needed for the hosted format model)")
    client = OpenAI(api_key=key, base_url=HOSTED_URL)

    feed = json.loads(FEED.read_text())
    targets = [(i, e) for i, e in enumerate(feed)
               if (e.get("plausibility_justification") or "").strip()]
    print(f"[fmt] reformatting {len(targets)} descriptions via {HOSTED_MODEL}...", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=6) as pool:
        fut = {pool.submit(_reformat, client, feed[i]["plausibility_justification"]): i
               for i, _ in targets}
        for f in as_completed(fut):
            i = fut[f]
            try:
                feed[i]["plausibility_justification"] = f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[fmt]   scene {feed[i].get('composition_id')}: {exc}", file=sys.stderr)

    FEED.write_text(json.dumps(feed, indent=2))
    print(f"[fmt] DONE — rewrote {FEED}")
    print("\n--- sample ---\n" + feed[0]["plausibility_justification"][:500])


if __name__ == "__main__":
    main()
