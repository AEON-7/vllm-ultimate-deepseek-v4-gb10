#!/usr/bin/env python3
import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass

import aiohttp


PROMPTS = {
    "reasoning": [
        "A train leaves city A at 60 mph heading east. Another train leaves city B at 80 mph heading west on the same track. The cities are 350 miles apart. When and where do they meet? Explain carefully.",
        "You have two ropes, each takes exactly 60 minutes to burn end-to-end but burns unevenly. Using only a lighter, measure exactly 45 minutes. Explain.",
    ],
    "math": [
        "Find all real solutions to x^3 - 6x^2 + 11x - 6 = 0. Factor and show the roots.",
        "Evaluate the integral of (2x + 3) dx from x=1 to x=4. Show the antiderivative and substitution.",
    ],
    "coding": [
        "Write a Python function fibonacci(n) that returns the n-th Fibonacci number using memoization. Include a docstring and one example call.",
        "Write a SQL query against orders(id, customer_id, total, created_at) and customers(id, name) that returns the top 5 customers by lifetime spend.",
    ],
    "prose": [
        "Write the opening 3 paragraphs of a short story set in a small coastal town where the tide has not come in for three days.",
        "Write a warm but resolute 200-word letter from a sailing captain to her crew on the eve of a long voyage.",
    ],
    "dialogue": [
        "Write a 5-turn dialogue between an old librarian and a curious child about why people still tell stories.",
        "A teacher explains recursion to a confused student using an analogy. Write a 6-turn exchange where the student gradually gets it.",
    ],
    "summary": [
        "Summarize the Big Bang theory and the main observational evidence supporting it in exactly 5 bullet points.",
        "Give a 200-word overview of the difference between supervised and unsupervised learning, with one example of each.",
    ],
}


@dataclass
class Metric:
    category: str
    ttft_ms: float = 0.0
    wall_s: float = 0.0
    output_tokens: int = 0
    text_chars: int = 0
    reasoning_chars: int = 0
    finish_reason: str = ""
    error: str = ""

    @property
    def generated_chars(self) -> int:
        return self.text_chars + self.reasoning_chars

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return max(0.0, (self.wall_s - self.ttft_ms / 1000.0) * 1000.0 / (self.output_tokens - 1))

    @property
    def decode_tok_s(self) -> float:
        return 1000.0 / self.tpot_ms if self.tpot_ms > 0 else 0.0

    @property
    def wall_tok_s(self) -> float:
        return self.output_tokens / self.wall_s if self.wall_s > 0 else 0.0

    def row(self):
        d = asdict(self)
        d.update({
            "generated_chars": self.generated_chars,
            "tpot_ms": self.tpot_ms,
            "decode_tok_s": self.decode_tok_s,
            "wall_tok_s": self.wall_tok_s,
        })
        return d


async def stream_one(session, base, model, prompt, category, max_tokens, temperature, thinking):
    m = Metric(category=category)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    start = time.perf_counter()
    first = None
    try:
        async with session.post(
            f"{base.rstrip('/')}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                m.error = f"HTTP {resp.status}: {await resp.text()}"
                m.wall_s = time.perf_counter() - start
                return m
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue
                usage = chunk.get("usage")
                if usage:
                    m.output_tokens = int(usage.get("completion_tokens") or m.output_tokens or 0)
                choice = (chunk.get("choices") or [{}])[0]
                m.finish_reason = choice.get("finish_reason") or m.finish_reason
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if text or reasoning:
                    if first is None:
                        first = time.perf_counter()
                        m.ttft_ms = (first - start) * 1000.0
                    m.text_chars += len(text)
                    m.reasoning_chars += len(reasoning)
    except Exception as e:
        m.error = repr(e)
    m.wall_s = time.perf_counter() - start
    if m.output_tokens == 0 and m.generated_chars:
        m.output_tokens = max(1, m.generated_chars // 4)
    return m


def stat(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
    }


def summarize(rows):
    ok = [r for r in rows if not r.error and r.output_tokens > 1]
    by_cat = {}
    for cat in PROMPTS:
        cr = [r for r in ok if r.category == cat]
        by_cat[cat] = {
            "n": len(cr),
            "ttft_ms": stat([r.ttft_ms for r in cr]),
            "tpot_ms": stat([r.tpot_ms for r in cr]),
            "decode_tok_s": stat([r.decode_tok_s for r in cr]),
            "wall_tok_s": stat([r.wall_tok_s for r in cr]),
            "output_tokens": stat([r.output_tokens for r in cr]),
        }
    return {
        "ok": len(ok),
        "errors": [r.row() for r in rows if r.error],
        "overall": {
            "ttft_ms": stat([r.ttft_ms for r in ok]),
            "tpot_ms": stat([r.tpot_ms for r in ok]),
            "decode_tok_s": stat([r.decode_tok_s for r in ok]),
            "wall_tok_s": stat([r.wall_tok_s for r in ok]),
            "output_tokens": stat([r.output_tokens for r in ok]),
        },
        "by_category": by_cat,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="aeon-fast")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", choices=["true", "false", "unset"], default="false")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    thinking = None if args.thinking == "unset" else args.thinking == "true"

    async with aiohttp.ClientSession() as session:
        await stream_one(session, args.base, args.model, "Reply with only: ready", "warmup", 16, args.temperature, thinking)

        single = []
        for cat, prompts in PROMPTS.items():
            for prompt in prompts:
                m = await stream_one(session, args.base, args.model, prompt, cat, args.max_tokens, args.temperature, thinking)
                single.append(m)
                print(f"single {cat:9s} ttft={m.ttft_ms:7.1f}ms tpot={m.tpot_ms:6.2f}ms decode={m.decode_tok_s:7.2f} tok/s out={m.output_tokens:4d} err={m.error[:80]}")

        concurrent = []
        cats = list(PROMPTS)
        for r in range(args.rounds):
            picks = []
            for i in range(args.concurrency):
                cat = cats[(r * args.concurrency + i) % len(cats)]
                prompt = PROMPTS[cat][r % len(PROMPTS[cat])]
                picks.append((cat, prompt))
            t0 = time.perf_counter()
            rows = await asyncio.gather(*[
                stream_one(session, args.base, args.model, prompt, cat, args.max_tokens, args.temperature, thinking)
                for cat, prompt in picks
            ])
            wall = time.perf_counter() - t0
            toks = sum(m.output_tokens for m in rows)
            agg = toks / wall if wall > 0 else 0.0
            concurrent.append({"round": r + 1, "wall_s": wall, "tokens": toks, "agg_tok_s": agg, "rows": [m.row() for m in rows]})
            print(f"concurrent round={r+1} c={args.concurrency} wall={wall:.2f}s tokens={toks} agg={agg:.2f} tok/s")

    payload = {
        "label": args.label,
        "base": args.base,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "single": summarize(single),
        "concurrent": concurrent,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["single"]["overall"], indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
