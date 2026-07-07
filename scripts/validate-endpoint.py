#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request


def post_json(url, payload, timeout=300):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer sk-local"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def get_json(url, timeout=30):
    req = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer sk-local"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    print("Checking /v1/models ...")
    models = get_json(f"{base}/models")
    ids = [item.get("id") for item in models.get("data", [])]
    print("Models:", ids)
    if args.model not in ids:
        raise SystemExit(f"Expected {args.model!r} in model list")

    prompts = [
        ("natural", "In two concise paragraphs, explain what this server is and why a tester should care."),
        ("coding", "Write a small Python function that computes rolling averages over a list."),
        ("reasoning", "A test rig has two nodes and one node loses RoCE connectivity. What symptoms should the tester expect?"),
    ]

    for category, prompt in prompts:
        print(f"\nRunning {category} smoke ...")
        started = time.perf_counter()
        data = post_json(
            f"{base}/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.2,
            },
        )
        elapsed = time.perf_counter() - started
        choice = data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        usage = data.get("usage", {})
        print("finish_reason:", choice.get("finish_reason"))
        print("elapsed_sec:", round(elapsed, 3))
        print("usage:", usage)
        print("preview:", content[:300].replace("\n", " "))
        if not content.strip():
            raise SystemExit(f"{category} smoke returned empty content")

    print("\nEndpoint validation passed.")


if __name__ == "__main__":
    main()
