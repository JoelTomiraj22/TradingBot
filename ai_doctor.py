"""
AI Provider Doctor — diagnoses every configured AI provider/key/endpoint.

Run: python ai_doctor.py

For each provider it makes a minimal real request and reports exactly what
works, what fails, and how to fix it. Costs a few tokens total.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GREEN, RED, YELLOW, DIM, BOLD, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"
PING = "Reply with the single word: OK"


def mask(key: str) -> str:
    return f"{key[:10]}...{key[-4:]}" if len(key) > 16 else "(short key)"


def verdict(ok: bool, label: str, detail: str, hint: str = ""):
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}")
    if detail:
        print(f"         {DIM}{detail}{RESET}")
    if hint and not ok:
        print(f"         {YELLOW}fix: {hint}{RESET}")


def hint_for(status: int, body: str, provider: str) -> str:
    if status == 401:
        return "key invalid/expired — regenerate it"
    if status == 402:
        return "model needs paid credits on this account"
    if status == 404 and "page not found" in body.lower():
        return ("gateway rejected the route for this key — on build.nvidia.com check the key is "
                "'Public API Endpoints' enabled, or regenerate it")
    if status == 404:
        return "model id not available to this key — check the exact model name on the provider's catalog"
    if status == 429:
        if provider == "openrouter":
            return "free pool rate-limited (≈20 req/min, 50-200/day) — wait or fund the account"
        return "quota/rate limit exhausted — wait for reset or use another key"
    if status == 400:
        return "request rejected — likely wrong endpoint for this key type"
    return ""


def test_nvidia(name: str, env: str, models: list):
    print(f"\n{BOLD}{name}{RESET}")
    keys = []
    for e in (env, "NVIDIA_API_KEY"):
        k = os.getenv(e, "").strip()
        if k and k not in [x[1] for x in keys]:
            keys.append((e, k))
    if not keys:
        verdict(False, f"{env}", "no key set", f"add {env}=... to .env")
        return
    for model in models:
        for env_name, key in keys:
            for url in ("https://integrate.api.nvidia.com/v1/chat/completions",
                        "https://ai.api.nvidia.com/v1/chat/completions"):
                host = url.split("/")[2]
                try:
                    r = requests.post(url, json={"model": model,
                                                 "messages": [{"role": "user", "content": PING}],
                                                 "max_tokens": 5},
                                      headers={"Authorization": f"Bearer {key}"}, timeout=30)
                    ok = r.status_code == 200
                    verdict(ok, f"{model} | {env_name} ({mask(key)}) @ {host}",
                            f"HTTP {r.status_code}: {r.text[:110]}" if not ok else "responded",
                            hint_for(r.status_code, r.text, "nvidia"))
                    if ok:
                        return
                except Exception as e:
                    verdict(False, f"{model} | {env_name} @ {host}", str(e)[:110],
                            "network/timeout — check connectivity")


def test_gemini():
    print(f"\n{BOLD}Gemini 2.5 Pro / Flash{RESET}")
    pro_keys = [k.strip() for k in os.getenv("GEMINI_PRO_API_KEYS", "").split(",") if k.strip()]
    flash_key = os.getenv("GEMINI_API_KEY", "").strip()
    keys = [("GEMINI_PRO_API_KEYS", k) for k in pro_keys]
    if flash_key:
        keys.append(("GEMINI_API_KEY", flash_key))
    if not keys:
        verdict(False, "GEMINI keys", "none set", "add GEMINI_PRO_API_KEYS / GEMINI_API_KEY to .env")
        return
    for env_name, key in keys:
        is_express = key.startswith("AQ.")
        kind = "Vertex Express (AQ.) key" if is_express else "AI Studio key"
        endpoints = [
            ("aiplatform.googleapis.com (Vertex Express)",
             "https://aiplatform.googleapis.com/v1/publishers/google/models/{m}:generateContent"),
            ("generativelanguage.googleapis.com (AI Studio)",
             "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"),
        ]
        if not is_express:
            endpoints.reverse()
        body = {"contents": [{"parts": [{"text": PING}]}],
                "generationConfig": {"maxOutputTokens": 5}}
        for model in ("gemini-2.5-pro", "gemini-2.5-flash"):
            passed = False
            for ep_label, ep in endpoints:
                for auth_label, kwargs in (
                    ("header", {"headers": {"X-goog-api-key": key}}),
                    ("?key=", {"params": {"key": key}}),
                ):
                    try:
                        r = requests.post(ep.format(m=model), json=body, timeout=30, **kwargs)
                        ok = r.status_code == 200
                        verdict(ok, f"{env_name} ({mask(key)}, {kind}) {model} @ {ep_label} [{auth_label}]",
                                f"HTTP {r.status_code}: {r.text[:110]}" if not ok else "responded",
                                hint_for(r.status_code, r.text, "gemini"))
                        if ok:
                            passed = True
                            break
                    except Exception as e:
                        verdict(False, f"{env_name} {model} @ {ep_label} [{auth_label}]", str(e)[:110], "network/timeout")
                if passed:
                    break
            if passed:
                break


def test_openrouter():
    print(f"\n{BOLD}OpenRouter free pool{RESET}")
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        verdict(False, "OPENROUTER_API_KEY", "no key set", "add OPENROUTER_API_KEY to .env")
        return
    try:
        r = requests.get("https://openrouter.ai/api/v1/key",
                         headers={"Authorization": f"Bearer {key}"}, timeout=20)
        verdict(r.status_code == 200, f"key auth ({mask(key)})",
                f"HTTP {r.status_code}: {r.text[:110]}" if r.status_code != 200 else "key valid",
                hint_for(r.status_code, r.text, "openrouter"))
    except Exception as e:
        verdict(False, "key auth", str(e)[:110], "network/timeout")
        return
    for model in ("deepseek/deepseek-r1:free", "deepseek/deepseek-chat-v3-0324:free",
                  "meta-llama/llama-3.3-70b-instruct:free", "openrouter/free"):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              json={"model": model,
                                    "messages": [{"role": "user", "content": PING}],
                                    "max_tokens": 5},
                              headers={"Authorization": f"Bearer {key}"}, timeout=45)
            ok = r.status_code == 200
            verdict(ok, model,
                    f"HTTP {r.status_code}: {r.text[:110]}" if not ok else "responded",
                    hint_for(r.status_code, r.text, "openrouter"))
            if ok:
                break
        except Exception as e:
            verdict(False, model, str(e)[:110], "network/timeout")


def test_groq():
    print(f"\n{BOLD}Groq{RESET}")
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        verdict(False, "GROQ_API_KEY", "no key set", "add GROQ_API_KEY to .env")
        return
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          json={"model": "llama-3.3-70b-versatile",
                                "messages": [{"role": "user", "content": PING}],
                                "max_tokens": 5},
                          headers={"Authorization": f"Bearer {key}"}, timeout=30)
        ok = r.status_code == 200
        verdict(ok, f"llama-3.3-70b-versatile ({mask(key)})",
                f"HTTP {r.status_code}: {r.text[:110]}" if not ok else "responded",
                hint_for(r.status_code, r.text, "groq"))
    except Exception as e:
        verdict(False, "groq", str(e)[:110], "network/timeout")


if __name__ == "__main__":
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  AI PROVIDER DOCTOR — testing every key & endpoint{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    test_nvidia("DeepSeek V4 Pro (NVIDIA)", "NVIDIA_DEEPSEEK_API_KEY",
                ["deepseek-ai/deepseek-v4-pro", "meta/llama-3.3-70b-instruct",
                 "nvidia/llama-3.1-nemotron-ultra-253b-v1"])
    test_nvidia("Qwen 3.5 397B (NVIDIA)", "NVIDIA_QWEN_API_KEY",
                ["qwen/qwen3.5-397b-a17b", "nvidia/llama-3.1-nemotron-ultra-253b-v1",
                 "meta/llama-3.3-70b-instruct"])
    test_gemini()
    test_openrouter()
    test_groq()
    print(f"\n{DIM}A provider needs only ONE passing line to be usable by the bot.{RESET}")
    print(f"{DIM}The bot tries the same key/endpoint combinations automatically.{RESET}\n")
