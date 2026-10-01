"""Provider-agnostic LLM client factory.

Reads config/agents.yaml and hands each agent a `complete(system, user) -> str`
callable, sourced from whichever provider/model that agent is configured for.

Anthropic is called through its native SDK. Every other provider (OpenAI,
NVIDIA NIM, a local Ollama server, Together.ai, ...) is called through the
OpenAI-compatible chat completions API by pointing the `openai` SDK's
`base_url` at that provider - so adding a new provider is a config entry,
not new code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AgentLLMConfig:
    use_llm: bool
    tier: str
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.3
    max_tokens: int = 2000


class LLMConfigError(RuntimeError):
    pass


def load_config(path: str = "config/agents.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_agent_config(agent_name: str, config: dict | None = None) -> AgentLLMConfig:
    config = config or load_config()
    agents = config.get("agents", {})
    if agent_name not in agents:
        raise LLMConfigError(f"No config entry for agent '{agent_name}' in agents.yaml")
    raw = agents[agent_name]
    return AgentLLMConfig(
        use_llm=raw.get("use_llm", False),
        tier=raw.get("tier", "none"),
        provider=raw.get("provider"),
        model=raw.get("model"),
        temperature=raw.get("temperature", 0.3),
        max_tokens=raw.get("max_tokens", 2000),
    )


def _resolve_provider(provider_name: str, config: dict) -> dict:
    providers = config.get("providers", {})
    if provider_name not in providers:
        raise LLMConfigError(f"No provider '{provider_name}' defined in agents.yaml")
    p = providers[provider_name]
    api_key = os.environ.get(p["api_key_env"], "")
    if not api_key and p["kind"] == "anthropic":
        raise LLMConfigError(
            f"Provider '{provider_name}' requires env var '{p['api_key_env']}' to be set"
        )
    return {"kind": p["kind"], "base_url": p["base_url"], "api_key": api_key}


def build_llm_client(agent_name: str, config: dict | None = None) -> Callable[[str, str], str]:
    """Returns a `complete(system_prompt, user_prompt) -> str` function
    configured for the given agent, per config/agents.yaml."""
    config = config or load_config()
    agent_cfg = get_agent_config(agent_name, config)

    if not agent_cfg.use_llm:
        raise LLMConfigError(f"Agent '{agent_name}' is configured with use_llm: false")

    provider = _resolve_provider(agent_cfg.provider, config)

    if provider["kind"] == "anthropic":
        return _anthropic_client(provider, agent_cfg)
    elif provider["kind"] == "openai_compatible":
        return _openai_compatible_client(provider, agent_cfg)
    else:
        raise LLMConfigError(f"Unknown provider kind '{provider['kind']}'")


def _anthropic_client(provider: dict, agent_cfg: AgentLLMConfig) -> Callable[[str, str], str]:
    import anthropic

    client = anthropic.Anthropic(api_key=provider["api_key"])

    def complete(system_prompt: str, user_prompt: str) -> str:
        resp = client.messages.create(
            model=agent_cfg.model,
            max_tokens=agent_cfg.max_tokens,
            temperature=agent_cfg.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    return complete


def _openai_compatible_client(provider: dict, agent_cfg: AgentLLMConfig) -> Callable[[str, str], str]:
    from openai import OpenAI

    client = OpenAI(api_key=provider["api_key"] or "not-needed", base_url=provider["base_url"])

    def complete(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=agent_cfg.model,
            temperature=agent_cfg.temperature,
            max_tokens=agent_cfg.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    return complete
