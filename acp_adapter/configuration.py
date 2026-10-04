"""Provider-owned readback on the authenticated ACP session channel."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any
from acp.exceptions import RequestError


def _session(manager, params):
    if set(params) - {"sessionId", "generation", "instructions", "nativeInstructions"} or not isinstance(params.get("sessionId"), str):
        raise RequestError.invalid_params({"reason": "An exact sessionId is required"})
    state = manager.get_session(params["sessionId"])
    if state is None:
        raise RequestError.invalid_params({"reason": "Native session not found"})
    return state


def bind_agent_instructions(state, agent):
    prior = getattr(agent, "ephemeral_system_prompt", None)
    state.configuration_base_prompt = prior
    agent.ephemeral_system_prompt = "\n\n".join(part for part in [prior, state.client_instructions, state.native_instructions] if isinstance(part, str) and part)


def _bounded_skill_file_digest(skill_file):
    if skill_file.is_symlink() or skill_file.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Skill file exceeds the byte limit or is a link")
    with skill_file.open("rb") as stream:
        content = stream.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("Skill file exceeds the byte limit")
    return hashlib.sha256(content).hexdigest()


def configure_session(manager, params: dict[str, Any]) -> dict[str, Any]:
    state = _session(manager, params)
    generation, instructions = params.get("generation"), params.get("instructions")
    native_instructions = params.get("nativeInstructions")
    if native_instructions is not None and (not isinstance(native_instructions, str) or len(native_instructions.encode("utf-8")) > 1_048_576):
        raise RequestError.invalid_params({"reason": "Native instructions must be bounded UTF-8 text"})
    if type(generation) is not int or generation < 0 or not isinstance(instructions, str) or len(instructions.encode("utf-8")) > 1_048_576:
        raise RequestError.invalid_params({"reason": "Configuration requires a nonnegative generation and bounded instructions"})
    with state.runtime_lock:
        if state.configuration_generation is not None and generation < state.configuration_generation:
            raise RequestError.invalid_params({"reason": "Native configuration generation cannot move backwards"})
        same = state.configuration_generation == generation and state.client_instructions == instructions and state.native_instructions == native_instructions
        if not same and (state.is_running or state.command_op or state.prompt_started_in_instance):
            raise RequestError.invalid_params({"reason": "Native configuration is immutable after the first prompt"})
        if state.configuration_bound_in_instance and not same:
            raise RequestError.invalid_params({"reason": "Native session configuration is already bound"})
        if not same:
            prior = state.configuration_base_prompt if state.configuration_generation is not None else getattr(state.agent, "ephemeral_system_prompt", None)
            state.configuration_base_prompt = prior
            state.agent.ephemeral_system_prompt = "\n\n".join(part for part in [prior, instructions, native_instructions] if isinstance(part, str) and part)
            state.configuration_generation, state.client_instructions, state.native_instructions = generation, instructions, native_instructions
        state.configuration_bound_in_instance = True
    manager.save_session(state.session_id)
    return read_session_configuration(manager, {"sessionId": state.session_id})


def skill_content_digest(directory: Path) -> str:
    digest = hashlib.sha256(b"hermes-skill-content-v1\0")
    files = []
    total = 0
    entries = 0
    for candidate in directory.rglob("*"):
        entries += 1
        if entries > 10000:
            raise ValueError("Skill content exceeds the entry limit")
        relative = candidate.relative_to(directory)
        if candidate.is_symlink() or len(relative.parts) > 24:
            raise ValueError("Skill content includes an unsupported link or depth")
        if candidate.is_file():
            files.append((relative.as_posix().encode("utf-8"), candidate))
        if len(files) > 1000:
            raise ValueError("Skill content exceeds the file limit")
    for relative, candidate in sorted(files):
        if candidate.stat().st_size > 10 * 1024 * 1024 - total:
            raise ValueError("Skill content exceeds the byte limit")
        with candidate.open("rb") as stream:
            content = stream.read(10 * 1024 * 1024 - total + 1)
        total += len(content)
        if total > 10 * 1024 * 1024:
            raise ValueError("Skill content exceeds the byte limit")
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">Q", len(content)))
        digest.update(content)
    return digest.hexdigest()


def _skills(state):
    from agent.runtime_cwd import set_session_cwd
    from agent.skill_utils import get_plugin_skills_dirs
    from tools.skills_tool import _find_all_skills, _locate_skill, _skill_search_dirs, _under_any
    set_session_cwd(state.cwd)
    project_dirs, all_dirs, _ = _skill_search_dirs()
    plugin_dirs = get_plugin_skills_dirs()
    result = []
    for metadata in _find_all_skills():
        error, directory, skill_file = _locate_skill(metadata["name"], None, project_dirs, all_dirs)
        if error is not None or skill_file is None:
            raise ValueError("Native selected skill cannot be resolved unambiguously")
        origin = "project" if _under_any(skill_file, project_dirs) else "plugin" if _under_any(skill_file, plugin_dirs) else "user"
        result.append({"name": metadata["name"], "origin": origin, "filePath": str(skill_file), "skillFileDigest": _bounded_skill_file_digest(skill_file), "contentDigest": skill_content_digest(directory)})
    return sorted(result, key=lambda skill: (skill["name"], skill["filePath"]))


def _validate_conversation_value(value, budget, depth=0):
    budget[0] += 1
    if depth > 32 or budget[0] > 100000:
        raise ValueError("Native conversation exceeds the structural limit")
    if isinstance(value, str):
        if len(value) > 10 * 1024 * 1024:
            raise ValueError("Native conversation exceeds the byte limit")
        budget[1] += len(value.encode("utf-8"))
        if budget[1] > 10 * 1024 * 1024:
            raise ValueError("Native conversation exceeds the byte limit")
    elif isinstance(value, dict):
        for key, nested in value.items():
            _validate_conversation_value(key, budget, depth + 1)
            _validate_conversation_value(nested, budget, depth + 1)
    elif isinstance(value, list):
        for nested in value:
            _validate_conversation_value(nested, budget, depth + 1)
    elif value is not None and type(value) not in (int, float, bool):
        raise ValueError("Native conversation contains unsupported content")


def native_conversation_readback(history):
    if len(history) > 10000:
        return {"status": "unavailable", "reason": "native_conversation_readback_limit"}
    selected = [{"role": message.get("role"), "content": message.get("content")} for message in history]
    digest = hashlib.sha256()
    total = 0
    try:
        _validate_conversation_value(selected, [0, 0])
        for chunk in json.JSONEncoder(sort_keys=True, ensure_ascii=False, separators=(",", ":")).iterencode(selected):
            if len(chunk) > 10 * 1024 * 1024:
                raise ValueError("Native conversation exceeds the byte limit")
            encoded = chunk.encode("utf-8")
            total += len(encoded)
            if total > 10 * 1024 * 1024:
                raise ValueError("Native conversation exceeds the byte limit")
            digest.update(encoded)
    except (ValueError, TypeError, RecursionError):
        return {"status": "unavailable", "reason": "native_conversation_readback_limit"}
    return {"status": "observed", "protocol": "hermes-acp", "messageCount": len(selected), "historyDigest": digest.hexdigest()}


def read_session_configuration(manager, params: dict[str, Any]) -> dict[str, Any]:
    if set(params) != {"sessionId"}:
        raise RequestError.invalid_params({"reason": "Readback accepts only the exact sessionId"})
    state = _session(manager, params)
    with state.runtime_lock:
        generation, instructions = state.configuration_generation, state.client_instructions
        native_instructions = state.native_instructions
        tool_names = sorted(getattr(state.agent, "valid_tool_names", set()))
        configured_mcp = dict(state.acp_mcp_configs)
        conversation = native_conversation_readback(state.history)
        expected_prompt = "\n\n".join(part for part in [state.configuration_base_prompt, instructions, native_instructions] if isinstance(part, str) and part)
        effective_instructions_match = getattr(state.agent, "ephemeral_system_prompt", None) == expected_prompt
    from tools.mcp_tool_discovery import get_mcp_configuration_observations
    try:
        skills = {"status": "observed", "protocol": "hermes-acp", "skills": _skills(state)}
    except (OSError, ValueError):
        skills = {"status": "unavailable", "reason": "native_skill_content_readback_failed"}
    return {"sessionId": state.session_id, "generation": generation,
            "instructions": {"status": "unavailable", "reason": "native_instructions_not_bound"} if instructions is None or not effective_instructions_match else {"status": "observed", "protocol": "hermes-acp", "instructionsDigest": hashlib.sha256(instructions.encode("utf-8")).hexdigest()},
            "nativeInstructions": {"status": "unavailable", "reason": "native_instructions_not_bound"} if native_instructions is None or not effective_instructions_match else {"status": "observed", "protocol": "hermes-acp", "nativeInstructionsDigest": hashlib.sha256(native_instructions.encode("utf-8")).hexdigest()},
            "skills": skills, "nativeConversation": conversation, "nativeToolNames": tool_names,
            "nativeMcp": {"status": "observed", "protocol": "hermes-acp", "servers": get_mcp_configuration_observations(configured_mcp)}}
