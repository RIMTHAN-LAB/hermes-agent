import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from acp.exceptions import RequestError
from acp_adapter.configuration import configure_session, read_session_configuration, skill_content_digest
from acp_adapter.session import SessionManager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.skills_tool import skill_view, skills_list


def write_skill(root: Path, name: str, body: str):
    directory = root / name
    (directory / "references").mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Readback fixture\n---\n{body}\n")
    (directory / "references" / "detail.md").write_text(body + " reference\n")
    return directory


def make_home(root: Path, marker: str):
    root.mkdir()
    package_root = root / ".factory-managed" / "package"
    package_root.mkdir(parents=True)
    directory = write_skill(package_root, "package-skill", marker)
    (root / "config.yaml").write_text(json.dumps({"skills": {"plugin_dirs": [str(package_root)]}}))
    return directory


def test_native_selected_skill_readback_follows_profile_a_b_a_and_hashes_references(tmp_path):
    homes = [tmp_path / "a", tmp_path / "b"]
    directories = [make_home(home, marker) for home, marker in zip(homes, ["revision-a", "revision-b"])]
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="fixture", valid_tool_names={"skill_view"}, ephemeral_system_prompt="existing"))
    observed = []
    for index in [0, 1, 0]:
        token = set_hermes_home_override(str(homes[index]))
        try:
            state = manager.create_session(cwd=str(tmp_path))
            result = configure_session(manager, {"sessionId": state.session_id, "generation": index, "instructions": "BB contribution", "nativeInstructions": "native base " + str(index)})
            skill = next(skill for skill in result["skills"]["skills"] if skill["name"] == "package-skill")
            assert skill["origin"] == "plugin"
            assert skill["contentDigest"] == skill_content_digest(directories[index])
            assert str(directories[index]) in skill["filePath"]
            assert json.loads(skills_list())["count"] == 1
            assert "revision-" + ("a" if index == 0 else "b") in skill_view("package-skill", preprocess=False)
            assert result["instructions"]["instructionsDigest"] == hashlib.sha256(b"BB contribution").hexdigest()
            assert result["nativeInstructions"]["nativeInstructionsDigest"] == hashlib.sha256(("native base " + str(index)).encode()).hexdigest()
            observed.append(skill["contentDigest"])
            before = state.agent.ephemeral_system_prompt
            assert configure_session(manager, {"sessionId": state.session_id, "generation": index, "instructions": "BB contribution", "nativeInstructions": "native base " + str(index)}) == result
            assert state.agent.ephemeral_system_prompt == before
            state.history.append({"role": "user", "content": "already started"})
            state.prompt_started_in_instance = True
            with pytest.raises(RequestError):
                configure_session(manager, {"sessionId": state.session_id, "generation": index + 1, "instructions": "changed"})
            assert read_session_configuration(manager, {"sessionId": state.session_id})["generation"] == index
        finally:
            reset_hermes_home_override(token)
    assert observed[0] == observed[2] != observed[1]
    (directories[0] / "references" / "detail.md").write_text("changed reference")
    assert skill_content_digest(directories[0]) != observed[0]


def test_user_skill_overrides_managed_plugin_and_links_are_refused(tmp_path):
    home = tmp_path / "home"
    package = make_home(home, "package")
    local = write_skill(home / "skills", "package-skill", "user version")
    token = set_hermes_home_override(str(home))
    try:
        assert "user version" in skill_view("package-skill", preprocess=False)
        manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="fixture", valid_tool_names=set()))
        state = manager.create_session(cwd=str(tmp_path))
        result = read_session_configuration(manager, {"sessionId": state.session_id})
        assert result["skills"]["skills"][0]["origin"] == "user"
        assert result["skills"]["skills"][0]["contentDigest"] == skill_content_digest(local)
        (package / "linked.md").symlink_to(local / "SKILL.md")
        with pytest.raises(ValueError):
            skill_content_digest(package)
    finally:
        reset_hermes_home_override(token)


def test_loaded_history_may_bind_before_first_prompt_in_provider_instance(tmp_path):
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="fixture", valid_tool_names=set(), ephemeral_system_prompt="base"))
    state = manager.create_session(cwd=str(tmp_path))
    state.history = [{"role": "user", "content": "preserved conversation"}]
    state.configuration_generation = 1
    state.client_instructions = "old contribution"
    state.native_instructions = "old native"
    state.configuration_base_prompt = "base"
    state.agent.ephemeral_system_prompt = "base\n\nold contribution\n\nold native"
    with pytest.raises(RequestError):
        configure_session(manager, {"sessionId": state.session_id, "generation": 0, "instructions": "stale"})
    result = configure_session(manager, {"sessionId": state.session_id, "generation": 2, "instructions": "new contribution", "nativeInstructions": "new native"})
    assert result["generation"] == 2
    assert [(message["role"], message["content"]) for message in state.history] == [("user", "preserved conversation")]
    assert state.agent.ephemeral_system_prompt == "base\n\nnew contribution\n\nnew native"
    with pytest.raises(RequestError):
        configure_session(manager, {"sessionId": state.session_id, "generation": 3, "instructions": "later"})


def test_skill_digest_refuses_oversized_selected_file_and_reference(tmp_path):
    from acp_adapter.configuration import _bounded_skill_file_digest
    skill = tmp_path / "SKILL.md"
    with skill.open("wb") as stream:
        stream.truncate(10 * 1024 * 1024 + 1)
    from tools.skills_tool_plugin import _read_skill_text
    assert len(_read_skill_text(skill, metadata_only=True)) <= 4000
    with pytest.raises(ValueError):
        _read_skill_text(skill)
    with pytest.raises(ValueError):
        _bounded_skill_file_digest(skill)
    with pytest.raises(ValueError):
        skill_content_digest(tmp_path)


def test_rebuilt_agent_keeps_exact_configured_instruction_contributions():
    from acp_adapter.configuration import bind_agent_instructions
    state = SimpleNamespace(client_instructions="BB", native_instructions="native", configuration_base_prompt="old base")
    agent = SimpleNamespace(ephemeral_system_prompt="new native base")
    bind_agent_instructions(state, agent)
    assert agent.ephemeral_system_prompt == "new native base\n\nBB\n\nnative"
    assert state.configuration_base_prompt == "new native base"


def test_acp_remote_mcp_preserves_negotiated_transport():
    from acp.schema import McpServerHttp, McpServerSse
    from acp_adapter.server import _mcp_server_config
    assert _mcp_server_config(McpServerHttp(name="http", url="http://localhost/mcp", headers=[]))["transport"] == "http"
    assert _mcp_server_config(McpServerSse(name="sse", url="http://localhost/sse", headers=[]))["transport"] == "sse"


def test_readback_refuses_instruction_digest_if_agent_lost_its_contribution(tmp_path):
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="fixture", valid_tool_names=set(), ephemeral_system_prompt="base"))
    state = manager.create_session(cwd=str(tmp_path))
    configure_session(manager, {"sessionId": state.session_id, "generation": 1, "instructions": "BB", "nativeInstructions": "native"})
    state.agent.ephemeral_system_prompt = "replacement without contribution"
    result = read_session_configuration(manager, {"sessionId": state.session_id})
    assert result["instructions"]["status"] == "unavailable"
    assert result["nativeInstructions"]["status"] == "unavailable"


def test_native_conversation_readback_hashes_loaded_content_without_metadata():
    from acp_adapter.configuration import native_conversation_readback
    expected = hashlib.sha256(b'[{"content":"preserved","role":"user"}]').hexdigest()
    result = native_conversation_readback([{"role": "user", "content": "preserved", "timestamp": 1, "_row_id": 2}])
    assert result == {"status": "observed", "protocol": "hermes-acp", "messageCount": 1, "historyDigest": expected}
    assert native_conversation_readback([{"role": "user", "content": "x" * (10 * 1024 * 1024 + 1)}])["status"] == "unavailable"
    assert native_conversation_readback([{}] * 10001)["status"] == "unavailable"


def test_native_conversation_refuses_excessive_nesting_before_encoding():
    from acp_adapter.configuration import native_conversation_readback
    content = "leaf"
    for _ in range(34):
        content = [content]
    assert native_conversation_readback([{"role": "user", "content": content}])["status"] == "unavailable"
