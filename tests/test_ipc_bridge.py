"""Pont générique IPC → sous-agent (extensions/ipc_bridge) : un plugin métier (profile/workflows — xauth/xpulses
restent de l'infra, pas de besoin ici) s'expose comme un sous-agent dont les tools sont ses actions IPC,
résolues contre schema_registry (xcoreruntime >= 2.9.0), filtrées par permission, toujours scopées
tenant_id/user_id depuis ctx — jamais depuis le modèle.

Deux phases testées séparément (voir discovery.py) :
- `discover_declared_actions` : lecture à plat des plugin.yaml, aucune dépendance à schema_registry.
- `resolve_actions` : construit les IPCAction réelles contre schema_registry — c'est ICI que
  paramètres/permissions/risque viennent du `@action`/`@schema` du plugin cible, jamais d'un YAML dupliqué."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from app.chat.src import tools
from extensions.ipc_bridge import discovery as ipc_discovery
from extensions.ipc_bridge.discovery import IPCAction
from extensions.ipc_bridge.service import IpcBridgeService

pytestmark = pytest.mark.anyio

TENANT, USER = "t1", "alice"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_ipc_bridge():
    """Chaque test repart d'un pont vide — jamais une fuite de l'état réel (résolu au vrai démarrage de chat
    dans d'autres tests du process) ni d'un test au précédent."""
    old = tools._IPC_BRIDGE
    tools.bind_ipc_bridge(None)
    yield
    tools.bind_ipc_bridge(old)


def ctx(**fields) -> tools.ToolContext:
    base = dict(db=None, websocket=None, tenant_id=TENANT, user_id=USER, conversation_id="c1", mcp=None, preferences={})
    return tools.ToolContext(**{**base, **fields})


def _fake_bridge(agents: dict | None = None, descriptions: dict | None = None):
    return SimpleNamespace(agents=agents or {}, descriptions=descriptions or {})


STOCK_ADJUST = IPCAction(
    plugin="stock", action="stock.adjust", tool_name="stock_adjust", description="Ajuste le compte d'un article.",
    parameters={"type": "object", "properties": {"sku": {"type": "string"}, "delta": {"type": "integer"}}, "required": ["sku", "delta"]},
    required_permissions=frozenset({"tenants:write"}), risk="write",
)
STOCK_LIST = IPCAction(
    plugin="stock", action="stock.list", tool_name="stock_list", description="Liste les articles en stock.",
)


# ═════════════════════════════════════════════════════════════════════════════
# Catalogue : schéma, garde-fou tenant_id/user_id, filtrage par permission
# ═════════════════════════════════════════════════════════════════════════════


def test_tool_schema_matches_the_openai_function_calling_shape():
    schema = ipc_discovery.tool_schema(STOCK_ADJUST)
    assert schema == {
        "type": "function",
        "function": {"name": "stock_adjust", "description": "Ajuste le compte d'un article.", "parameters": STOCK_ADJUST.parameters},
    }


@pytest.mark.parametrize("action", [STOCK_ADJUST, STOCK_LIST])
def test_no_action_ever_asks_the_model_for_tenant_or_user_identity(action):
    fields = set((action.parameters or {}).get("properties", {}))
    assert not (fields & ipc_discovery.FORBIDDEN_MODEL_FIELDS), action.tool_name


def test_actions_for_filters_by_granted_permissions():
    agents = {"stock_manager": (STOCK_ADJUST, STOCK_LIST)}
    assert ipc_discovery.actions_for(agents, "stock_manager", frozenset()) == (STOCK_LIST,)  # lecture seule : pas de grant requis
    assert ipc_discovery.actions_for(agents, "stock_manager", frozenset({"tenants:write"})) == (STOCK_ADJUST, STOCK_LIST)
    assert ipc_discovery.actions_for(agents, "inconnu", frozenset({"tenants:write"})) == ()


def test_action_for_resolves_by_tool_name_across_all_agents():
    agents = {"stock_manager": (STOCK_ADJUST, STOCK_LIST)}
    assert ipc_discovery.action_for(agents, "stock_adjust") is STOCK_ADJUST
    assert ipc_discovery.action_for(agents, "inconnu") is None


# ═════════════════════════════════════════════════════════════════════════════
# Dispatch : _call_ipc_action — permissions revérifiées, tenant_id/user_id TOUJOURS depuis ctx
# ═════════════════════════════════════════════════════════════════════════════


class FakeCallPlugin:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict]] = []

    async def __call__(self, plugin, action, payload):
        self.calls.append((plugin, action, dict(payload)))
        return self.response


async def test_call_ipc_action_injects_tenant_and_user_from_ctx_never_from_the_model():
    plugin = FakeCallPlugin({"status": "ok", "count": 3})
    context = ctx(call_plugin=plugin, permissions=frozenset({"tenants:write"}))
    # Le modèle tente de fournir SON PROPRE tenant_id/user_id (halluciné ou injecté) : ignoré, écrasé par ctx.
    result = await tools._call_ipc_action(context, STOCK_ADJUST, {"sku": "ABC", "delta": -1, "tenant_id": "autre-tenant", "user_id": "quelqu-un-dautre"})
    assert "count" in result and "ok" not in result  # status retiré du résultat renvoyé au modèle
    assert plugin.calls == [("stock", "stock.adjust", {"sku": "ABC", "delta": -1, "tenant_id": TENANT, "user_id": USER})]


async def test_call_ipc_action_refuses_without_the_required_permission_even_if_allowed_let_it_through():
    plugin = FakeCallPlugin({"status": "ok"})
    context = ctx(call_plugin=plugin, permissions=frozenset())  # aucune permission
    result = await tools._call_ipc_action(context, STOCK_ADJUST, {"sku": "ABC", "delta": 1})
    assert "non disponible" in result
    assert plugin.calls == []  # jamais appelé : refusé AVANT d'atteindre le pont IPC


async def test_call_ipc_action_without_a_required_permission_is_open_to_any_authenticated_user():
    plugin = FakeCallPlugin({"status": "ok", "items": []})
    context = ctx(call_plugin=plugin, permissions=frozenset())
    result = await tools._call_ipc_action(context, STOCK_LIST, {})
    assert "items" in result
    assert plugin.calls == [("stock", "stock.list", {"tenant_id": TENANT, "user_id": USER})]


async def test_call_ipc_action_reports_a_plugin_error_without_raising():
    plugin = FakeCallPlugin({"status": "error", "msg": "rupture de stock"})
    context = ctx(call_plugin=plugin, permissions=frozenset({"tenants:write"}))
    result = await tools._call_ipc_action(context, STOCK_ADJUST, {"sku": "ABC", "delta": -100})
    assert result == "Échoué : rupture de stock"


async def test_call_ipc_action_without_a_loaded_bridge_says_so():
    context = ctx(call_plugin=None, permissions=frozenset({"tenants:write"}))
    assert "pont IPC" in await tools._call_ipc_action(context, STOCK_ADJUST, {"sku": "ABC", "delta": 1})


async def test_execute_tool_call_raw_routes_an_ipc_tool_name_to_the_ipc_dispatcher(monkeypatch):
    monkeypatch.setattr(tools, "_IPC_BRIDGE", _fake_bridge({"stock_manager": (STOCK_LIST,)}))
    plugin = FakeCallPlugin({"status": "ok", "items": ["x"]})
    context = ctx(call_plugin=plugin, permissions=frozenset())
    result = await tools._execute_tool_call(context, "stock_list", {}, frozenset({"stock_list"}))
    assert "items" in result and plugin.calls


# ═════════════════════════════════════════════════════════════════════════════
# delegate_to_subagent : fusion MCP + IPC, agent IPC-only sans pont MCP chargé
# ═════════════════════════════════════════════════════════════════════════════


class FakeOllama:
    def __init__(self, replies) -> None:
        self.replies = list(replies)

    async def chat_with_tools(self, turns, tools, force_ollama=False):
        return self.replies.pop(0)


def _final_reply(text: str) -> dict:
    return {"content": text, "tool_calls": [], "usage": None}


async def test_an_ipc_only_agent_works_with_no_mcp_bridge_loaded(monkeypatch):
    """ctx.mcp is None ne doit plus refuser l'appel pour un agent purement IPC — seul un agent MCP en a besoin."""
    monkeypatch.setattr(tools, "_IPC_BRIDGE", _fake_bridge({"stock_manager": (STOCK_LIST,)}, {"stock_manager": "gère le stock"}))
    plugin = FakeCallPlugin({"status": "ok", "items": ["ABC"]})
    context = ctx(mcp=None, call_plugin=plugin, permissions=frozenset(), ollama=FakeOllama([_final_reply("3 articles en stock.")]))
    result = await tools._delegate_to_subagent(context, {"agent": "stock_manager", "task": "combien d'articles ?"})
    assert result == "3 articles en stock."


async def test_a_write_tier_ipc_action_is_hidden_from_a_subagent_without_the_permission(monkeypatch):
    monkeypatch.setattr(tools, "_IPC_BRIDGE", _fake_bridge({"stock_manager": (STOCK_ADJUST, STOCK_LIST)}))
    seen_tools: list[list[str]] = []

    class RecordingOllama(FakeOllama):
        async def chat_with_tools(self, turns, tools, force_ollama=False):
            seen_tools.append([t["function"]["name"] for t in tools])
            return self.replies.pop(0)

    context = ctx(mcp=None, call_plugin=FakeCallPlugin({"status": "ok"}), permissions=frozenset(), ollama=RecordingOllama([_final_reply("ok")]))
    await tools._delegate_to_subagent(context, {"agent": "stock_manager", "task": "liste le stock"})
    assert seen_tools == [["stock_list"]]  # stock_adjust (tenants:write) jamais présenté sans la permission


async def test_an_unknown_agent_lists_both_mcp_and_ipc_agent_names(monkeypatch):
    monkeypatch.setattr(tools, "_IPC_BRIDGE", _fake_bridge({"stock_manager": (STOCK_LIST,)}))

    class Mcp:
        def list_tools_schema_for_agent(self, agent):
            return []

        def list_sub_agents(self, who):
            return [{"name": "redacteur", "description": "x"}]

    context = ctx(mcp=Mcp(), permissions=frozenset(), ollama=FakeOllama([]))
    result = await tools._delegate_to_subagent(context, {"agent": "inconnu", "task": "x"})
    assert "redacteur" in result and "stock_manager" in result


async def test_sync_delegate_to_subagent_agents_reflects_the_bound_bridge(monkeypatch):
    monkeypatch.setattr(tools, "_IPC_BRIDGE", _fake_bridge({"stock_manager": (STOCK_LIST,)}, {"stock_manager": "gère le stock"}))
    tools.sync_delegate_to_subagent_agents()
    schema = next(t for t in tools.TOOLS_SCHEMA if t["function"]["name"] == "delegate_to_subagent")
    prop = schema["function"]["parameters"]["properties"]["agent"]
    assert "stock_manager" in prop["enum"]
    assert "stock_manager" in prop["description"] and "gère le stock" in prop["description"]
    assert prop["description"].startswith(tools._AGENT_PARAM_BASE_DESCRIPTION)


async def test_syncing_twice_does_not_duplicate_the_description(monkeypatch):
    monkeypatch.setattr(tools, "_IPC_BRIDGE", _fake_bridge({"stock_manager": (STOCK_LIST,)}, {"stock_manager": "gère le stock"}))
    tools.sync_delegate_to_subagent_agents()
    tools.sync_delegate_to_subagent_agents()
    schema = next(t for t in tools.TOOLS_SCHEMA if t["function"]["name"] == "delegate_to_subagent")
    description = schema["function"]["parameters"]["properties"]["agent"]["description"]
    assert description.count("stock_manager") == 1


# ═════════════════════════════════════════════════════════════════════════════
# IpcBridgeService : init() ne fait QUE déclarer (lecture disque), resolve() doit être appelé séparément — la
# preuve que le service respecte bien les deux phases (voir discovery.py pour pourquoi).
# ═════════════════════════════════════════════════════════════════════════════


def _write_stock_manifest(root):
    (root / "stock").mkdir()
    (root / "stock" / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "stock", "agent": {"name": "stock_manager", "actions": [{"action": "stock.list"}]}}),
        encoding="utf-8",
    )


async def test_service_init_only_declares_resolve_is_a_separate_step(tmp_path):
    _write_stock_manifest(tmp_path)
    service = IpcBridgeService({"apps_root": str(tmp_path)})
    await service.init()
    assert service.agents == {}  # pas encore résolu : schema_registry n'est pas forcément complet ici
    service.resolve(_FakeRegistry({("stock", "stock.list"): _FakeSchema(side_effect="read")}))
    assert service.agents["stock_manager"][0].tool_name == "stock_list"


async def test_service_health_check_and_status_report_the_resolved_catalog(tmp_path):
    _write_stock_manifest(tmp_path)
    service = IpcBridgeService({"apps_root": str(tmp_path)})
    await service.init()
    service.resolve(_FakeRegistry({("stock", "stock.list"): _FakeSchema(side_effect="read")}))
    ok, msg = await service.health_check()
    assert ok and "1" in msg
    assert service.status()["agents"] == ["stock_manager"]
    await service.shutdown()


# ═════════════════════════════════════════════════════════════════════════════
# discover_declared_actions : lecture des plugin.yaml frères — aucun Python requis par plugin, aucune
# dépendance à schema_registry (voir discovery.py pour pourquoi c'est une phase séparée de la résolution)
# ═════════════════════════════════════════════════════════════════════════════


def _write_manifest(root, plugin_dir: str, name: str, agent: dict | None = None, rbac_actions: list[dict] | None = None) -> None:
    (root / plugin_dir).mkdir(parents=True, exist_ok=True)
    manifest: dict = {"name": name}
    if agent is not None:
        manifest["agent"] = agent
    if rbac_actions is not None:
        manifest["rbac"] = {"actions": rbac_actions}
    (root / plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


def test_declare_reads_the_agent_block_and_its_listed_actions(tmp_path):
    _write_manifest(
        tmp_path, "stock", "stock",
        agent={
            "name": "stock_manager",
            "description": "Gère le stock.",
            "version": "2.1.0",
            "capabilities": ["lister les articles"],
            "actions": [{"action": "stock.list", "description": "Liste les articles."}],
        },
        # `rbac.actions` n'est PLUS lu : cette entrée ne doit produire aucun tool.
        rbac_actions=[{"action": "stock.internal_sync", "tier": "system"}],
    )
    declared, agents = ipc_discovery.discover_declared_actions(tmp_path)
    assert len(declared) == 1
    [d] = declared
    assert (d.plugin, d.action, d.tool_name, d.agent) == ("stock", "stock.list", "stock_list", "stock_manager")
    assert d.description == "Liste les articles."
    assert agents["stock_manager"].version == "2.1.0"
    # Les capacités déclarées rejoignent ce que le modèle lit pour CHOISIR l'agent.
    assert agents["stock_manager"].prompt_description() == "Gère le stock. Sait : lister les articles."


def test_declare_ignores_rbac_actions_entirely_even_agent_tagged_ones(tmp_path):
    """Garde-fou de la migration : l'ancienne forme ne doit plus rien exposer en silence."""
    _write_manifest(tmp_path, "stock", "stock", rbac_actions=[{"action": "stock.list", "agent": "stock_manager", "tool_name": "stock_list"}])
    declared, agents = ipc_discovery.discover_declared_actions(tmp_path)
    assert declared == [] and agents == {}


def test_declare_ignores_a_plugin_without_an_agent_block(tmp_path):
    """Le cas NORMAL : un plugin qui n'expose rien (xauth, xpulses…), sans avertissement."""
    _write_manifest(tmp_path, "stock", "stock")
    declared, agents = ipc_discovery.discover_declared_actions(tmp_path)
    assert declared == [] and agents == {}


def test_declare_ignores_an_agent_block_without_a_usable_name(tmp_path):
    _write_manifest(tmp_path, "stock", "stock", agent={"description": "x", "actions": [{"action": "stock.list"}]})
    _write_manifest(tmp_path, "crm", "crm", agent={"name": "   ", "actions": [{"action": "crm.list"}]})
    declared, agents = ipc_discovery.discover_declared_actions(tmp_path)
    assert declared == [] and agents == {}


def test_declare_drops_an_agent_whose_actions_are_all_unusable(tmp_path):
    """Un agent sans aucune action exposable ferait un choix vide dans l'enum de delegate_to_subagent."""
    _write_manifest(tmp_path, "stock", "stock", agent={"name": "stock_manager", "actions": [{"action": ""}, "pas-un-bloc"]})
    declared, agents = ipc_discovery.discover_declared_actions(tmp_path)
    assert declared == [] and agents == {}


def test_declare_skips_a_second_action_that_reuses_a_tool_name(tmp_path):
    _write_manifest(tmp_path, "stock", "stock", agent={"name": "a", "actions": [{"action": "stock.list", "tool_name": "shared_name"}]})
    _write_manifest(tmp_path, "crm", "crm", agent={"name": "b", "actions": [{"action": "crm.list", "tool_name": "shared_name"}]})
    declared, agents = ipc_discovery.discover_declared_actions(tmp_path)
    # Le premier plugin lu (ordre alphabétique des dossiers : crm < stock) gagne, le second est ignoré — jamais les deux.
    assert len(declared) == 1 and declared[0].plugin == "crm"
    assert sorted(agents) == ["b"]  # l'agent sans action exposable n'est pas déclaré non plus


def test_declare_refuses_two_plugins_claiming_the_same_agent_name(tmp_path):
    _write_manifest(tmp_path, "stock", "stock", agent={"name": "twin", "actions": [{"action": "stock.list"}]})
    _write_manifest(tmp_path, "crm", "crm", agent={"name": "twin", "actions": [{"action": "crm.list"}]})
    declared, agents = ipc_discovery.discover_declared_actions(tmp_path)
    assert sorted(agents) == ["twin"] and [d.plugin for d in declared] == ["crm"]


def test_declare_ignores_a_manifest_without_a_name(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "plugin.yaml").write_text(
        yaml.safe_dump({"agent": {"name": "a", "actions": [{"action": "x.y"}]}}), encoding="utf-8"
    )
    declared, _ = ipc_discovery.discover_declared_actions(tmp_path)
    assert declared == []


def test_declare_never_raises_on_malformed_yaml_and_still_reads_the_other_plugins(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "plugin.yaml").write_text("name: [unclosed", encoding="utf-8")
    _write_manifest(tmp_path, "stock", "stock", agent={"name": "stock_manager", "actions": [{"action": "stock.list"}]})
    declared, _ = ipc_discovery.discover_declared_actions(tmp_path)
    assert len(declared) == 1 and declared[0].agent == "stock_manager"


# ═════════════════════════════════════════════════════════════════════════════
# resolve_actions : contre un schema_registry FAUX (duck-typé) — paramètres/permissions/risque viennent
# UNIQUEMENT de là, jamais d'un champ YAML (ils n'existent plus dans le YAML depuis xcoreruntime 2.9.0)
# ═════════════════════════════════════════════════════════════════════════════


class _FakeSchema:
    def __init__(self, input_json_schema=None, permissions=None, permission_groups=None, side_effect=None, description=""):
        self.input_json_schema = input_json_schema or {}
        self.permissions = permissions or []
        self.permission_groups = permission_groups or []
        self.side_effect = side_effect
        self.description = description


class _FakeRegistry:
    def __init__(self, schemas: dict[tuple[str, str], _FakeSchema]) -> None:
        self._schemas = schemas

    def get(self, plugin: str, action: str):
        return self._schemas.get((plugin, action))


def _declared(plugin="stock", action="stock.adjust", tool_name="stock_adjust", description=None, agent="stock_manager"):
    return ipc_discovery._Declared(plugin=plugin, action=action, tool_name=tool_name, description=description, agent=agent)


def test_resolve_strips_tenant_and_user_from_parameters_instead_of_rejecting_the_action():
    registry = _FakeRegistry({
        ("stock", "stock.adjust"): _FakeSchema(
            input_json_schema={
                "type": "object",
                "properties": {"tenant_id": {"type": "string"}, "user_id": {"type": "string"}, "sku": {"type": "string"}},
                "required": ["tenant_id", "user_id", "sku"],
            },
            side_effect="write",
        ),
    })
    agents = ipc_discovery.resolve_actions([_declared()], registry)
    [action] = agents["stock_manager"]
    assert action.parameters == {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}


def test_resolve_copies_permissions_and_risk_from_the_real_schema_not_from_any_yaml_field():
    registry = _FakeRegistry({("stock", "stock.adjust"): _FakeSchema(permissions=["tenants:write"], side_effect="write")})
    agents = ipc_discovery.resolve_actions([_declared()], registry)
    [action] = agents["stock_manager"]
    assert action.required_permissions == frozenset({"tenants:write"})
    assert action.risk == "write"


def test_resolve_defaults_an_undeclared_side_effect_to_outbound_the_most_risky():
    """Pas de side_effect déclaré = pas de feu vert implicite (voir xcore decorators.py::action) : traité comme
    le risque le plus élevé, jamais comme 'read' par défaut."""
    registry = _FakeRegistry({("stock", "stock.adjust"): _FakeSchema()})  # side_effect=None
    agents = ipc_discovery.resolve_actions([_declared()], registry)
    assert agents["stock_manager"][0].risk == "outbound"


def test_resolve_skips_an_agent_tagged_action_without_a_registered_schema():
    """Typo, action jamais décorée @action, ou plugin pas chargé : ignorée, JAMAIS exposée avec des paramètres
    inventés ou vides par défaut."""
    registry = _FakeRegistry({})  # rien enregistré
    agents = ipc_discovery.resolve_actions([_declared()], registry)
    assert agents == {}


def test_resolve_skips_an_action_whose_permission_groups_are_not_representable():
    registry = _FakeRegistry({("stock", "stock.adjust"): _FakeSchema(permission_groups=[["tenants:write"], ["admin:*"]])})
    agents = ipc_discovery.resolve_actions([_declared()], registry)
    assert agents == {}


def test_resolve_falls_back_to_the_schema_description_when_the_yaml_entry_has_none():
    registry = _FakeRegistry({("stock", "stock.adjust"): _FakeSchema(description="Ajuste le compte d'un article.")})
    agents = ipc_discovery.resolve_actions([_declared(description=None)], registry)
    assert agents["stock_manager"][0].description == "Ajuste le compte d'un article."


# ═════════════════════════════════════════════════════════════════════════════
# Essai sur de VRAIS plugins : profile.get ("profile_assistant") et workflows.list/pause
# ("workflow_manager") exposés comme tools — preuve de bout en bout, pas une maquette.
# ═════════════════════════════════════════════════════════════════════════════


def test_discovering_the_real_app_directory_finds_the_two_live_trials():
    """profile.get et workflows.list/pause sont les seuls essais réels en production (voir leurs plugin.yaml,
    commentaires « ESSAI ») — xauth/xpulses ne déclarent toujours rien (de l'infra, pas un agent). Ce test
    casse si un autre apparaît, prévu ou pas : à relire pour confirmer que c'est volontaire avant de le mettre
    à jour."""
    declared, agents = ipc_discovery.discover_declared_actions(Path(__file__).resolve().parents[1] / "app")
    assert {d.agent for d in declared} == {"profile_assistant", "workflow_manager"}
    assert {d.tool_name for d in declared} == {"get_my_profile", "list_workflows", "pause_workflow"}
    # Chaque agent réel se présente au modèle : sans description, il n'est pas choisissable.
    assert agents["profile_assistant"].prompt_description() and agents["workflow_manager"].prompt_description()


async def test_profile_assistant_reads_the_real_profile_plugin_end_to_end(tmp_path):
    """Le VRAI IPCCommands de profile (app/profile/src/ipc.py), une VRAIE base SQLite, un VRAI schema_registry
    (son @action/@schema réels, pas un faux), câblé via le dispatcher générique _call_ipc_action — seul le
    modèle est simulé (FakeOllama scripté) : on prouve le câblage réel YAML -> discover_declared_actions ->
    schema_registry -> resolve_actions -> delegate_to_subagent -> call_plugin -> profile.get."""
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.profile.src import models as profile_models
    from app.profile.src.ipc import IPCCommands as ProfileIPCCommands

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/profile.db")
    async with engine.begin() as conn:
        await conn.run_sync(profile_models.Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add(profile_models.Profile(tenant_id=TENANT, user_id=USER, timezone="Europe/Paris", preferred_tts_voice="fr_FR-siwis-medium"))
        await session.commit()

    class Db:
        @asynccontextmanager
        async def session(self):
            async with maker() as s:
                yield s
                await s.commit()

    class ProfileHost(ProfileIPCCommands):
        def __init__(self) -> None:
            self._db = Db()

    profile_plugin = ProfileHost()
    profile_plugin._register_schemas("profile")  # AutoDispatchMixin : remplit le VRAI schema_registry

    declared, descriptions = ipc_discovery.discover_declared_actions(Path(__file__).resolve().parents[1] / "app")
    agents = ipc_discovery.resolve_actions(declared)
    [profile_action] = agents["profile_assistant"]
    assert (profile_action.tool_name, profile_action.parameters) == ("get_my_profile", {"type": "object", "properties": {}})
    tools.bind_ipc_bridge(_fake_bridge(agents, descriptions))

    async def real_call_plugin(plugin: str, action: str, payload: dict) -> dict:
        assert plugin == "profile" and action == "profile.get"
        return await profile_plugin.handle(action, payload)  # AutoDispatchMixin.handle, pas de raccourci de test

    class ScriptedOllama:
        def __init__(self) -> None:
            self.step = 0
            self.tool_result: str | None = None

        async def chat_with_tools(self, turns, tools, force_ollama=False):
            self.step += 1
            if self.step == 1:
                return {"content": "", "tool_calls": [{"name": "get_my_profile", "arguments": {}}], "usage": None}
            self.tool_result = turns[-1]["content"]
            return {"content": "Ton fuseau est Europe/Paris.", "tool_calls": [], "usage": None}

    ollama = ScriptedOllama()
    # Le modèle n'a AUCUN moyen de fournir tenant_id/user_id (parameters n'a que des properties vides) :
    # seul ce que ctx porte compte — ici TENANT/USER, pas un autre espace.
    context = ctx(call_plugin=real_call_plugin, permissions=frozenset(), ollama=ollama)
    result = await tools._delegate_to_subagent(context, {"agent": "profile_assistant", "task": "quel est mon fuseau horaire ?"})

    assert result == "Ton fuseau est Europe/Paris."
    assert ollama.tool_result is not None and "Europe/Paris" in ollama.tool_result
    await engine.dispose()


async def test_workflow_manager_pause_end_to_end_against_the_real_workflows_plugin(tmp_path):
    """Même preuve que profile_assistant, sur le second plugin métier : workflows.pause a un VRAI paramètre
    modèle (workflow_id) contrairement à profile.get — prouve que resolve_actions construit des `parameters`
    non vides correctement, pas seulement le cas trivial à zéro champ."""
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.workflows.src import models as wf_models
    from app.workflows.src.ipc import IPCCommands as WorkflowsIPCCommands
    from app.workflows.src.models import Workflow

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/workflows.db")
    async with engine.begin() as conn:
        await conn.run_sync(wf_models.Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        session.add(Workflow(id="wf1", tenant_id=TENANT, user_id=USER, name="Briefing", definition="{}", status="active"))
        await session.commit()

    class Db:
        @asynccontextmanager
        async def session(self):
            async with maker() as s:
                yield s
                await s.commit()

    class WorkflowsHost(WorkflowsIPCCommands):
        def __init__(self) -> None:
            self._db = Db()

    workflows_plugin = WorkflowsHost()
    workflows_plugin._register_schemas("workflows")

    declared, descriptions = ipc_discovery.discover_declared_actions(Path(__file__).resolve().parents[1] / "app")
    agents = ipc_discovery.resolve_actions(declared)
    pause_action = next(a for a in agents["workflow_manager"] if a.tool_name == "pause_workflow")
    assert pause_action.parameters == {"type": "object", "properties": {"workflow_id": {"type": "string"}}, "required": ["workflow_id"]}
    assert pause_action.risk == "write"
    tools.bind_ipc_bridge(_fake_bridge(agents, descriptions))

    async def real_call_plugin(plugin: str, action: str, payload: dict) -> dict:
        assert plugin == "workflows" and action == "workflows.pause"
        return await workflows_plugin.handle(action, payload)

    class ScriptedOllama:
        def __init__(self) -> None:
            self.step = 0

        async def chat_with_tools(self, turns, tools, force_ollama=False):
            self.step += 1
            if self.step == 1:
                return {"content": "", "tool_calls": [{"name": "pause_workflow", "arguments": {"workflow_id": "wf1"}}], "usage": None}
            return {"content": "Mis en pause.", "tool_calls": [], "usage": None}

    context = ctx(call_plugin=real_call_plugin, permissions=frozenset(), ollama=ScriptedOllama())
    result = await tools._delegate_to_subagent(context, {"agent": "workflow_manager", "task": "mets en pause le briefing"})
    assert result == "Mis en pause."
    async with maker() as session:
        workflow = await session.get(Workflow, "wf1")
        assert workflow.status == "paused"
    await engine.dispose()
