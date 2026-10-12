"""Pont générique IPC → sous-agent (ipc_bridge) : un plugin métier (ex. une gestion de stock) s'expose comme un
sous-agent dont les tools sont ses actions IPC, résolues contre schema_registry (xcoreruntime >= 2.9.0),
filtrées par permission, toujours scopées à l'identité de l'appelant — jamais au modèle de la fournir.

Deux phases testées séparément (voir ipc_bridge/discovery.py) :
- `discover_declared_actions` : lecture à plat des plugin.yaml, aucune dépendance à schema_registry.
- `resolve_actions` : construit les IPCAction réelles contre schema_registry — c'est ICI que
  paramètres/permissions/risque viennent du `@action`/`@schema` du plugin cible, jamais d'un YAML dupliqué.

Ce fichier ne teste QUE le pont lui-même (discovery.py + service.py), jamais le code d'un host qui le consomme
(ex. l'exécution d'un tool, la boucle LLM) — ça appartient aux tests de VOTRE app."""

from __future__ import annotations

import pytest
import yaml

from ipc_bridge import discovery as ipc_discovery
from ipc_bridge.discovery import IPCAction
from ipc_bridge.service import IpcBridgeService

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


STOCK_ADJUST = IPCAction(
    plugin="stock", action="stock.adjust", tool_name="stock_adjust", description="Ajuste le compte d'un article.",
    parameters={"type": "object", "properties": {"sku": {"type": "string"}, "delta": {"type": "integer"}}, "required": ["sku", "delta"]},
    required_permissions=frozenset({"tenants:write"}), risk="write",
)
STOCK_LIST = IPCAction(
    plugin="stock", action="stock.list", tool_name="stock_list", description="Liste les articles en stock.",
)


# ═════════════════════════════════════════════════════════════════════════════
# Catalogue : schéma, garde-fou d'identité, filtrage par permission
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
# IpcBridgeService : init() ne fait QUE déclarer (lecture disque), resolve() doit être appelé séparément — la
# preuve que le service respecte bien les deux phases (voir discovery.py pour pourquoi).
# ═════════════════════════════════════════════════════════════════════════════


def _write_stock_manifest(root):
    (root / "stock").mkdir()
    (root / "stock" / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "stock", "rbac": {"actions": [{"action": "stock.list", "agent": "stock_manager"}]}}),
        encoding="utf-8",
    )


def test_service_requires_apps_root_explicitly():
    """Pas de dossier deviné par défaut : la structure d'une app xcore n'est pas universelle, mieux vaut
    échouer tout de suite qu'explorer silencieusement le mauvais dossier."""
    with pytest.raises(ValueError):
        IpcBridgeService({})


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


def _write_manifest(root, plugin_dir: str, name: str, rbac_actions: list[dict] | None = None) -> None:
    (root / plugin_dir).mkdir(parents=True, exist_ok=True)
    manifest = {"name": name}
    if rbac_actions is not None:
        manifest["rbac"] = {"actions": rbac_actions}
    (root / plugin_dir / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


def test_declare_reads_agent_tagged_actions_and_ignores_documentation_only_ones(tmp_path):
    _write_manifest(
        tmp_path, "stock", "stock",
        rbac_actions=[
            {"action": "stock.list", "agent": "stock_manager", "description": "Liste les articles.", "agent_description": "gère le stock"},
            # Pas de `agent:` : documentation pure — ne doit PAS devenir un tool.
            {"action": "stock.internal_sync", "tier": "system"},
        ],
    )
    declared, descriptions = ipc_discovery.discover_declared_actions(tmp_path)
    assert len(declared) == 1
    [d] = declared
    assert (d.plugin, d.action, d.tool_name, d.agent) == ("stock", "stock.list", "stock_list", "stock_manager")
    assert descriptions == {"stock_manager": "gère le stock"}


def test_declare_skips_a_second_action_that_reuses_a_tool_name(tmp_path):
    _write_manifest(tmp_path, "stock", "stock", rbac_actions=[{"action": "stock.list", "agent": "a", "tool_name": "shared_name"}])
    _write_manifest(tmp_path, "crm", "crm", rbac_actions=[{"action": "crm.list", "agent": "b", "tool_name": "shared_name"}])
    declared, _ = ipc_discovery.discover_declared_actions(tmp_path)
    # Le premier plugin lu (ordre alphabétique des dossiers : crm < stock) gagne, le second est ignoré — jamais les deux.
    assert len(declared) == 1


def test_declare_ignores_a_manifest_without_a_name(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "plugin.yaml").write_text(yaml.safe_dump({"rbac": {"actions": [{"action": "x.y", "agent": "a"}]}}), encoding="utf-8")
    declared, _ = ipc_discovery.discover_declared_actions(tmp_path)
    assert declared == []


def test_declare_never_raises_on_malformed_yaml_and_still_reads_the_other_plugins(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "plugin.yaml").write_text("name: [unclosed", encoding="utf-8")
    _write_manifest(tmp_path, "stock", "stock", rbac_actions=[{"action": "stock.list", "agent": "stock_manager"}])
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
