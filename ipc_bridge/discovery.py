"""Pont générique entre une action IPC d'un plugin xcore (ex. une gestion de stock, un CRM…) et un sous-agent
appelable par un LLM — « un plugin, un agent, ses tools ».

Pourquoi un pont générique plutôt qu'un wrapper écrit à la main par action : un plugin métier ne devrait pas
avoir à écrire la moindre ligne de Python côté de l'app qui orchestre le modèle (le « host ») pour s'exposer
comme agent. Il suffit de déclarer ses actions dans son propre `plugin.yaml` (voir `discover_declared_actions`)
et de porter `@action`/`@schema` (xcoreruntime >= 2.9.0) sur le code réel — ce module fait le reste.

Deux phases, DÉLIBÉRÉMENT séparées (voir `service.py` pour qui les appelle, et pourquoi pas au même moment) :

1. `discover_declared_actions` — lit la section `rbac.actions` de chaque `plugin.yaml` frère sous le dossier de
   vos plugins métier : SEULES les entrées portant un champ `agent:` EN TOUTES LETTRES deviennent des
   candidates à l'exposition comme tool (opt-in explicite — avoir une action IPC ne suffit pas). Lecture de
   fichiers, sans dépendance à l'état d'aucun autre plugin : peut tourner à n'importe quel moment du démarrage.

2. `resolve_actions` — pour chaque candidate, va chercher le VRAI schéma dans `xcore.kernel.schema.schema_registry`
   (`@action(permissions=…, side_effect=…)` + `@schema(...)` du plugin cible, xcoreruntime >= 2.9.0) : paramètres
   JSON Schema réels (`ActionSchema.input_json_schema`, champs interdits au modèle retirés), permissions requises
   (`ActionSchema.permissions`), risque (`ActionSchema.side_effect`). Ne DOIT tourner qu'une fois que CE plugin
   cible a fini son propre démarrage — `schema_registry` ne contient les actions d'un plugin qu'après son
   `on_load` (xcore `lifecycle.py::_register_schemas`, appelé juste après). Un plugin qui en dépend d'un autre
   (via `requires:` dans son `plugin.yaml` — xcore charge les dépendances AVANT lui) n'a donc pas encore de
   schéma enregistré au moment où le VOTRE démarre si le vôtre démarre plus tôt : résoudre trop tôt (ex. pendant
   votre propre `on_load`) lirait `schema_registry.get(...)` → `None` et construirait un tool aux paramètres
   vides en silence. Appelez `resolve_actions` seulement une fois que TOUS les plugins de l'app ont démarré —
   par exemple en vous abonnant à l'évènement `xcore.plugins.booted` (émis par `PluginSupervisor.boot()` après
   que tout a chargé) plutôt qu'en résolvant pendant votre propre `on_load`.

Ce que ce mécanisme NE fait PAS, par choix de sécurité délibéré :
- Une action n'est exposée que si son `plugin.yaml` porte `agent:` sur son entrée `rbac.actions` — les entrées
  sans `agent:` restent de la documentation pure.
- Toute action dont les paramètres demanderaient tenant_id/user_id/sub au modèle les voit retirés (jamais
  rejetée pour autant — voir FORBIDDEN_MODEL_FIELDS et `_strip_forbidden`) : ces champs doivent TOUJOURS venir
  du contexte d'exécution (session/jeton authentifié), jamais du modèle.
- Une action `agent:`-taguée sans schéma réellement enregistré (typo, action jamais décorée `@action`, ou
  `permission_groups` non représentable ici) est IGNORÉE avec un avertissement, jamais exposée avec des
  paramètres inventés ou vides par défaut.
- Un `plugin.yaml` absent, illisible ou mal formé ne fait jamais échouer la découverte — juste ignoré, avec un
  avertissement dans les journaux.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from xcore.sdk import get_logger

logger = get_logger("ext.ipc_bridge")

# Champs qu'une action IPC ne doit JAMAIS demander au modèle : ils viennent du jeton de la requête (ctx.tenant_id
# / ctx.user_id), jamais d'un argument que le modèle choisit — sinon un modèle (ou une injection dans les
# données qu'il lit) pourrait se faire passer pour un autre tenant ou un autre utilisateur. Vérifié par un test
# sur CHAQUE entrée résolue (tests/test_ipc_bridge.py), pas seulement documenté ici.
FORBIDDEN_MODEL_FIELDS = frozenset({"tenant_id", "user_id", "sub"})

_RISK_VALUES = frozenset({"read", "write", "outbound"})


@dataclass(frozen=True)
class IPCAction:
    """Une action IPC exposée comme tool à un sous-agent — voir le docstring du module pour comment elle est
    construite (toujours via `resolve_actions`, jamais à la main hors tests).

    `plugin`/`action` : adressage `call_plugin(plugin, action, payload)` (manifeste xcore, pas le nom du
    dossier — voir CLAUDE.md sur `call_plugin` vs `name:` du manifeste).
    `tool_name` : nom présenté au modèle — doit être unique parmi TOUTES les actions de tous les agents (le
    dispatch à l'exécution résout par ce nom seul, pas par agent).
    `parameters` : JSON Schema (format OpenAI function-calling) des champs que le MODÈLE fournit — construit
    depuis `ActionSchema.input_json_schema`, tenant_id/user_id/sub toujours retirés (voir FORBIDDEN_MODEL_FIELDS),
    injectés par le dispatcher depuis `ctx`.
    `required_permissions` : ET de permissions xauth (voir access.py), copié de `ActionSchema.permissions` —
    vide = accessible à tout utilisateur authentifié. `permission_groups` (OU de ET) n'est pas représentable
    ici : une action qui en déclare est ignorée par `resolve_actions`, jamais approximée.
    `risk` : copié de `ActionSchema.side_effect` ("read"/"write"/"outbound") — "outbound" si l'action ne déclare
    rien (pas de côté sûr implicite, voir xcore `decorators.py::action` : « pas de valeur = pas de feu vert »).
    Documentation du niveau d'effet, pas encore branché sur une approbation humaine (confirm_action) : si un
    futur plugin a des actions d'écriture qui le justifient, cette brique sera ajoutée explicitement à ce
    moment-là, pas devinée ici.
    """

    plugin: str
    action: str
    tool_name: str
    description: str
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    required_permissions: frozenset[str] = frozenset()
    risk: Literal["read", "write", "outbound"] = "outbound"


@dataclass(frozen=True)
class _Declared:
    """Une entrée `rbac.actions[]` avec `agent:` — lue à plat, AVANT toute résolution contre schema_registry."""

    plugin: str
    action: str
    tool_name: str
    description: str | None  # None = reprendre la description de l'ActionSchema réelle, à la résolution
    agent: str


def tool_schema(action: IPCAction) -> dict:
    """Schéma de function-calling (format OpenAI) pour une IPCAction — même gabarit que TOOLS_SCHEMA (tools.py)."""
    return {
        "type": "function",
        "function": {
            "name": action.tool_name,
            "description": action.description,
            "parameters": action.parameters,
        },
    }


def actions_for(agents: dict[str, tuple[IPCAction, ...]], agent: str, granted: frozenset[str]) -> tuple[IPCAction, ...]:
    """Actions de cet agent que `granted` (ctx.permissions) autorise entièrement — vide si l'agent est inconnu
    ou si aucune action n'est accessible avec ces permissions. Filtre de CONSTRUCTION (ce qui est offert au
    modèle) ; `required_permissions` est revérifié à l'exécution par _call_ipc_action, indépendamment."""
    return tuple(a for a in agents.get(agent, ()) if a.required_permissions <= granted)


def action_for(agents: dict[str, tuple[IPCAction, ...]], tool_name: str) -> IPCAction | None:
    """Résout un nom de tool vers son IPCAction, tous agents confondus — la restriction par agent vient de
    `allowed` (voir tools.py::_execute_tool_call), pas de cette résolution, même logique que la résolution
    globale des noms mcp_* (ctx.mcp.resolve_tool_name)."""
    for actions in agents.values():
        for a in actions:
            if a.tool_name == tool_name:
                return a
    return None


def _declared_from_entry(plugin_name: str, entry: dict, seen_tool_names: set[str]) -> _Declared | None:
    """Une entrée `rbac.actions[]` avec `agent:` -> _Declared, ou None si elle doit être ignorée (avec un
    avertissement) : action absente/invalide, ou tool_name déjà pris par une autre entrée (collision entre deux
    plugins ou deux actions du même plugin)."""
    action = entry.get("action")
    if not isinstance(action, str) or not action:
        logger.warning("entrée rbac.actions de '%s' ignorée : `action` absente ou invalide", plugin_name)
        return None
    agent = entry.get("agent")
    tool_name = entry.get("tool_name") or action.replace(".", "_")
    if not isinstance(tool_name, str) or not tool_name:
        logger.warning("action IPC '%s:%s' ignorée : `tool_name` invalide", plugin_name, action)
        return None
    if tool_name in seen_tool_names:
        logger.warning(
            "action IPC '%s:%s' ignorée : tool_name '%s' déjà utilisé par une autre action IPC (doit être "
            "unique tous plugins confondus)",
            plugin_name, action, tool_name,
        )
        return None
    description = entry.get("description")
    description = description.strip() if isinstance(description, str) and description.strip() else None
    seen_tool_names.add(tool_name)
    return _Declared(plugin=plugin_name, action=action, tool_name=tool_name, description=description, agent=agent)


def discover_declared_actions(apps_root: Path) -> tuple[list[_Declared], dict[str, str]]:
    """Lit `rbac.actions` de chaque `plugin.yaml` sous `apps_root` et renvoie les entrées `agent:`-taguées (pas
    encore résolues contre schema_registry — voir le docstring du module) + la description de chaque agent
    trouvée. Jamais levé : un plugin.yaml absent, illisible ou mal formé est ignoré avec un avertissement — un
    plugin tiers en panne ne doit jamais empêcher votre app de démarrer."""
    declared: list[_Declared] = []
    descriptions: dict[str, str] = {}
    seen_tool_names: set[str] = set()
    # Pas de Path.glob() : son attribut porte le même nom que le module stdlib `glob`, interdit par le
    # scanner de sécurité des plugins trusted (xcore/kernel/security/validation.py — non bloquant pour un
    # plugin trusted, mais inutile de déclencher l'avertissement pour autant). iterdir() + un test de nom
    # fait exactement la même chose ici.
    candidates = sorted(apps_root.iterdir()) if apps_root.is_dir() else []
    for manifest_path in (p / "plugin.yaml" for p in candidates if p.is_dir()):
        if not manifest_path.is_file():
            continue
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("plugin.yaml illisible (%s) : %s", manifest_path, exc)
            continue
        if not isinstance(manifest, dict):
            continue
        plugin_name = manifest.get("name")
        if not isinstance(plugin_name, str) or not plugin_name:
            continue  # manifeste sans `name:` : rien d'adressable par call_plugin, on ignore
        entries = ((manifest.get("rbac") or {}).get("actions")) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            agent = entry.get("agent")
            if not isinstance(agent, str) or not agent:
                continue  # pas de tool déclaré, reste de la documentation pure (tier/requires/description)
            built = _declared_from_entry(plugin_name, entry, seen_tool_names)
            if built is not None:
                declared.append(built)
                agent_description = entry.get("agent_description")
                if isinstance(agent_description, str) and agent_description.strip() and agent not in descriptions:
                    descriptions[agent] = agent_description.strip()
    return declared, descriptions


def _strip_forbidden(input_json_schema: dict | None) -> dict:
    """`ActionSchema.input_json_schema` décrit le contrat IPC COMPLET (tenant_id/user_id compris) — jamais ce
    que le modèle doit fournir. Retire les champs interdits des `properties`/`required` plutôt que de rejeter
    l'action (contrairement à l'ancien mécanisme qui exigeait des `parameters` déjà nettoyés à la main). Le
    `title` par champ que `model_json_schema()` dérive du nom Python (ex. "Workflow Id" pour `workflow_id`) est
    retiré aussi : bruit redondant avec le nom du champ, pas une information utile pour le modèle."""
    schema = input_json_schema or {}
    properties = {
        k: {sk: sv for sk, sv in v.items() if sk != "title"}
        for k, v in (schema.get("properties") or {}).items()
        if k not in FORBIDDEN_MODEL_FIELDS
    }
    required = [r for r in (schema.get("required") or []) if r not in FORBIDDEN_MODEL_FIELDS]
    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


def resolve_actions(declared: list[_Declared], registry: Any = None) -> dict[str, tuple[IPCAction, ...]]:
    """Résout chaque `_Declared` contre `schema_registry` (xcoreruntime >= 2.9.0) en vraie `IPCAction` — voir le
    docstring du module pour QUAND ceci peut tourner sans risque (après `xcore.plugins.booted`, jamais avant).
    `registry` est injectable pour les tests ; par défaut, `xcore.kernel.schema.schema_registry` (le singleton
    réel). Une action sans schéma enregistré, ou dont les permissions ne sont pas représentables ici
    (`permission_groups`), est ignorée avec un avertissement — jamais exposée avec des paramètres devinés."""
    if registry is None:
        from xcore.kernel.schema import schema_registry as registry  # noqa: PLC0414

    agents: dict[str, list[IPCAction]] = {}
    for d in declared:
        schema = registry.get(d.plugin, d.action)
        if schema is None:
            logger.warning(
                "agent IPC '%s' : action '%s:%s' absente de schema_registry (pas de @action, action renommée, "
                "ou plugin pas encore chargé) — ignorée",
                d.agent, d.plugin, d.action,
            )
            continue
        if getattr(schema, "permission_groups", None):
            logger.warning(
                "agent IPC '%s' : action '%s:%s' déclare permission_groups (OU de ET), non représentable comme "
                "un simple ET de permissions — ignorée",
                d.agent, d.plugin, d.action,
            )
            continue
        risk = getattr(schema, "side_effect", None) or "outbound"
        if risk not in _RISK_VALUES:
            logger.warning("agent IPC '%s' : side_effect '%s' invalide pour '%s:%s', repli sur 'outbound'", d.agent, risk, d.plugin, d.action)
            risk = "outbound"
        description = d.description or (schema.description or "").strip() or d.action
        agents.setdefault(d.agent, []).append(
            IPCAction(
                plugin=d.plugin, action=d.action, tool_name=d.tool_name, description=description,
                parameters=_strip_forbidden(getattr(schema, "input_json_schema", None)),
                required_permissions=frozenset(getattr(schema, "permissions", None) or ()),
                risk=risk,
            )
        )
    return {agent: tuple(actions) for agent, actions in agents.items()}
