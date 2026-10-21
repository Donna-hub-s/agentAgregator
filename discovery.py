"""Pont générique entre une action IPC d'un plugin xcore (profile, workflows, xauth… ou un futur plugin métier,
ex. une gestion de stock) et un sous-agent de DONNA appelable via `delegate_to_subagent`.

Pourquoi un pont générique plutôt qu'un wrapper écrit à la main par action (comme `propose_workflow` pour
`workflows.propose`) : un plugin métier futur, sans lien avec DONNA elle-même, doit pouvoir s'exposer comme un
agent — « un plugin, un agent, ses tools » — sans qu'on écrive la moindre ligne de Python côté DONNA. Déplacé de
`app/chat/src/ipc_agents.py` vers `extensions/` : c'est un mécanisme générique xcore (n'importe quel plugin peut
s'y déclarer), pas une logique propre au plugin chat — même raison que `mcp_bridge`.

Deux phases, DÉLIBÉRÉMENT séparées (voir `service.py` pour qui les appelle, et pourquoi pas au même moment) :

1. `discover_declared_actions` — lit le bloc `agent:` de chaque `plugin.yaml` frère sous `app/`. Un plugin
   qui n'en a pas n'expose RIEN (opt-in explicite — avoir des actions IPC ne suffit pas) ; un plugin qui en a un
   expose UN agent, le sien, et les actions qu'il y liste :

       agent:
         name: "workflow_manager"            # obligatoire : identifie l'agent, unique tous plugins confondus
         description: "Gère les automatisations de l'utilisateur."   # ce que DONNA lit pour CHOISIR cet agent
         version: "1.0.0"                    # métadonnée, journalisée au démarrage
         capabilities:                       # ajoutées à la description vue par le modèle
           - "Consultation des automatisations"
         actions:
           - action: "workflows.list"        # obligatoire : ce que call_plugin adressera
             tool_name: "list_workflows"     # facultatif (défaut : l'action, points -> souligné)
             description: "Liste les workflows de l'appelant."   # facultatif (défaut : celle du @schema)

   Un plugin = UN agent : avant, chaque entrée de `rbac.actions` portait son propre champ `agent:`, et rien
   n'empêchait un même plugin d'en déclarer trois. Lecture de fichiers, sans dépendance à l'état d'aucun autre
   plugin : peut tourner à n'importe quel moment du démarrage.

2. `resolve_actions` — pour chaque candidate, va chercher le VRAI schéma dans `xcore.kernel.schema.schema_registry`
   (`@action(permissions=…, side_effect=…)` + `@schema(...)` du plugin cible, xcoreruntime >= 2.9.0) : paramètres
   JSON Schema réels (`ActionSchema.input_json_schema`, champs interdits au modèle retirés), permissions requises
   (`ActionSchema.permissions`), risque (`ActionSchema.side_effect`). Ne DOIT tourner qu'une fois que CE plugin
   cible a fini son propre démarrage — `schema_registry` ne contient les actions d'un plugin qu'après son
   `on_load` (xcore `lifecycle.py::_register_schemas`, appelé juste après). Un plugin qui en dépend (ex.
   `workflows` déclare `requires: chat` dans son `plugin.yaml` — xcore charge donc `chat` AVANT `workflows`)
   n'a donc PAS encore de schéma enregistré au moment où `chat` démarre : résoudre ici, pendant le `on_load` de
   chat, lirait `schema_registry.get("workflows", "workflows.list")` → `None` et construirait un tool aux
   paramètres vides en silence. `resolve_actions` est donc appelé plus tard, une fois l'évènement
   `xcore.plugins.booted` reçu (émis par `PluginSupervisor.boot()` après que TOUS les plugins ont démarré) — voir
   `app/chat/src/main.py::_on_plugins_booted`.

`rbac.actions` n'est PLUS lu du tout. Ce bloc ne décrit plus rien d'exécutable : ni xcore (ni son validateur
de manifeste, ni son middleware de permissions) ni ce pont ne le consultent — les permissions réelles viennent
de `@action(permissions=…)`. Ce qu'il en reste dans les manifestes est de la documentation (`tier`, `requires`),
et elle n'est volontairement pas recopiée dans le bloc `agent:` : un `requires: []` à côté d'un vrai décorateur
Python est exactement le genre de duplication qui dérive en silence — c'est déjà pour cette raison que
`parameters`/`risk` avaient quitté le YAML à la v2.9.0 de xcoreruntime.

Ce que ce mécanisme NE fait PAS, par choix de sécurité délibéré (voir AGENTS.md — allowlist d'exécution) :
- Une action n'est exposée que si son `plugin.yaml` porte un bloc `agent:` ET la liste dans ses `actions:` —
  un plugin sans ce bloc n'expose rien, quelles que soient ses actions IPC.
- Toute action dont les paramètres demanderaient tenant_id/user_id/sub au modèle les voit retirés (jamais
  rejetée pour autant, contrairement à l'ancien mécanisme — voir FORBIDDEN_MODEL_FIELDS et `_strip_forbidden`) :
  ces champs viennent TOUJOURS de `ctx`, jamais du modèle.
- Une action listée sans schéma réellement enregistré (typo, action jamais décorée `@action`, ou
  `permission_groups` non représentable ici) est IGNORÉE avec un avertissement, jamais exposée avec des
  paramètres inventés ou vides par défaut.
- Un `plugin.yaml` absent, illisible ou mal formé ne fait jamais échouer le démarrage de chat — juste ignoré,
  avec un avertissement dans les journaux.
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
    """Une entrée `agent.actions[]` — lue à plat, AVANT toute résolution contre schema_registry."""

    plugin: str
    action: str
    tool_name: str
    description: str | None  # None = reprendre la description de l'ActionSchema réelle, à la résolution
    agent: str


@dataclass(frozen=True)
class _Agent:
    """L'identité déclarée par le bloc `agent:` d'un plugin — un plugin, un agent.

    `description` et `capabilities` sont ce que DONNA lit pour CHOISIR cet agent (voir `prompt_description`) :
    c'est la seule chose qui distingue deux sous-agents aux yeux du modèle. `version` n'est que de la
    métadonnée, journalisée au démarrage et rendue par `status()` — rien ne s'y branche.
    """

    name: str
    description: str | None = None
    capabilities: tuple[str, ...] = ()
    version: str | None = None

    def prompt_description(self) -> str:
        """Description présentée au modèle : la description, puis les capacités déclarées. Sans elles, un
        `capabilities:` écrit dans un manifeste ne servirait à rien — autant ne pas l'offrir."""
        parts = [self.description.strip()] if self.description and self.description.strip() else []
        if self.capabilities:
            parts.append("Sait : " + ", ".join(self.capabilities) + ".")
        return " ".join(parts)


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


def _declared_from_entry(plugin_name: str, agent: str, entry: dict, seen_tool_names: set[str]) -> _Declared | None:
    """Une entrée `agent.actions[]` -> _Declared, ou None si elle doit être ignorée (avec un avertissement) :
    action absente/invalide, ou tool_name déjà pris par une autre entrée (collision entre deux plugins ou deux
    actions du même plugin)."""
    action = entry.get("action")
    if not isinstance(action, str) or not action:
        logger.warning("entrée agent.actions de '%s' ignorée : `action` absente ou invalide", plugin_name)
        return None
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


def _agent_from_manifest(plugin_name: str, manifest: dict) -> _Agent | None:
    """Le bloc `agent:` d'un manifeste -> _Agent, ou None si ce plugin n'expose rien.

    Absence de bloc = ce plugin n'a rien à exposer, le cas NORMAL (silencieux). Un bloc présent mais sans `name:`
    utilisable est en revanche une faute de frappe qui se voit : avertie, jamais devinée depuis le nom du plugin.
    """
    block = manifest.get("agent")
    if block is None:
        return None
    if not isinstance(block, dict):
        logger.warning("plugin '%s' : `agent:` n'est pas un bloc de clés — ignoré", plugin_name)
        return None
    name = block.get("name")
    if not isinstance(name, str) or not name.strip():
        logger.warning("plugin '%s' : bloc `agent:` sans `name:` utilisable — ignoré", plugin_name)
        return None
    description = block.get("description")
    raw_capabilities = block.get("capabilities")
    capabilities = (
        tuple(str(c).strip() for c in raw_capabilities if str(c).strip())
        if isinstance(raw_capabilities, (list, tuple))
        else ()
    )
    version = block.get("version")
    return _Agent(
        name=name.strip(),
        description=description.strip() if isinstance(description, str) and description.strip() else None,
        capabilities=capabilities,
        version=str(version).strip() if version is not None and str(version).strip() else None,
    )


def discover_declared_actions(apps_root: Path) -> tuple[list[_Declared], dict[str, _Agent]]:
    """Lit le bloc `agent:` de chaque `plugin.yaml` sous `apps_root` (backend/app/ en prod) et renvoie les
    actions déclarées (pas encore résolues contre schema_registry — voir le docstring du module) + l'agent de
    chaque plugin. Jamais levé : un plugin.yaml absent, illisible ou mal formé est ignoré avec un avertissement —
    un plugin tiers en panne ne doit jamais empêcher chat de démarrer."""
    declared: list[_Declared] = []
    agents: dict[str, _Agent] = {}
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
        agent = _agent_from_manifest(plugin_name, manifest)
        if agent is None:
            continue
        if agent.name in agents:
            logger.warning(
                "plugin '%s' : agent '%s' déjà déclaré par un autre plugin — ignoré (un nom d'agent est unique)",
                plugin_name, agent.name,
            )
            continue
        entries = (manifest.get("agent") or {}).get("actions") or []
        if not isinstance(entries, list):
            logger.warning("agent '%s' : `actions:` n'est pas une liste — aucun tool exposé", agent.name)
            entries = []
        built_any = False
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            built = _declared_from_entry(plugin_name, agent.name, entry, seen_tool_names)
            if built is not None:
                declared.append(built)
                built_any = True
        # Un agent sans AUCUNE action exposable n'a rien à proposer au modèle : le déclarer ferait un choix
        # vide dans l'enum de `delegate_to_subagent`.
        if built_any:
            agents[agent.name] = agent
        else:
            logger.warning("agent '%s' (plugin '%s') : aucune action exposable — non déclaré", agent.name, plugin_name)
    return declared, agents


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
