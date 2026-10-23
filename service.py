"""Extension xcore — `ext.ipc_bridge` : service qui porte le catalogue d'agents IPC (voir discovery.py pour le
mécanisme et pourquoi il est en deux phases).

Config (integration.yaml) :
    services:
      extensions:
        ipc_bridge:
          module: extensions.ipc_bridge.service:IpcBridgeService
          config: {}

`config.apps_root` est surtout pour les tests (pointer vers un `tmp_path` au lieu de `backend/app/`) ; en
production, laissé vide, il résout tout seul vers le dossier frère `app/` de `backend/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xcore.sdk import get_logger
from xcore.services import BaseService, ServiceStatus

from .discovery import IPCAction, _Agent, _Declared, discover_declared_actions, resolve_actions

__all__ = ["IpcBridgeService", "IPCAction"]

logger = get_logger("ext.ipc_bridge")


def _default_apps_root() -> Path:
    # extensions/ipc_bridge/service.py -> extensions -> backend ; tous les plugins DONNA sont sous backend/app/.
    return Path(__file__).resolve().parents[2] / "app"


class IpcBridgeService(BaseService):
    """Service partagé `ext.ipc_bridge` — voir le docstring du module et de `discovery.py`."""

    name = "ipc_bridge"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        apps_root = config.get("apps_root")
        self._apps_root = Path(apps_root) if apps_root else _default_apps_root()
        self._declared: list[_Declared] = []
        self._declared_agents: dict[str, _Agent] = {}
        self._agents: dict[str, tuple[IPCAction, ...]] = {}

    async def init(self) -> None:
        # Phase 1 seulement (voir discovery.py) : lecture à plat des plugin.yaml, aucune dépendance à l'état
        # d'un autre plugin. La résolution contre schema_registry (phase 2) attend `resolve()`, appelé par le
        # plugin chat une fois l'évènement xcore.plugins.booted reçu — jamais ici, trop tôt dans le démarrage.
        self._declare()
        self._status = ServiceStatus.READY

    def _declare(self) -> None:
        try:
            self._declared, self._declared_agents = discover_declared_actions(self._apps_root)
        except Exception as exc:  # filet ultime : jamais bloquer le démarrage de chat pour un plugin tiers
            logger.warning("découverte des agents IPC (déclaration) échouée, catalogue vide : %s", exc)
            self._declared, self._declared_agents = [], {}
        for agent in self._declared_agents.values():
            logger.info("agent IPC déclaré", agent=agent.name, version=agent.version or "-")

    def resolve(self, registry: Any = None) -> None:
        """A appeler UNE FOIS que tous les plugins ont fini de démarrer (voir chat/src/main.py, abonné à
        xcore.plugins.booted) — voir discovery.py pour pourquoi plus tôt serait silencieusement faux. Jamais
        levé : au pire `agents` reste vide ou incomplet."""
        try:
            self._agents = resolve_actions(self._declared, registry)
        except Exception as exc:
            logger.warning("résolution des agents IPC échouée, catalogue vide : %s", exc)
            self._agents = {}
        if self._agents:
            logger.info(
                "agents IPC résolus : %s",
                {agent: [a.tool_name for a in actions] for agent, actions in self._agents.items()},
            )

    async def shutdown(self) -> None:
        self._status = ServiceStatus.STOPPED

    async def health_check(self) -> tuple[bool, str]:
        return True, f"{len(self._agents)} agent(s) IPC résolu(s), {len(self._declared)} action(s) déclarée(s)"

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self._status.value,
            "declared": len(self._declared),
            "agents": sorted(self._agents),
            # Version déclarée par chaque agent résolu : métadonnée du manifeste, utile pour savoir
            # quelle révision d'un plugin métier est réellement en place.
            "versions": {
                name: self._declared_agents[name].version
                for name in sorted(self._agents)
                if name in self._declared_agents and self._declared_agents[name].version
            },
        }

    @property
    def agents(self) -> dict[str, tuple[IPCAction, ...]]:
        return self._agents

    @property
    def routing(self) -> dict[str, dict[str, str]]:
        """Le bloc `routing:` déclaré par chaque agent RÉSOLU, transporté tel quel.

        Consommé par `app/chat/src/agent_registry.py`, qui seul sait ce qu'est un « niveau ».
        Même filtre que `descriptions` : un agent dont aucune action n'a survécu à la
        résolution n'existe pas pour le modèle, sa politique n'a donc rien à régler.
        """
        return {
            name: dict(agent.routing)
            for name, agent in self._declared_agents.items()
            if name in self._agents and agent.routing
        }

    @property
    def descriptions(self) -> dict[str, str]:
        """Ce que DONNA lit pour CHOISIR un sous-agent (description + capacités déclarées). Seuls les agents
        RÉSOLUS y figurent : annoncer au modèle un agent dont aucune action n'a survécu à la résolution lui
        offrirait un choix qui échoue."""
        return {
            name: text
            for name, agent in self._declared_agents.items()
            if name in self._agents and (text := agent.prompt_description())
        }
