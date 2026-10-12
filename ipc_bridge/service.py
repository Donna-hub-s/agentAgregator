"""Extension xcore — `ext.ipc_bridge` : service qui porte le catalogue d'agents IPC (voir discovery.py pour le
mécanisme et pourquoi il est en deux phases).

Config (integration.yaml de votre app) :
    services:
      extensions:
        ipc_bridge:
          module: ipc_bridge.service:IpcBridgeService
          config:
            apps_root: app  # dossier qui contient les plugin.yaml de vos plugins métier

`apps_root` est OBLIGATOIRE (pas de valeur par défaut devinée) : la structure de dossiers d'une app xcore
n'est pas universelle, mieux vaut échouer tôt et clairement si l'appelant l'a oublié que lire silencieusement
le mauvais dossier (ou pire, le dossier courant du process).

Deux phases — voir discovery.py pour pourquoi elles sont séparées, et quand appeler chacune :
    service = IpcBridgeService({"apps_root": "app"})
    await service.init()        # déclare SEULEMENT (lecture des plugin.yaml, aucune dépendance à un autre plugin)
    ...                         # votre app démarre tous ses plugins
    service.resolve()           # UNE FOIS que TOUS les plugins ont fini de démarrer (voir discovery.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xcore.sdk import get_logger
from xcore.services import BaseService, ServiceStatus

from .discovery import IPCAction, _Declared, discover_declared_actions, resolve_actions

__all__ = ["IpcBridgeService", "IPCAction"]

logger = get_logger("ext.ipc_bridge")


class IpcBridgeService(BaseService):
    """Service partagé `ext.ipc_bridge` — voir le docstring du module et de `discovery.py`."""

    name = "ipc_bridge"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        apps_root = config.get("apps_root")
        if not apps_root:
            raise ValueError("IpcBridgeService : `config.apps_root` est obligatoire (dossier contenant les plugin.yaml).")
        self._apps_root = Path(apps_root)
        self._declared: list[_Declared] = []
        self._descriptions: dict[str, str] = {}
        self._agents: dict[str, tuple[IPCAction, ...]] = {}

    async def init(self) -> None:
        # Phase 1 seulement (voir discovery.py) : lecture à plat des plugin.yaml, aucune dépendance à l'état
        # d'un autre plugin. La résolution contre schema_registry (phase 2) attend resolve() — appelez-la une
        # fois que TOUS les plugins de votre app ont fini de démarrer (ex. un handler sur l'évènement
        # xcore.plugins.booted), jamais ici : trop tôt dans le démarrage.
        self._declare()
        self._status = ServiceStatus.READY

    def _declare(self) -> None:
        try:
            self._declared, self._descriptions = discover_declared_actions(self._apps_root)
        except Exception as exc:  # filet ultime : un plugin tiers mal formé ne doit jamais bloquer le démarrage
            logger.warning("découverte des agents IPC (déclaration) échouée, catalogue vide : %s", exc)
            self._declared, self._descriptions = [], {}

    def resolve(self, registry: Any = None) -> None:
        """A appeler UNE FOIS que tous les plugins de votre app ont fini de démarrer — voir discovery.py pour
        pourquoi plus tôt serait silencieusement faux. Jamais levé : au pire `agents` reste vide ou incomplet."""
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
        }

    @property
    def agents(self) -> dict[str, tuple[IPCAction, ...]]:
        return self._agents

    @property
    def descriptions(self) -> dict[str, str]:
        return self._descriptions
