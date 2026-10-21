# agentAgregator

Pont générique entre une **action IPC** d'un plugin xcore et un **sous-agent** appelable par
le modèle. Un plugin métier s'expose comme agent — « un plugin, un agent, ses outils » — sans
qu'une seule ligne de Python soit écrite côté assistant.

## Déclaration

Dans le `plugin.yaml` du plugin, un bloc `agent:` — c'est lui, et lui seul, qui active
l'exposition :

```yaml
agent:
  name: "workflow_manager"                 # obligatoire ; unique tous plugins confondus
  description: "Consulte et met en pause les automatisations."   # lu pour CHOISIR l'agent
  version: "1.0.0"
  capabilities:                            # ajoutées à la description vue par le modèle
    - "lister les automatisations"
  actions:
    - action: "workflows.list"             # obligatoire : ce que call_plugin adressera
      tool_name: "list_workflows"          # facultatif (défaut : l'action, points → souligné)
      description: "Liste les workflows de l'appelant."   # facultatif (défaut : celle du @schema)
```

Les **paramètres**, **permissions** et **niveau de risque** ne sont jamais écrits ici : ils
sont résolus depuis `schema_registry` (les `@action`/`@schema` réels du plugin). Les recopier
dans le YAML les ferait dériver en silence.

## Deux phases, délibérément séparées

1. **Déclaration** — lecture à plat des `plugin.yaml`, sans dépendre de l'état d'un autre
   plugin. Peut tourner à n'importe quel moment du démarrage.
2. **Résolution** — recherche du vrai schéma dans `schema_registry`. Ne peut tourner qu'une
   fois **tous** les plugins démarrés (`xcore.plugins.booted`) : un plugin chargé après
   celui-ci n'a pas encore enregistré ses schémas, et on construirait des outils aux
   paramètres vides en silence.

## Ce qui n'est pas automatique (sécurité délibérée)

- Sans bloc `agent:`, un plugin n'expose **rien**, quelles que soient ses actions IPC.
- `tenant_id`, `user_id` et `sub` sont **retirés** des paramètres montrés au modèle, quoi que
  dise le schéma réel : ils viennent toujours du contexte d'exécution, jamais d'un argument
  choisi par le modèle.
- Une action sans `@action`/`@schema` enregistrés est ignorée avec un avertissement — jamais
  exposée avec des paramètres devinés.
- `side_effect` non déclaré = traité comme `outbound`, le risque le plus élevé.
- Un agent dont aucune action n'est exposable n'est pas déclaré : il ferait un choix vide.

## Place dans DONNA

Composant de [DONNA](https://github.com/Donna-hub-s), monté comme sous-module à
`backend/extensions/ipc_bridge/`.

## Licence

MIT — voir [LICENSE](LICENSE).
