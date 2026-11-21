# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

"""The gf command table: groups, commands, admin nouns, their actions and defaults.

The parser, the dispatcher and the renderer all read these tables, so a noun
or an action is named in exactly one place. Kept free of heavy imports so
`gf --help` and argument errors stay fast.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generalforecast.service.boundary import ResourceKind

TAGLINE = "General Forecast: Inference for structured data foundation models"
CLOSING = "See 'gf help <command>' for more information"

# `gf --help`: three groups, one line per command, in this order.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Predictors", ("init", "check", "eval", "predict")),
    ("Hosted", ("login", "logout", "whoami", "push", "status", "promote", "rollback")),
    ("Resources", ("model", "context", "admin")),
)

COMMAND_HELP = {
    "init": "Write a predictor manifest here",
    "check": "Resolve gf.yaml on the service and write gf.lock",
    "eval": "Score the held-out rows on the service",
    "predict": "Score rows with a hosted predictor or a catalog model",
    "login": "Log in through your browser",
    "logout": "Remove this machine's credentials",
    "whoami": "Show the account and project in use",
    "push": "Deploy a predictor to an environment",
    "status": "Show a predictor's environments",
    "promote": "Point an environment at another's revision",
    "rollback": "Point an environment at an earlier revision",
    "model": "Catalog models: list, show",
    "context": "Live contexts: create, append, label, show",
    "admin": "Keys, secrets, projects and hosted predictors",
}

COMMANDS = tuple(name for _, names in GROUPS for name in names)

MODEL_ACTIONS: tuple[tuple[str, str], ...] = (
    ("list", "List catalog models"),
    ("show", "Show one model's task and limits"),
)

CONTEXT_ACTIONS: tuple[tuple[str, str], ...] = (
    ("create", "Create a live context from a config file"),
    ("append", "Append rows from a file"),
    ("label", "Attach target values from a file"),
    ("show", "Show a live context"),
)

# Commands outside gf, each with the one line it answers: models run on the
# hosted service, so nothing serves or downloads on this machine.
ELSEWHERE = {
    "serve": "local serving is not part of gf; push the predictor with "
    "`gf push DIR` and score rows with `gf predict NAME`",
    "model pull": "local serving is not part of gf, so models do not download; "
    "score with `gf predict --model ID` or push a predictor with `gf push DIR`",
    "admin node": "local serving is not part of gf; predictors run on the hosted "
    "service: push one with `gf push DIR`",
    "admin router": "local serving is not part of gf; predictors run on the hosted "
    "service: push one with `gf push DIR`",
}
# The hosted environment a command uses when none is named.
DEFAULT_ENV = "production"
# Seconds `push`, `promote` and `rollback` wait for an environment to be ready.
READY_SECONDS = 600.0
# Seconds between polls of a hosted evaluation.
EVAL_POLL_SECONDS = 2.0


@dataclass(frozen=True)
class AdminAction:
    """One `gf admin <noun> <action>`.

    `target` names what the NAME argument is, or is None for an action that
    takes none. An action with a `body` reads its request from `--file`; one
    without takes no file. A `destructive` action asks before it runs unless
    `--yes`.
    """

    name: str
    help: str
    target: str | None = None
    body: bool = False
    destructive: bool = False


# `gf admin <noun> <action>` over the API: everything an operator needs and a
# user rarely does.
ADMIN_ACTIONS: dict[str, tuple[AdminAction, ...]] = {
    "key": (
        AdminAction("list", "List API keys"),
        AdminAction("create", "Create an API key from --file", body=True),
        AdminAction("revoke", "Revoke an API key", target="key id", destructive=True),
        AdminAction("rotate", "Rotate an API key", target="key id", destructive=True),
    ),
    "secret": (
        AdminAction("list", "List secrets"),
        AdminAction("get", "Show one secret's metadata", target="secret"),
        AdminAction("create", "Create a secret from --file", body=True),
        AdminAction(
            "rotate", "Rotate a secret from --file", target="secret", body=True
        ),
        AdminAction("delete", "Delete a secret", target="secret", destructive=True),
    ),
    "workspace": (AdminAction("get", "Show the workspace"),),
    "project": (
        AdminAction("list", "List projects"),
        AdminAction("create", "Create a project from --file", body=True),
    ),
    "organization": (
        AdminAction("list", "List organizations"),
        AdminAction("create", "Create an organization from --file", body=True),
    ),
    "sso": (
        AdminAction("get", "Show the single sign-on settings"),
        AdminAction("configure", "Configure single sign-on from --file", body=True),
    ),
    "audit": (AdminAction("list", "List audit events"),),
    "source": (
        AdminAction("list", "List pinned sources"),
        AdminAction("pin", "Pin a source from --file", body=True),
        AdminAction("prepare", "Show the identity to grant a data source of --kind"),
    ),
}

# The kinds of data source `gf admin source prepare --kind` answers for.
SOURCE_KINDS = ("bigquery", "gcs", "snowflake")

# The request field an action's NAME fills, per noun.
NAME_FIELDS = {"key": "id", "secret": "name"}

# Admin nouns that act on hosted predictors rather than API collections.
ADMIN_PREDICTOR_NOUNS = ("logs", "diff", "delete")

ADMIN_HELP = {
    "key": "API keys",
    "secret": "Project secrets",
    "workspace": "The current workspace",
    "project": "Projects",
    "organization": "Organizations",
    "sso": "Single sign-on settings",
    "audit": "Audit log",
    "source": "Data sources and pinned files",
    "logs": "Show an environment's logs",
    "diff": "Compare a predictor with a revision",
    "delete": "Delete a hosted predictor",
}

ADMIN_NOUNS = tuple(ADMIN_HELP)


def admin_action(noun: str, action: str) -> AdminAction:
    """The table entry for `gf admin <noun> <action>`."""
    return next(spec for spec in ADMIN_ACTIONS[noun] if spec.name == action)


# The SDK routing tables, which the hosted test suite shares.

COLLECTIONS: "dict[str, tuple[ResourceKind, str]]" = {
    "context": ("context", "contexts"),
    "predictor": ("predictor", "predictors"),
    "source": ("source", "sources"),
    "model": ("model", "models"),
    "key": ("api-key", "keys"),
    "secret": ("secret", "secrets"),
    "auth": ("workspace", "auth"),
    "workspace": ("workspace", "workspace"),
    "organization": ("workspace", "organizations"),
    "project": ("project", "projects"),
    "sso": ("workspace", "sso"),
    "audit": ("workspace", "audit"),
}

OPERATIONS: dict[str, set[str]] = {
    "context": {"list", "get", "create", "mutate", "snapshot"},
    "predictor": {"list", "get"},
    "source": {"list", "pin", "prepare"},
    "model": {"list", "get"},
    "key": {"list", "create", "revoke", "rotate"},
    "secret": {"list", "get", "create", "rotate", "delete"},
    "auth": {"status"},
    "workspace": {"get"},
    "organization": {"list", "create"},
    "project": {"list", "create"},
    "sso": {"get", "configure"},
    "audit": {"list"},
}

IDENTITY_GROUPS = {
    "key": "keys",
    "secret": "secrets",
    "auth": "auth",
    "workspace": "workspace",
    "organization": "organizations",
    "project": "projects",
    "sso": "sso",
    "audit": "audit",
}
