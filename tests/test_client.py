# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

import pickle
import ssl

import certifi
import pytest

import generalforecast as gf
from generalforecast import _auth, _http, _origin, client, errors
from tests.fake import TOKEN


def test_the_project_is_learned_from_the_key_and_every_request_says_its_origin(fake):
    gf.contexts.list()
    status, listing = fake.calls[:2]
    assert (
        status.path == "/v1/auth/status"
        and listing.path == "/v1/projects/acme/contexts"
    )
    assert listing.headers["x-gf-project"] == "acme"
    origin = listing.headers["x-gf-origin"]
    assert origin.startswith("script; client=generalforecast/")
    assert listing.headers["user-agent"].startswith("generalforecast-python/")
    gf.contexts.list()
    assert len(fake.calls_to("GET", "/v1/auth/status")) == 1


def test_gf_project_skips_the_lookup(fake, env):
    env.setenv("GF_PROJECT", "acme")
    gf.contexts.list()
    assert not fake.calls_to("GET", "/v1/auth/status")


def test_notebook_origin_names_the_tool(monkeypatch):
    monkeypatch.setattr(_origin, "_in_ipython", lambda: True)
    for env, tool in (
        ({"COLAB_RELEASE_TAG": "x"}, "colab"),
        ({"DATABRICKS_RUNTIME_VERSION": "15"}, "databricks"),
        ({"VSCODE_PID": "1"}, "vscode"),
        ({}, "jupyter"),
    ):
        found = _origin.detect(env)
        assert found == _origin.Detected("notebook", tool)
        assert found.header("generalforecast/1.0.0") == (
            f"notebook; notebook={tool}; client=generalforecast/1.0.0"
        )


def test_credentials_resolve_in_order(env, tmp_path):
    env.setenv("GF_URL", "http://127.0.0.1:9")
    assert _auth.resolve(None, None, None).source == "none"
    store = _auth.CredentialStore()
    store.save(
        _auth.Stored("http://127.0.0.1:9", "key-1", project="acme", key="from-file")
    )
    found = _auth.resolve(None, None, None)
    assert (found.token, found.project, found.source) == (
        "from-file",
        "acme",
        "credentials",
    )
    env.setenv("GF_API_KEY", "from-env")
    assert _auth.resolve(None, None, None).token == "from-env"
    env.setattr(
        _auth, "SESSION", _auth.Session("http://127.0.0.1:9", "from-login", "beta")
    )
    found = _auth.resolve(None, None, None)
    assert (found.token, found.project) == ("from-login", "beta")
    env.setenv("GF_PROJECT", "gamma")
    assert _auth.resolve(None, None, None).project == "gamma"
    assert _auth.resolve("explicit", "delta", None).token == "explicit"
    assert (
        store.remove("http://127.0.0.1:9") and store.load("http://127.0.0.1:9") is None
    )


def test_not_logged_in_says_how(env):
    env.setenv("GF_URL", "http://127.0.0.1:9")
    with pytest.raises(gf.Unauthorized) as error:
        gf.contexts.list()
    assert error.value.message == "not logged in"
    assert error.value.hint == "run gf.login(), or set GF_API_KEY for unattended use"


def test_transient_failures_of_safe_requests_are_retried(fake):
    fake.fail["contexts_list"] = 2
    assert gf.contexts.list() == []
    assert len(fake.calls_to("GET", r".*/contexts")) == 3
    fake.fail["keys_create"] = 1
    with pytest.raises(gf.Unavailable) as error:
        gf.keys.create("ci")
    assert not error.value.recoverable
    assert len(fake.calls_to("POST", r".*/keys")) == 1


def test_a_429_is_retried_only_when_it_can_pass(fake):
    fake.throttle["runs_list"] = ["rate_limited"]
    assert gf.fine_tunes.runs.list() == []
    assert len(fake.calls_to("GET", r".*/runs")) == 2
    # A standing limit lifts only once the caller ends a run or deletes data.
    for reason in ("run_limit_reached", "quota_exceeded", "snapshot_quota_exceeded"):
        fake.throttle["runs_list"] = [reason]
        with pytest.raises(gf.CapacityExceeded) as error:
            gf.fine_tunes.runs.list()
        assert error.value.reason == reason and error.value.limit == 1
    assert len(fake.calls_to("GET", r".*/runs")) == 5


def test_proxies_from_the_environment_are_honoured_unless_trust_env_is_false(fake, env):
    env.setenv("GF_PROJECT", "acme")
    env.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    with gf.Client(retries=0) as proxied, pytest.raises(gf.Unavailable):
        proxied.contexts.list()
    with gf.Client(trust_env=False) as direct:
        assert direct.contexts.list() == []


def test_the_system_ca_bundle_and_named_bundles_are_trusted(env):
    env.setenv("SSL_CERT_FILE", certifi.where())
    context = _http.ssl_context(True)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert context.cert_store_stats()["x509_ca"] > 0


@pytest.mark.parametrize(
    "url", ["http://example.com", "ftp://x", "https://u:p@x.com", "nope"]
)
def test_credentials_never_go_in_clear(url):
    with pytest.raises(gf.InvalidInput):
        _http.server(url)


def test_errors_carry_recoverable_and_survive_pickling():
    for cls in (gf.InvalidInput, gf.NotFound, gf.Conflict, gf.Unauthorized, gf.PaymentRequired,
                gf.CapacityExceeded, gf.ModelNotReady, gf.AmbiguousReference, gf.Unsupported,
                gf.Timeout):  # fmt: skip
        assert cls("x").recoverable
    assert not gf.Internal("x").recoverable and not gf.Unavailable("x").recoverable
    error = errors.from_wire(
        {
            "code": "Conflict",
            "message": "in use",
            "reason": "context_in_use",
            "hint": "h",
        },
        409,
        "r-1",
    )
    assert isinstance(error, gf.Conflict) and error.request_id == "r-1"
    assert str(error) == "in use\nhelp: h\nrequest: r-1"
    copy = pickle.loads(pickle.dumps(error))
    assert isinstance(copy, gf.Conflict) and copy.reason == "context_in_use"
    assert isinstance(errors.from_wire({"code": "Weird"}, 503), gf.Unavailable)
    assert isinstance(errors.from_wire(None, 404), gf.NotFound)
    assert errors.from_wire(None, 418).message == "the service answered 418"
    failed = errors.failed(
        "run 1 failed", {"code": "CapacityExceeded", "message": "oom"}, code="RunFailed"
    )
    assert isinstance(failed, gf.CapacityExceeded) and not failed.recoverable


def test_login_runs_the_device_flow_and_holds_the_token_in_memory(
    fake, env, capsys, tmp_path
):
    env.delenv("GF_API_KEY")
    me = gf.login()
    out = capsys.readouterr().out
    assert f"Open {fake.url}/device and enter the code WXHG-PTRB" in out
    assert (
        "Logged in as maya@acme.com, project acme. This session holds a token until"
        in out
    )
    assert me.email == "maya@acme.com" and me.workspace == "Acme Analytics"
    assert me.token_expires is not None and me.project == "acme"
    assert _auth.SESSION.token == TOKEN and _auth.SESSION.minted
    assert not (tmp_path / "config").exists()
    assert gf.whoami().email == "maya@acme.com"
    assert fake.calls_to("GET", "/v1/auth/status")[-1].headers.get("authorization")
    gf.logout()
    assert _auth.SESSION is None and fake.calls_to("POST", "/v1/auth/logout")


def test_a_failed_revoke_on_logout_is_raised_once_the_login_is_forgotten(fake, env):
    env.delenv("GF_API_KEY")
    gf.login()
    fake.fail["auth_logout"] = 1
    with pytest.raises(gf.Unavailable):
        gf.logout()
    assert _auth.SESSION is None


def test_a_key_that_never_expires_reads_as_one(fake):
    me = gf.whoami()
    assert me.key.name == "notebook-maya-mbp" and me.key.expires is None
    assert me.token_expires is None and me.key.state == "active"


def test_an_interrupted_login_asks_for_a_key(fake, env, monkeypatch, capsys):
    env.delenv("GF_API_KEY")

    def interrupted(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(client.time, "sleep", interrupted)
    asked = []
    monkeypatch.setattr(
        client.getpass, "getpass", lambda prompt: asked.append(prompt) or TOKEN
    )
    me = gf.login(project="acme")
    assert asked == ["Paste an API key instead: "]
    assert me.project == "acme" and not _auth.SESSION.minted
    gf.logout()
    assert not fake.calls_to("POST", "/v1/auth/logout")


def test_login_with_a_key_skips_the_flow(fake, env, capsys):
    env.delenv("GF_API_KEY")
    gf.login(key=TOKEN)
    assert not fake.calls_to("POST", "/v1/device-logins")
    assert gf.contexts.list() == []
    assert capsys.readouterr().out == "Logged in as maya@acme.com, project acme.\n"


def test_a_closed_client_refuses_calls(fake):
    c = gf.Client()
    c.close()
    with pytest.raises(gf.InvalidInput):
        c.contexts.list()
    assert repr(c).startswith("Client(url='http://127.0.0.1:")
