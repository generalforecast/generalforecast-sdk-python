# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: MIT

import pytest

from generalforecast import cli


def test_the_entry_point_prints_help_and_version(capsys):
    cli.main(["--help"])
    assert capsys.readouterr().out.startswith("usage: gf <resource> <verb>")
    cli.main(["-V"])
    assert capsys.readouterr().out.startswith("gf ")


def test_commands_point_at_the_python_api(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["contexts", "list"])
    assert stopped.value.code == 2
    assert "import generalforecast as gf" in capsys.readouterr().err
