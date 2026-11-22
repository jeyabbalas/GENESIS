"""The ``genesis`` command-line interface, exercised in-process on the packaged chr22 data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import genesis_gwas as gg
from genesis_gwas.cli.main import build_parser, main
from genesis_gwas.hpc.swarm import swarm_command, trait_name, write_swarm


@pytest.fixture(scope="module")
def example():
    return str(gg.load_reference("mini_chr22").example_path())


@pytest.fixture(scope="module")
def fitted(example, tmp_path_factory):
    """One `genesis fit` run shared by the tests of the downstream commands."""
    out = tmp_path_factory.mktemp("cli") / "run" / "height"
    code = main(
        [
            "fit",
            example,
            "--reference",
            "mini_chr22",
            "--out",
            str(out),
            "--model",
            "2",
            "--threads",
            "2",
            "--print-every",
            "0",
            "--quiet",
            "--project",
            "2e5,1e6",
            "--posterior",
            "--qq",
            "--qq-sims",
            "5",
        ]
    )
    assert code == 0
    return out


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0 and gg.__version__ in capsys.readouterr().out
    assert main([]) == 2
    assert "fit" in capsys.readouterr().out
    for cmd in (
        "fit",
        "project",
        "predict",
        "posterior",
        "qq",
        "plot",
        "data",
        "swarm",
        "demo",
        "doctor",
    ):
        with pytest.raises(SystemExit) as exc:
            main([cmd, "--help"])
        assert exc.value.code == 0
    with pytest.raises(SystemExit):
        main(["ld", "build", "--help"])
    assert build_parser().prog == "genesis"


def test_fit_writes_all_outputs(fitted):
    files = {p.name for p in fitted.parent.iterdir()}
    expected = {
        "height.M2.json",
        "height.best.json",
        "height.estimates.tsv",
        "height.summary.txt",
        "height.log",
        "height.projection.tsv",
        "height.posterior.tsv.gz",
        "height.qq.png",
        "height.qq.tsv.gz",
    }
    assert expected <= files
    res = gg.FitResult.from_json(f"{fitted}.M2.json")
    assert res.model == "M2" and res.converged and res.n_snps > 12000
    best = json.loads(Path(f"{fitted}.best.json").read_text())
    assert best["model"] == "M2" and "history" not in best

    est = pd.read_csv(f"{fitted}.estimates.tsv", sep="\t")
    assert list(est.columns) == ["model", "selected", "quantity", "label", "estimate", "se"]
    row = est.set_index("quantity").loc["pic"]
    assert row["estimate"] == pytest.approx(res.pic, rel=1e-8) and row["selected"] == 1
    assert {"heritability", "n_ssnp", "loglik", "bic", "n_snps"} <= set(est["quantity"])

    summary = Path(f"{fitted}.summary.txt").read_text()
    assert (
        "GENESIS fit: M2" in summary
        and "Quality control" in summary
        and "chi-square > 80" in summary
    )
    proj = pd.read_csv(f"{fitted}.projection.tsv", sep="\t")
    assert proj["n"].tolist() == [2e5, 1e6] and "n_discoveries_lower" in proj.columns
    post = pd.read_csv(f"{fitted}.posterior.tsv.gz", sep="\t")
    assert len(post) == res.n_snps and post["p_null"].between(0, 1).all()
    qq = pd.read_csv(f"{fitted}.qq.tsv.gz", sep="\t")
    assert {"observed", "expected", "lower", "upper"} <= set(qq.columns)
    log = Path(f"{fitted}.log").read_text()  # complete even though the terminal was --quiet
    assert "genesis-gwas" in log and "Command: genesis fit" in log and "Prepared" in log


def test_downstream_commands(fitted, example, tmp_path, capsys):
    fit_json = f"{fitted}.M2.json"
    assert main(["project", fit_json, "--n", "1e5,5e5", "--quiet"]) == 0
    table = capsys.readouterr().out.strip().splitlines()
    assert table[0].split("\t")[:2] == ["n", "heritability"] and len(table) == 3

    out = tmp_path / "pred.tsv"
    assert (
        main(
            ["predict", fit_json, "--n", "1e5", "--threshold", "gwas", "--out", str(out), "--quiet"]
        )
        == 0
    )
    pred = pd.read_csv(out, sep="\t")
    assert 0.5 <= pred["auc"].iloc[0] < 1 and pred["alpha"].iloc[0] == 5e-8

    post = tmp_path / "post.tsv.gz"
    args = [
        "posterior",
        example,
        "--fit",
        fit_json,
        "--reference",
        "mini_chr22",
        "--out",
        str(post),
        "--quiet",
        "--threads",
        "2",
    ]
    assert main(args) == 0
    a = pd.read_csv(post, sep="\t")
    assert main([*args, "--ignore-inflation"]) == 0
    b = pd.read_csv(post, sep="\t")
    assert len(a) == len(b) and not np.allclose(a["p_null"], b["p_null"])

    assert (
        main(
            [
                "qq",
                example,
                "--fit",
                fit_json,
                "--reference",
                "mini_chr22",
                "--out",
                str(tmp_path / "q"),
                "--n-sim",
                "3",
                "--quiet",
            ]
        )
        == 0
    )
    assert (tmp_path / "q.qq.png").stat().st_size > 5000
    for kind in ("density", "discoveries", "variance"):
        img = tmp_path / f"{kind}.png"
        assert main(["plot", kind, f"mine={fit_json}", "--out", str(img), "--quiet"]) == 0
        assert img.stat().st_size > 5000


def test_fit_both_models_legacy_and_options(example, tmp_path):
    out = tmp_path / "both"
    code = main(
        [
            "fit",
            example,
            "--reference",
            "mini_chr22",
            "--out",
            str(out),
            "--no-qc",
            "--legacy",
            "--max-iter",
            "40",
            "--threads",
            "2",
            "--print-every",
            "20",
            "--quiet",
            "--population-prevalence",
            "0.01",
            "--sample-prevalence",
            "0.5",
            "--starting-pic",
            "0.02",
        ]
    )
    assert code == 0
    m2 = gg.FitResult.from_json(f"{out}.M2.json")
    m3 = gg.FitResult.from_json(f"{out}.M3.json")
    assert m2.settings["legacy"] and m2.settings["sandwich_double_count"] and m2.n_iter == 40
    assert m2.n_snps == 13230  # --no-qc keeps every reference SNP
    assert m3.starting[1] == pytest.approx(1 / 9) and m3.starting[0] == pytest.approx(
        m2.estimates[0]
    )
    est = pd.read_csv(f"{out}.estimates.tsv", sep="\t")
    assert set(est["model"]) == {"M2", "M3"} and "heritability_liability" in set(est["quantity"])
    assert est.groupby("model")["selected"].max().sum() == 1


def test_errors_are_reported_cleanly(example, tmp_path, capsys):
    assert (
        main(
            [
                "fit",
                str(tmp_path / "missing.tsv"),
                "--reference",
                "mini_chr22",
                "--out",
                str(tmp_path / "x"),
            ]
        )
        == 1
    )
    assert "not found" in capsys.readouterr().err
    assert (
        main(
            ["fit", example, "--reference", "no_such_ref", "--out", str(tmp_path / "x"), "--quiet"]
        )
        == 1
    )
    assert (
        main(
            [
                "fit",
                example,
                "--reference",
                "mini_chr22",
                "--out",
                str(tmp_path / "x"),
                "--starting",
                "0.1,0.2",
                "--quiet",
            ]
        )
        == 1
    )
    assert main(["project", str(tmp_path / "nope.json"), "--n", "1e5"]) == 1
    assert (
        main(
            [
                "fit",
                example,
                "--reference",
                "mini_chr22",
                "--out",
                str(tmp_path / "x"),
                "--device",
                "cuda:7",
                "--quiet",
            ]
        )
        == 1
    )
    with pytest.raises(SystemExit) as exc:
        main(["fit", example])  # --out is required
    assert exc.value.code == 2


def test_data_commands(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GENESIS_DATA_DIR", str(tmp_path))
    assert main(["data", "list"]) == 0
    out = capsys.readouterr().out
    assert "eur_1kg_hm3" in out and "mini_chr22" in out and "w1000kb_r0.1" in out
    root = gg.load_reference("mini_chr22").root
    assert main(["data", "verify", str(root)]) == 0
    assert "ok" in capsys.readouterr().out
    assert main(["data", "verify", str(tmp_path)]) == 1


def test_demo_and_doctor(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path / "demo"), "--threads", "2", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "GENESIS fit: M2" in out and "Model comparison" in out
    assert (tmp_path / "demo.M2.json").exists() and (tmp_path / "demo.M3.json").exists()
    assert main(["doctor", "--json", "--threads", "2", "--quiet"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["genesis_gwas"] == gg.__version__
    assert all(c["ok"] for c in report["self_test"]) and len(report["self_test"]) >= 2
    assert main(["doctor", "--threads", "2", "--quiet"]) == 0
    assert "All backends agree" in capsys.readouterr().out
    args = ["doctor", "--threads", "2", "--quiet", "--benchmark", "--benchmark-snps", "5000"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "ms per pass" in out and "5,000 synthetic SNPs" in out
    assert main([*args, "--json"]) == 0
    bench = json.loads(capsys.readouterr().out)["benchmark"]
    assert bench and all(b["seconds_per_pass"] > 0 for b in bench)


# ----------------------------------------------------------------------- swarm
def test_trait_names():
    assert trait_name("/data/x/BCAC_overall.sumstats.tsv.gz") == "BCAC_overall"
    assert trait_name("er neg (v2).txt.bz2") == "er_neg__v2_"
    assert trait_name("a.b.c.csv") == "a.b.c"
    assert trait_name(".gz") == ".gz"


def test_swarm_file_and_command(tmp_path):
    files = ["/data/me/gwas/ERpos.tsv.gz", "/data/me/gwas/ER neg.tsv.gz"]
    cmd = write_swarm(
        files,
        outdir="/data/me/results",
        swarmfile=tmp_path / "jobs" / "g.swarm",
        fit_args=["--model", "both", "--n-cases", "NCASE"],
        cpus=8,
        mem_gb=12,
        walltime="02:00:00",
        activate="source /data/me/conda/etc/profile.d/conda.sh && conda activate genesis",
    )
    lines = [
        ln
        for ln in (tmp_path / "jobs" / "g.swarm").read_text().splitlines()
        if not ln.startswith("#")
    ]
    assert len(lines) == 2
    assert lines[0].startswith(
        "source /data/me/conda/etc/profile.d/conda.sh && conda activate genesis && genesis fit"
    )
    assert "--out /data/me/results/ERpos --device cpu --model both --n-cases NCASE" in lines[0]
    assert "'/data/me/gwas/ER neg.tsv.gz'" in lines[1] and "/data/me/results/ER_neg" in lines[1]
    assert (
        cmd.startswith("swarm -f ")
        and "-g 12 -t 8 --time 02:00:00" in cmd
        and "--logdir /data/me/results/logs" in cmd
    )
    assert "gpu" not in cmd

    gpu = write_swarm(files, outdir="out", swarmfile=tmp_path / "gpu.swarm", gpu=True)
    assert (
        "--partition gpu --gres gpu:1" in gpu
        and "--device cuda" in (tmp_path / "gpu.swarm").read_text()
    )
    assert swarm_command("f.swarm", cpus=4, mem_gb=8).split()[:3] == ["swarm", "-f", "f.swarm"]

    from genesis_gwas.exceptions import InputError

    with pytest.raises(InputError, match="same output prefix"):
        write_swarm(["a/x.tsv", "b/x.tsv.gz"], outdir="o", swarmfile=tmp_path / "d.swarm")
    with pytest.raises(InputError):
        write_swarm([], outdir="o", swarmfile=tmp_path / "e.swarm")


def test_swarm_command_line(tmp_path, capsys):
    sf = tmp_path / "t.swarm"
    code = main(
        [
            "swarm",
            "a.tsv.gz",
            "b.tsv.gz",
            "--outdir",
            "res",
            "--swarmfile",
            str(sf),
            "--cpus",
            "4",
            "--quiet",
            "--",
            "--model",
            "2",
            "--no-qc",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.strip().startswith("swarm -f")
    body = sf.read_text()
    assert body.count("genesis fit") == 2 and "--model 2 --no-qc" in body
