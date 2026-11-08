"""配置解析、端口选择和出图规则。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import skrf as rf
import yaml
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_sparam_curves as plot


def _s_matrix(nfreq: int) -> np.ndarray:
    s = np.zeros((nfreq, 16, 16), dtype=complex)
    for port in range(1, 17):
        db = -10 - (port - 1)
        s[:, port - 1, port - 1] = 10 ** (db / 20.0) * np.exp(1j * (0.01 * port))
    for rx in range(1, 17):
        for tx in range(1, 17):
            if rx == tx:
                continue
            db = -30 - abs(rx - tx)
            s[:, rx - 1, tx - 1] = 10 ** (db / 20.0) * np.exp(1j * (0.1 * rx))
    return s


def write_s16p(directory: Path, stem: str, f0: float = 0.5, f1: float = 3.0, nfreq: int = 6) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    freq = rf.Frequency(f0, f1, nfreq, unit="ghz")
    network = rf.Network(frequency=freq, s=_s_matrix(nfreq), name=stem)
    network.write_touchstone(filename=stem, dir=directory)
    path = directory / f"{stem}.s16p"
    if not path.is_file():
        found = sorted(directory.glob(f"{stem}*"))
        raise AssertionError(f"没有写出 {path}，目录里是 {found}")
    return path


def make_config(root: Path, out_dir: Path, **overrides) -> plot.Config:
    data = dict(
        root=root,
        out_dir=out_dir,
        freq_min_ghz=None,
        freq_max_ghz=2.5,
        apply_antenna_map=True,
        map_preset="cad",
        plot="all",
        reflection_ports="all",
        transmission=plot.TransmissionSpec(kind="all"),
        metrics=("mag", "phase"),
        dpi=80,
        skip_existing=True,
        suffixes=(".s16p",),
        label_prefix=None,
    )
    data.update(overrides)
    return plot.Config(**data)


class ParseTests(unittest.TestCase):
    def test_reflection_forms(self):
        self.assertEqual(plot.parse_reflection_ports("all"), "all")
        self.assertEqual(plot.parse_reflection_ports("全部"), "all")
        self.assertEqual(plot.parse_reflection_ports("1-3,5"), (1, 2, 3, 5))
        self.assertEqual(plot.parse_reflection_ports([8, "10-12", 8]), (8, 10, 11, 12))

    def test_reflection_range_must_increase(self):
        with self.assertRaises(plot.ConfigError):
            plot.parse_reflection_ports("8-1")

    def test_transmission_pairs(self):
        spec = plot.parse_transmission(["S2,1", "16,2", [2, 1]])
        self.assertEqual(spec.kind, "select")
        self.assertEqual(spec.pairs, ((2, 1), (16, 2)))
        one = plot.parse_transmission([2, 1])
        self.assertEqual(one.pairs, ((2, 1),))
        text = plot.parse_transmission("S02,01")
        self.assertEqual(text.pairs, ((2, 1),))

    def test_transmission_rejects_diagonal(self):
        with self.assertRaises(plot.ConfigError):
            plot.parse_transmission([[1, 1]])

    def test_transmission_tx_filter_keeps_explicit_pair_first(self):
        spec = plot.parse_transmission({"pairs": [[5, 6]], "tx": [1]})
        pairs = plot.resolve_transmission_pairs(spec, 16, None)
        self.assertEqual(pairs[0], (5, 6))
        self.assertEqual(pairs[1], (2, 1))
        self.assertIn((16, 1), pairs)
        self.assertNotIn((1, 1), pairs)
        self.assertEqual(len(pairs), 16)

    def test_rx_filter(self):
        spec = plot.parse_transmission({"rx": [2]})
        pairs = plot.resolve_transmission_pairs(spec, 16, None)
        self.assertEqual(pairs[0], (2, 1))
        self.assertEqual(pairs[1], (2, 3))
        self.assertEqual(len(pairs), 15)

    def test_yaml_config_words_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "root": tmp,
                        "out_dir": tmp,
                        "plot": "透射",
                        "reflection_ports": "1-2",
                        "transmission_pairs": {"tx": [1], "rx": [2, 3]},
                        "freq_min_ghz": None,
                        "freq_max_ghz": 2.5,
                        "metrics": ["phase", "mag"],
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            cfg = plot.load_config(path)
        self.assertEqual(cfg.plot, "transmission")
        self.assertEqual(cfg.reflection_ports, (1, 2))
        self.assertEqual(cfg.metrics, ("phase", "mag"))
        pairs = plot.resolve_transmission_pairs(cfg.transmission, 16, None)
        self.assertEqual(pairs, [(2, 1), (3, 1)])

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yaml"
            bad.write_text("root: a\nout_dir: b\nplot: sideways\n", encoding="utf-8")
            with self.assertRaises(plot.ConfigError):
                plot.load_config(bad)
            bad.write_text("root: a\nout_dir: b\nfreq_min_ghz: 3\nfreq_max_ghz: 1\n", encoding="utf-8")
            with self.assertRaises(plot.ConfigError):
                plot.load_config(bad)
            bad.write_text("root: a\nout_dir: b\nnot_a_key: 1\n", encoding="utf-8")
            with self.assertRaises(plot.ConfigError):
                plot.load_config(bad)

    def test_shipped_config_selects_everything(self):
        cfg = plot.load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        self.assertEqual(cfg.plot, "all")
        self.assertEqual(cfg.reflection_ports, "all")
        self.assertEqual(cfg.transmission.kind, "all")
        self.assertEqual(cfg.freq_max_ghz, 2.5)
        self.assertIsNone(cfg.freq_min_ghz)
        self.assertEqual(cfg.map_preset, "cad")
        self.assertEqual(cfg.metrics, ("mag", "phase"))
        self.assertTrue(cfg.skip_existing)
        self.assertTrue(cfg.apply_antenna_map)
        self.assertIn("钱鑫源", str(cfg.root))


class TaskRuleTests(unittest.TestCase):
    def test_reflection_all_is_one_figure_and_transmission_is_one_each(self):
        ant = np.arange(16)
        tasks = plot.build_tasks("all", "all", plot.TransmissionSpec(kind="all"), 16, ant)
        self.assertEqual(tasks[0].kind, "reflection_overlay")
        self.assertEqual(len(tasks[0].pairs), 16)
        self.assertEqual(tasks[0].pairs[0], (1, 1))
        self.assertEqual(tasks[0].pairs[-1], (16, 16))
        singles = [task for task in tasks if task.kind == "transmission"]
        self.assertEqual(len(singles), 240)
        self.assertTrue(all(len(task.pairs) == 1 for task in singles))
        self.assertEqual(singles[0].pairs[0], (2, 1))
        self.assertEqual(singles[-1].pairs[0], (15, 16))
        paths = [plot.figure_path(Path("out"), task) for task in tasks]
        self.assertEqual(len(set(paths)), 241)
        self.assertEqual(paths[0], Path("out/reflection_all.png"))
        self.assertEqual(
            paths[1],
            Path("out/transmission/Tx_Port_01_L/S02_01_LL.png"),
        )

    def test_listed_reflection_ports_are_separate_figures(self):
        tasks = plot.build_tasks(
            "reflection",
            (1, 5, 11),
            plot.TransmissionSpec(kind="all"),
            16,
            np.arange(16),
        )
        self.assertEqual([task.kind for task in tasks], ["reflection", "reflection", "reflection"])
        self.assertEqual(
            [plot.figure_path(Path("out"), task).name for task in tasks],
            ["S01_01_L.png", "S05_05_R.png", "S11_11_L.png"],
        )

    def test_ring_order_only_changes_stack_order(self):
        ant = plot.ant_from_ts_0based("cad_ring", 16)
        tasks = plot.build_tasks("reflection", "all", plot.TransmissionSpec(kind="select"), 16, ant)
        silks = [rx for rx, _tx in tasks[0].pairs]
        self.assertEqual(silks, list(plot.RING_CW_PORTS))
        pairs = plot.resolve_transmission_pairs(plot.TransmissionSpec(kind="all"), 16, ant)
        self.assertEqual(len(pairs), 240)
        self.assertEqual(pairs[0], (1, 2))

    def test_port_out_of_range(self):
        with self.assertRaises(plot.ConfigError):
            plot.build_tasks("reflection", (1, 20), plot.TransmissionSpec(kind="select"), 16, None)


class CurveValueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sfile = write_s16p(self.root, "sample")

    def tearDown(self):
        plt.close("all")
        self.tmp.cleanup()

    def test_frequency_crop_and_silk_value_survives_ring_map(self):
        freq, s, ant = plot.load_touchstone(self.sfile, 1.0, 2.5, True, "cad")
        self.assertTrue(np.allclose(freq, [1.0, 1.5, 2.0, 2.5]))
        mag, _, _ = plot.metric_array(plot.s_element(s, 2, 2, ant), "mag")
        phase, _, _ = plot.metric_array(plot.s_element(s, 2, 2, ant), "phase")
        self.assertTrue(np.allclose(mag, -11))
        self.assertTrue(np.allclose(phase, 0.02))

        freq_r, s_r, ant_r = plot.load_touchstone(self.sfile, None, 2.5, True, "cad_ring")
        self.assertEqual(len(freq_r), 5)
        mag_r, _, _ = plot.metric_array(plot.s_element(s_r, 2, 2, ant_r), "mag")
        trans, _, _ = plot.metric_array(plot.s_element(s_r, 2, 1, ant_r), "mag")
        self.assertTrue(np.allclose(mag_r, -11))
        self.assertTrue(np.allclose(trans, -31))
        self.assertEqual(int(ant_r[0] + 1), 2)

    def test_overlay_has_sixteen_curves_and_single_has_one(self):
        freq, s, ant = plot.load_touchstone(self.sfile, None, 2.5, True, "cad")
        overlay = plot.build_tasks("reflection", "all", plot.TransmissionSpec(kind="select"), 16, ant)[0]
        fig = plot.draw_task(freq, s, overlay, ant, ("mag", "phase"), "sample.s16p", "Ant", None, 2.5)
        try:
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(len(fig.axes[0].lines), 16)
            self.assertEqual(len(fig.axes[1].lines), 16)
            self.assertTrue(np.allclose(fig.axes[0].lines[0].get_ydata(), -10))
            self.assertTrue(np.allclose(fig.axes[0].lines[15].get_ydata(), -25))
            self.assertTrue(np.allclose(fig.axes[1].lines[1].get_ydata(), 0.02))
            labels = [text.get_text() for text in fig.legends[0].get_texts()]
            self.assertEqual(len(labels), 16)
            self.assertIn("S1,1", labels[0])
            self.assertIn("[L]", labels[0])
        finally:
            plt.close(fig)

        single = plot.PlotTask("transmission", ((2, 1),))
        fig = plot.draw_task(freq, s, single, ant, ("mag", "phase"), "sample.s16p", "Ant", None, 2.5)
        try:
            self.assertEqual(len(fig.axes[0].lines), 1)
            self.assertTrue(np.allclose(fig.axes[0].lines[0].get_ydata(), -31))
            self.assertTrue(np.allclose(fig.axes[1].lines[0].get_ydata(), 0.2))
        finally:
            plt.close(fig)


class BatchTests(unittest.TestCase):
    def test_batch_layout_skip_and_single_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "数据"
            out = Path(tmp) / "结果"
            write_s16p(root / "11.6", "MW_a")
            write_s16p(root / "11.6", "MW_b")
            touched = root / "11.6" / "note.S16P"
            touched.write_text("not a touchstone", encoding="utf-8")
            cfg = make_config(
                root,
                out,
                plot="all",
                reflection_ports="all",
                transmission=plot.parse_transmission([[2, 1], [3, 1]]),
                freq_max_ghz=2.5,
            )
            found_root, found = plot.collect_files(root, (".s16p",))
            self.assertEqual(found_root, root.resolve())
            self.assertEqual(len(found), 3)

            stats = plot.run(cfg)
            self.assertEqual(stats.fail, 1)
            self.assertEqual(stats.ok, 2 * 3)
            overlay = out / "11.6" / "MW_a" / "reflection_all.png"
            trans = out / "11.6" / "MW_a" / "transmission" / "Tx_Port_01_L" / "S02_01_LL.png"
            other = out / "11.6" / "MW_a" / "transmission" / "Tx_Port_01_L" / "S03_01_RL.png"
            self.assertTrue(overlay.is_file() and overlay.stat().st_size > 1000)
            self.assertTrue(trans.is_file())
            self.assertTrue(other.is_file())
            self.assertEqual(overlay.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertFalse((out / "11.6" / "MW_a" / "reflection").exists())

            again = plot.run(cfg)
            self.assertEqual(again.ok, 0)
            self.assertEqual(again.skip, 6)
            self.assertEqual(again.fail, 1)

            one = write_s16p(Path(tmp) / "only", "solo")
            one_out = Path(tmp) / "only_out"
            stats = plot.run(
                make_config(
                    one,
                    one_out,
                    plot="reflection",
                    reflection_ports=(1, 5),
                    transmission=plot.TransmissionSpec(kind="all"),
                )
            )
            self.assertEqual(stats.ok, 2)
            self.assertEqual(stats.fail, 0)
            self.assertTrue((one_out / "solo" / "reflection" / "S01_01_L.png").is_file())
            self.assertTrue((one_out / "solo" / "reflection" / "S05_05_R.png").is_file())
            self.assertFalse((one_out / "solo" / "reflection_all.png").exists())
            self.assertFalse((one_out / "solo" / "transmission").exists())

    def test_main_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            missing = Path(tmp) / "missing"
            path.write_text(
                f"root: {missing}\nout_dir: {Path(tmp) / 'out'}\nplot: reflection\n",
                encoding="utf-8",
            )
            self.assertEqual(plot.main(["-c", str(path)]), 1)


if __name__ == "__main__":
    unittest.main()
