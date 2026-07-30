from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from atom_phase_law_experiment import (
    PhaseConfig,
    build_tiny_world,
    config_with,
    evaluate_rows,
    model_payload,
    train_phase_model,
)
from atom_phase_side_view import (
    ATOM_ARTIFACT_BINDING,
    ATOM_SIDE_VIEW_RUNTIME,
    render_phase_artifact,
)
from atom_runtime_knowledge import (
    ATOM_RAG_RUNTIME,
    ATOM_WIKI_GRAPH_RUNTIME,
    AtomWikiGraph,
    retrieve_atom_context,
)


class AtomPhaseRuntimeIntegrationTests(unittest.TestCase):
    def test_runtime_wires_graph_rag_and_real_artifact_side_view(self) -> None:
        graph = AtomWikiGraph()
        graph.assert_all_leaves_are_universe_primitives()
        retrieved = retrieve_atom_context(
            graph, "phase interference thermal cooling abstraction", limit=8
        )
        names = {row["name"] for row in retrieved}
        self.assertIn("phase_mix", names)
        self.assertIn("thermal_anneal", names)
        self.assertEqual(ATOM_WIKI_GRAPH_RUNTIME, "atom-wiki-graph-v1")
        self.assertEqual(ATOM_RAG_RUNTIME, "atom-graph-rag-v1")

        program = build_tiny_world()
        config = config_with(PhaseConfig(), epochs=3, crystallization_coherence=0.0)
        runtime, history = train_phase_model(program, config=config)
        evaluation = {
            "heldout_single_step": evaluate_rows(
                runtime, program["heldout_single_step"], "integration-heldout"
            ),
            "unseen_two_step": evaluate_rows(
                runtime, program["unseen_two_step"], "integration-two"
            ),
            "unseen_three_step": evaluate_rows(
                runtime, program["unseen_three_step"], "integration-three"
            ),
        }
        runtime.consolidate("integration-abstract")
        model = model_payload(runtime, program)
        report = {
            "experiment": "atom_phase_integration",
            "model_hash": model["model_hash"],
            "training": {
                "final_energy": history[-1]["energy"],
                "raw_traces_after_abstraction": len(runtime.state.traces),
            },
            "evaluation": evaluation,
            "controlled_chaos": {
                "initial_temperature": config.initial_temperature,
                "final_temperature": runtime.state.temperature,
                "cumulative_phase_energy": runtime.state.cumulative_phase_energy,
            },
            "experiment_gates": {
                "gates": {
                    "knowledge_runtime": bool(retrieved),
                    "artifact_bound": True,
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "atom_phase_law_side_view.html"
            rendered = render_phase_artifact(report, model, output)
            document = rendered.read_text(encoding="utf-8")
            self.assertIn(model["model_hash"], document)
            self.assertIn(ATOM_SIDE_VIEW_RUNTIME, document)
            self.assertIn(ATOM_ARTIFACT_BINDING, document)
            self.assertIn("Learned phase lattice", document)


if __name__ == "__main__":
    unittest.main()
