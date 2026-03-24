import shutil
import unittest
from pathlib import Path

from deeplense_agent.models import ModelConfiguration, SimulationRequest, SubstructureType
from deeplense_agent.simulator import DeepLenseSimulationService


class DeepLenseAgentServiceTests(unittest.TestCase):
    workspace_root = Path('artifacts/test_suite')

    def setUp(self):
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def test_preview_uses_service_output_root_and_defaults(self):
        service = DeepLenseSimulationService(output_root='artifacts/test_service_root')
        request = SimulationRequest(
            model_config=ModelConfiguration.MODEL_I,
            substructure_type=SubstructureType.NO_SUB,
        )

        plan = service.preview(request)

        self.assertEqual(Path(plan.resolved_request.output_root), Path('artifacts/test_service_root'))
        self.assertEqual(plan.resolved_request.configuration, ModelConfiguration.MODEL_I)
        self.assertEqual(plan.resolved_request.resolution, 150)
        self.assertIn('resolution', plan.resolved_request.defaulted_fields)

    def test_request_rejects_invalid_redshift_order(self):
        with self.assertRaises(ValueError):
            SimulationRequest(
                model_config=ModelConfiguration.MODEL_II,
                substructure_type=SubstructureType.CDM,
                lens_redshift=1.0,
                source_redshift=0.5,
            )

    def test_run_generates_artifacts_for_model_i_and_model_ii(self):
        cases = [
            SimulationRequest(
                model_config=ModelConfiguration.MODEL_I,
                substructure_type=SubstructureType.NO_SUB,
                image_count=1,
                resolution=16,
                seed=3,
            ),
            SimulationRequest(
                model_config=ModelConfiguration.MODEL_II,
                substructure_type=SubstructureType.CDM,
                image_count=1,
                resolution=16,
                cdm_subhalo_mean=5,
                seed=5,
            ),
        ]

        for request in cases:
            output_root = self.workspace_root / request.configuration.value.lower()
            if output_root.exists():
                shutil.rmtree(output_root)
            service = DeepLenseSimulationService(output_root=output_root)

            with self.subTest(model=request.configuration.value):
                result = service.run(request)
                self.assertTrue(result.output_dir.exists())
                self.assertTrue(result.metadata_path.exists())
                self.assertEqual(len(result.artifacts), 1)
                self.assertTrue(result.artifacts[0].npy_path.exists())
                self.assertTrue(result.artifacts[0].png_path.exists())
                self.assertEqual(result.artifacts[0].shape, (16, 16))
                self.assertIsNotNone(result.contact_sheet_path)
                self.assertTrue(result.contact_sheet_path.exists())


if __name__ == '__main__':
    unittest.main()
