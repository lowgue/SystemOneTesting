"""Regression tests for the legacy model runner and adapter integrations."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from models.common import motor
from vision_bench.adapters.d1_omni import D1Omni
from vision_bench.adapters.http import SystemOneHTTP
from vision_bench.core import DEFAULT_CONTRACT, load_json
from vision_bench.protocols import PROFILES


class LegacyModelRuntimeTests(unittest.TestCase):
    """Cover regressions found in the shared legacy model runner."""

    def test_bundle_score_compares_all_contract_fields(self) -> None:
        """Verify a complete bundle counts as correct when every field matches."""
        expected = {
            "presence": "yes",
            "make": "audi",
            "color": "white",
            "body": "coupe",
        }
        adapter = Mock()
        adapter.predict.return_value = (expected, {"simulated": True})
        adapter.close.return_value = None

        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_root = Path(temporary_directory)
            image_path = temp_root / "vehicle.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            output_path = temp_root / "result.json"
            item = {
                "id": "bundle-fixture",
                "rel_path": str(image_path),
                "expected": expected,
            }
            config = {
                "nome": "Bundle fixture",
                "campo_alvo": "bundle_completo",
                "dominio": "vehicle",
                "dataset": "unused-by-patched-loader",
            }

            with patch.object(motor, "carregar_itens_dataset", return_value=[item]):
                with patch.object(motor, "create_adapter", return_value=adapter):
                    result = motor.executar_teste_dataset(
                        "blip",
                        config,
                        args_custom=["--output", str(output_path)],
                    )

        self.assertEqual(result["acertos"], 1)
        self.assertEqual(result["acuracia"], 1.0)
        self.assertEqual(result["itens"][0]["esperado"], expected)
        self.assertEqual(result["itens"][0]["predito"], expected)

    def test_prompt_profiles_reject_profiles_ignored_by_the_adapter(self) -> None:
        """Verify unsupported prompt profiles fail instead of producing a false comparison."""
        config = {
            "nome": "Invalid profile fixture",
            "formatos": ["short_open"],
        }

        with self.assertRaisesRegex(ValueError, "Perfis de prompt não suportados"):
            motor.executar_teste_prompt("clef", config, args_custom=[])

    def test_prompt_errors_are_counted_and_later_images_continue(self) -> None:
        """Verify a failed image does not abort the remaining prompt evaluation."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_root = Path(temporary_directory)
            first_image = temp_root / "first.png"
            second_image = temp_root / "second.png"
            Image.new("RGB", (8, 8), "white").save(first_image)
            Image.new("RGB", (8, 8), "black").save(second_image)
            items = [
                {"rel_path": str(first_image), "expected": {"color": "white"}},
                {"rel_path": str(second_image), "expected": {"color": "white"}},
            ]
            adapter = Mock()
            adapter.predict.side_effect = [
                ({"color": "white"}, {}),
                RuntimeError("temporary prediction failure"),
            ]
            adapter.close.return_value = None

            config = {
                "nome": "Prompt error fixture",
                "campo_alvo": "color",
                "dominio": "vehicle",
                "pergunta_base": "What color is visible?",
                "formatos": ["native_short"],
                "criterios": {"white": "White", "black": "Black"},
                "dataset": "unused-by-patched-loader",
            }

            with patch.object(motor, "carregar_itens_dataset", return_value=items):
                with patch.object(motor, "create_adapter", return_value=adapter):
                    result = motor.executar_teste_prompt(
                        "laya", config, args_custom=[]
                    )

        self.assertEqual(result["native_short"]["erros"], 1)
        self.assertEqual(result["native_short"]["acuracia"], 0.5)
        adapter.close.assert_called_once_with()

    def test_http_native_descriptive_profile_changes_question_choices(self) -> None:
        """Verify HTTP requests use the selected native prompt profile."""
        contract = load_json(DEFAULT_CONTRACT)
        spec = {
            "id": "http-profile",
            "adapter": "systemone_http",
            "url": "http://localhost/v1/systemone",
            "profile": "native_descriptive",
        }
        response = BytesIO(
            json.dumps(
                {"success": True, "result": {"answers": {"color": {"choice": "white"}}}}
            ).encode("utf-8")
        )
        adapter = SystemOneHTTP(spec, contract)

        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            adapter.predict(Image.new("RGB", (8, 8), "white"), "vehicle")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            payload["questions"]["color"]["criteria"]["white"],
            "White or off-white bright tone",
        )

    def test_d1_predict_loads_protocols_from_parent_package(self) -> None:
        """Verify D1 prediction resolves the shared protocol module."""
        adapter = object.__new__(D1Omni)
        adapter.spec = {"profile": "native_short"}
        adapter.contract = {
            "domains": {
                "vehicle": {
                    "questions": {
                        "color": {
                            "type": "choice",
                            "instructions": "Choose the color.",
                            "criteria": {"white": "White"},
                        }
                    }
                }
            }
        }
        adapter.device = "cpu"
        adapter.model = Mock()
        adapter.model.system_one.return_value = {
            "answers": {"color": {"choice": "white"}}
        }

        answers, _ = adapter.predict(Image.new("RGB", (8, 8), "white"), "vehicle")

        self.assertEqual(answers, {"color": "white"})

    def test_prompt_profiles_are_declared_for_typed_adapters(self) -> None:
        """Verify HTTP and D1 expose only the profiles their protocols implement."""
        self.assertEqual(
            {profile["id"] for profile in PROFILES["systemone_http"]},
            {"native_short", "native_descriptive"},
        )
        self.assertEqual(
            {profile["id"] for profile in PROFILES["d1_omni"]},
            {"native_short", "native_descriptive"},
        )
