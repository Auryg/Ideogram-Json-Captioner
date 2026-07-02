import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from ideogram_captioner.exif_caption import (
    try_import_caption_from_exif,
    try_import_prompt_text_from_exif,
    workflow_text_candidates,
)
from ideogram_captioner.schema import normalize_caption
from ideogram_captioner.store import CaptionStore


CAPTION_WITH_OBJ_BBOX = {
    "high_level_description": "A red box on white",
    "style_description": {
        "aesthetics": "minimal",
        "lighting": "flat",
        "photo": "studio",
        "medium": "photograph",
    },
    "compositional_deconstruction": {
        "background": "white",
        "elements": [{"type": "obj", "bbox": [100, 200, 300, 400], "desc": "red box"}],
    },
}

CAPTION_TEXT_ONLY_BBOX = {
    "high_level_description": "Sign",
    "compositional_deconstruction": {
        "background": "wall",
        "elements": [{"type": "text", "bbox": [10, 20, 30, 40], "text": "SALE", "desc": "letters"}],
    },
}


def _save_png_with_comfy_prompt(path: Path, node_texts: list[str], *, class_type: str = "CLIPTextEncode") -> None:
    prompt: dict[str, dict] = {}
    for index, text in enumerate(node_texts, start=1):
        prompt[str(index)] = {
            "class_type": class_type,
            "inputs": {"text": text},
            "_meta": {"title": f"Prompt {index}"},
        }

    image = Image.new("RGB", (8, 8), color="white")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", json.dumps(prompt))
    image.save(path, pnginfo=metadata)


def _save_png_with_caption_metadata(path: Path, caption: dict) -> None:
    image = Image.new("RGB", (8, 8), color="white")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("caption", json.dumps(caption))
    image.save(path, pnginfo=metadata)


def _save_png_with_parameters_metadata(path: Path, parameters: str) -> None:
    image = Image.new("RGB", (8, 8), color="white")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("parameters", parameters)
    image.save(path, pnginfo=metadata)


def _save_png_with_comfy_workflow(path: Path, node_texts: list[str]) -> None:
    workflow = {
        "nodes": [
            {
                "id": index,
                "type": "CLIPTextEncode",
                "title": f"Prompt {index}",
                "widgets_values": [text],
            }
            for index, text in enumerate(node_texts, start=1)
        ]
    }
    image = Image.new("RGB", (8, 8), color="white")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("workflow", json.dumps(workflow))
    image.save(path, pnginfo=metadata)


def _save_png_with_ideogram_prompt_builder(path: Path) -> None:
    prompt = {
        "211": {
            "class_type": "Ideogram4PromptBuilderKJ",
            "inputs": {
                "high_level_description": "A photograph of Tara Glenz, standing outside.",
                "background": "outside, trees in the background",
                "style": "photo",
                "style.photo": "85mm",
                "aesthetics": "",
                "lighting": "diffused daylight",
                "medium": "photograph",
                "elements_data": json.dumps(
                    [
                        {
                            "x": 0.17153097989241708,
                            "y": 0.021125327888722192,
                            "w": 0.6658940219757437,
                            "h": 0.9788746721112778,
                            "type": "obj",
                            "text": "",
                            "desc": "Tara Glenz, looking at the camera.",
                            "palette": [],
                        }
                    ]
                ),
            },
            "_meta": {"title": "Ideogram 4 Prompt Builder KJ"},
        }
    }
    image = Image.new("RGB", (8, 8), color="white")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", json.dumps(prompt))
    image.save(path, pnginfo=metadata)


class ExifCaptionTests(unittest.TestCase):
    def test_workflow_text_candidates_sort_by_length_desc(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_comfy_prompt(image_path, ["short", "x" * 30, "medium length text here"])

            candidates = workflow_text_candidates(image_path, limit=5)

            self.assertEqual(candidates[0], "x" * 30)
            self.assertEqual(len(candidates), 3)

    def test_imports_valid_obj_bbox_json_from_workflow_text(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            valid_json = json.dumps(CAPTION_WITH_OBJ_BBOX, separators=(",", ":"))
            invalid_json = '{"high_level_description":"no bbox"}'
            _save_png_with_comfy_prompt(image_path, [invalid_json, valid_json])

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")
            self.assertEqual(
                normalize_caption(caption)["compositional_deconstruction"]["elements"][0]["bbox"],
                [100, 200, 300, 400],
            )

    def test_imports_valid_obj_bbox_json_from_ui_workflow_text(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_comfy_workflow(image_path, [json.dumps(CAPTION_WITH_OBJ_BBOX)])

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")

    def test_imports_json_wrapped_in_markdown_fence(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            fenced = "```json\n" + json.dumps(CAPTION_WITH_OBJ_BBOX, indent=2) + "\n```"
            _save_png_with_comfy_prompt(image_path, [fenced])

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")

    def test_imports_direct_caption_json_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_caption_metadata(image_path, CAPTION_WITH_OBJ_BBOX)

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")

    def test_imports_jpeg_exif_image_description(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.jpg"
            image = Image.new("RGB", (8, 8), color="white")
            exif = Image.Exif()
            exif[270] = json.dumps(CAPTION_TEXT_ONLY_BBOX)
            image.save(image_path, exif=exif)

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")
            self.assertEqual(caption["high_level_description"], "Sign")

    def test_imports_caption_without_obj_bbox(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_comfy_prompt(image_path, [json.dumps(CAPTION_TEXT_ONLY_BBOX)])

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")
            self.assertEqual(caption["high_level_description"], "Sign")

    def test_imports_ideogram_prompt_builder_as_structured_caption(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_ideogram_prompt_builder(image_path)

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported Ideogram caption fields", message or "")
            self.assertEqual(caption["high_level_description"], "A photograph of Tara Glenz, standing outside.")
            self.assertEqual(caption["style_description"]["photo"], "85mm")
            self.assertEqual(caption["compositional_deconstruction"]["background"], "outside, trees in the background")
            self.assertEqual(
                caption["compositional_deconstruction"]["elements"][0],
                {
                    "type": "obj",
                    "bbox": [21, 172, 1000, 837],
                    "desc": "Tara Glenz, looking at the camera.",
                },
            )

    def test_imports_longest_workflow_prompt_as_plain_text_when_no_json(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            prompt = (
                "Tara Glenz, wearing the Mighty Morphin Power Ranger Pink Ranger's outfit. "
                "Tara Glenz is inside the Pterodactyl Zord cockpit, and her face is visible."
            )
            _save_png_with_comfy_prompt(image_path, [prompt, "helmet"])

            caption, caption_message = try_import_caption_from_exif(image_path)
            text, text_message = try_import_prompt_text_from_exif(image_path)

            self.assertIsNone(caption)
            self.assertIsNone(caption_message)
            self.assertIn("Imported prompt text", text_message or "")
            self.assertEqual(text, prompt)

    def test_imports_automatic1111_parameters_as_plain_text(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            prompt = "masterpiece, best quality, a woman standing in a neon city"
            parameters = (
                f"{prompt}\n"
                "Negative prompt: blurry, low quality\n"
                "Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1234, Size: 768x1024, "
                "Model hash: abcdef12, Model: realisticVision, Version: v1.10.1"
            )
            _save_png_with_parameters_metadata(image_path, parameters)

            caption, caption_message = try_import_caption_from_exif(image_path)
            text, text_message = try_import_prompt_text_from_exif(image_path)

            self.assertIsNone(caption)
            self.assertIsNone(caption_message)
            self.assertIn("Imported prompt text", text_message or "")
            self.assertEqual(text, prompt)

    def test_imports_automatic1111_multiline_prompt_without_negative_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            prompt = "cinematic portrait of a pilot\ninside a spacecraft cockpit"
            parameters = (
                f"{prompt}\n"
                "Steps: 20, Sampler: Euler a, CFG scale: 6, Seed: 5678, Size: 512x768, Model: dreamshaper"
            )
            _save_png_with_parameters_metadata(image_path, parameters)

            text, text_message = try_import_prompt_text_from_exif(image_path)

            self.assertIn("Imported prompt text", text_message or "")
            self.assertEqual(text, prompt)

    def test_imports_jpeg_user_comment_parameters_as_plain_text(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.jpg"
            prompt = "oil painting of a quiet library with warm lamps"
            parameters = (
                f"{prompt}\n"
                "Negative prompt: noisy\n"
                "Steps: 32, Sampler: UniPC, CFG scale: 5, Seed: 42, Size: 832x1216, Model: artModel"
            )
            image = Image.new("RGB", (8, 8), color="white")
            exif = Image.Exif()
            exif[37510] = b"ASCII\x00\x00\x00" + parameters.encode("ascii")
            image.save(image_path, exif=exif)

            text, text_message = try_import_prompt_text_from_exif(image_path)

            self.assertIn("Imported prompt text", text_message or "")
            self.assertEqual(text, prompt)

    def test_imports_plain_prompt_metadata_field(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            prompt = "a detailed macro photograph of rain on a red flower"
            image = Image.new("RGB", (8, 8), color="white")
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("prompt", prompt)
            image.save(image_path, pnginfo=metadata)

            text, text_message = try_import_prompt_text_from_exif(image_path)

            self.assertIn("Imported prompt text", text_message or "")
            self.assertEqual(text, prompt)

    def test_ignores_workflow_without_caption_json(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_comfy_prompt(image_path, ["short"])

            caption, message = try_import_caption_from_exif(image_path)
            text, text_message = try_import_prompt_text_from_exif(image_path)

            self.assertIsNone(caption)
            self.assertIsNone(message)
            self.assertIsNone(text)
            self.assertIsNone(text_message)


class StoreExifImportTests(unittest.TestCase):
    def test_missing_caption_file_imports_from_image_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image_path = folder / "sample.png"
            _save_png_with_comfy_prompt(image_path, [json.dumps(CAPTION_WITH_OBJ_BBOX)])
            store = CaptionStore(folder, ".json")

            caption, message = store.load_caption(image_path)

            self.assertIn("Imported caption JSON", message or "")
            self.assertEqual(caption["high_level_description"], "A red box on white")
            self.assertFalse(store.caption_path(image_path).exists())

    def test_existing_caption_file_takes_priority_over_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image_path = folder / "sample.png"
            _save_png_with_comfy_prompt(image_path, [json.dumps(CAPTION_WITH_OBJ_BBOX)])
            sidecar = {
                "high_level_description": "From sidecar",
                "compositional_deconstruction": {"background": "", "elements": []},
            }
            store = CaptionStore(folder, ".json")
            store.save_caption(image_path, sidecar)

            caption, message = store.load_caption(image_path)

            self.assertIsNone(message)
            self.assertEqual(caption["high_level_description"], "From sidecar")


if __name__ == "__main__":
    unittest.main()
