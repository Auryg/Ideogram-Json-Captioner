import unittest
import tempfile
import json
from unittest.mock import patch
from pathlib import Path

from ideogram_captioner.llm_captioning import (
    AutoCaptionError,
    CaptioningSettings,
    DEFAULT_YOLOE26_BBOX_MODEL,
    ModelAssets,
    ModelJsonError,
    PLAIN_CAPTION_MODE_PERSON,
    add_bboxes_to_caption,
    append_image_exif_context,
    build_llama_server_command,
    bbox_target_indices,
    bbox_target_indices_with_reasons,
    bbox_xyxy_pixels_to_yxyx,
    bbox_xyxy_to_yxyx,
    chat_text,
    chat_vision,
    ensure_model_assets,
    extract_json,
    format_prompt,
    generate_json_from_image,
    generate_json_refinement,
    generate_plain_caption,
    image_exif_context,
    json_system_prompt,
    load_model_profiles,
    load_prompts,
    parse_batch_bboxes,
    parse_batch_bboxes_with_reasons,
    parse_json_with_repair,
    request_user_prompt,
    resolve_server_model_id,
    runtime_config_for_task,
    safe_repo_dir,
    server_host_port,
    server_model_ids,
    should_try_bbox,
    strip_thinking_output,
    yolo_detections_for_image,
    write_default_prompts,
)


class FakeMessage:
    def __init__(self, content, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class FakeChoice:
    def __init__(self, content, finish_reason="stop", reasoning_content=None):
        self.message = FakeMessage(content, reasoning_content=reasoning_content)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content, finish_reason="stop", model="fake-model", usage=None, reasoning_content=None):
        self.choices = [FakeChoice(content, finish_reason, reasoning_content=reasoning_content)]
        self.model = model
        self.usage = usage


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake response left.")
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = self


class FakeYoloBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class FakeYoloResult:
    def __init__(self, boxes, names, orig_shape):
        self.boxes = boxes
        self.names = names
        self.orig_shape = orig_shape


class FakeYoloModel:
    def __init__(self, result):
        self.result = result
        self.requests = []
        self.classes = []
        self.names = result.names

    def set_classes(self, classes):
        self.classes = list(classes)
        self.names = {index: name for index, name in enumerate(self.classes)}
        self.result.names = self.names

    def predict(self, **kwargs):
        self.requests.append(kwargs)
        return [self.result]


class LlmCaptioningTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        parsed = extract_json('```json\n{"high_level_description":"sign"}\n```')
        self.assertEqual(parsed["high_level_description"], "sign")

    def test_repairs_malformed_json_once(self):
        progress_messages = []
        with patch(
            "ideogram_captioner.llm_captioning.chat_text",
            return_value='{"high_level_description":"sign"}',
        ) as chat:
            parsed = parse_json_with_repair(
                CaptioningSettings(caption_model="repair-model"),
                "caption",
                '{"high_level_description":"sign"',
                "a caption object",
                max_tokens=100,
                progress=progress_messages.append,
            )

        self.assertEqual(parsed["high_level_description"], "sign")
        self.assertEqual(chat.call_count, 1)
        self.assertIn("retrying", progress_messages[0])
        self.assertIn("succeeded", progress_messages[-1])

    def test_reports_original_and_repair_json_failures(self):
        with patch("ideogram_captioner.llm_captioning.chat_text", return_value='{"still":"broken"'):
            with self.assertRaises(ModelJsonError) as raised:
                parse_json_with_repair(
                    CaptioningSettings(caption_model="repair-model"),
                    "caption",
                    '{"high_level_description":"sign"',
                    "a caption object",
                    max_tokens=100,
                )

        self.assertIn("repair retry failed", str(raised.exception))
        self.assertIn("high_level_description", raised.exception.raw_output)
        self.assertIn("still", raised.exception.repair_output)

    def test_converts_bbox_coordinates(self):
        self.assertEqual(bbox_xyxy_to_yxyx([200, 100, 400, 300]), [100, 200, 300, 400])
        self.assertIsNone(bbox_xyxy_to_yxyx([200, 100, 200, 300]))

    def test_converts_yolo_pixel_bbox_coordinates(self):
        self.assertEqual(bbox_xyxy_pixels_to_yxyx([20, 10, 180, 90], width=200, height=100), [100, 100, 900, 900])
        self.assertIsNone(bbox_xyxy_pixels_to_yxyx([20, 10, 20, 90], width=200, height=100))

    def test_yoloe26_bbox_backend_uses_element_prompts(self):
        result = FakeYoloResult(
            FakeYoloBoxes(
                xyxy=[[50, 20, 180, 90]],
                conf=[0.81],
                cls=[0],
            ),
            names={},
            orig_shape=(100, 200),
        )
        model = FakeYoloModel(result)
        caption = {
            "high_level_description": "A product label.",
            "style_description": {"aesthetics": "", "lighting": "", "photo": "", "medium": "photograph"},
            "compositional_deconstruction": {
                "background": "",
                "elements": [
                    {"type": "obj", "desc": "A small brass buckle on the black bag."},
                    {"type": "obj", "desc": "A red embroidered sleeve patch."},
                ],
            },
        }

        with patch("ideogram_captioner.llm_captioning._load_yolo_model", return_value=model):
            updated, attempted, added, reasons = add_bboxes_to_caption(
                CaptioningSettings(bbox_backend="yoloe26", yolo_bbox_confidence=0.35),
                Path("sample.jpg"),
                caption,
            )

        elements = updated["compositional_deconstruction"]["elements"]
        self.assertEqual(model.classes, ["A small brass buckle on the black bag.", "A red embroidered sleeve patch."])
        self.assertEqual(elements[0]["bbox"], [200, 250, 900, 900])
        self.assertNotIn("bbox", elements[1])
        self.assertEqual(attempted, 2)
        self.assertEqual(added, 1)
        self.assertEqual(reasons["YOLOE-26 detected no matching prompt"], 1)
        self.assertEqual(model.requests[0]["conf"], 0.35)
        self.assertEqual(model.requests[0]["imgsz"], 1024)

    def test_yoloe26_defaults_to_large_model_and_can_omit_imgsz(self):
        settings = CaptioningSettings(bbox_backend="yoloe26", yolo_bbox_model="yolo26n.pt", yolo_bbox_imgsz=0)
        self.assertEqual(settings.yolo_bbox_model, DEFAULT_YOLOE26_BBOX_MODEL)
        self.assertEqual(settings.yolo_bbox_imgsz, 0)

        result = FakeYoloResult(
            FakeYoloBoxes(xyxy=[], conf=[], cls=[]),
            names={},
            orig_shape=(100, 200),
        )
        model = FakeYoloModel(result)
        with patch("ideogram_captioner.llm_captioning._load_yolo_model", return_value=model):
            yolo_detections_for_image(settings, Path("sample.jpg"))

        self.assertNotIn("imgsz", model.requests[0])

    def test_legacy_yolo26_settings_migrate_to_yoloe26(self):
        settings = CaptioningSettings(bbox_backend="yolo26", yolo_bbox_model="yolo26x.pt")

        self.assertEqual(settings.bbox_backend, "yoloe26")
        self.assertEqual(settings.yolo_bbox_model, DEFAULT_YOLOE26_BBOX_MODEL)

    def test_parses_batch_bbox_response(self):
        parsed = parse_batch_bboxes('{"bboxes":{"0":[10,20,30,40],"1":null}}')
        self.assertEqual(parsed["0"], [20, 10, 40, 30])
        self.assertIsNone(parsed["1"])

    def test_parses_batch_bbox_skip_reasons(self):
        parsed, reasons = parse_batch_bboxes_with_reasons(
            '{"bboxes":{"0":[10,20,30,40],"1":null,"2":[10,20,10,40]}}'
        )

        self.assertEqual(parsed["0"], [20, 10, 40, 30])
        self.assertIsNone(parsed["1"])
        self.assertIsNone(parsed["2"])
        self.assertEqual(reasons["1"], "model returned null")
        self.assertEqual(reasons["2"], "model returned invalid bbox")

    def test_bbox_filter_keeps_concrete_element_with_patterned_clothing(self):
        self.assertTrue(
            should_try_bbox(
                {
                    "type": "obj",
                    "desc": (
                        "A young woman sitting on a white toilet, wearing white panties "
                        "with a small heart pattern and a white headband."
                    ),
                }
            )
        )
        self.assertFalse(should_try_bbox({"type": "obj", "desc": "A repeating background pattern."}))

    def test_bbox_target_filter_is_opt_in(self):
        elements = [
            {"type": "obj", "desc": "A repeating background pattern."},
            {"type": "obj", "desc": "A woman wearing a floral pattern dress."},
            {"type": "obj", "bbox": [1, 2, 3, 4], "desc": "A chair."},
        ]

        self.assertEqual(bbox_target_indices(elements, CaptioningSettings()), [0, 1, 2])
        self.assertEqual(
            bbox_target_indices(elements, CaptioningSettings(filter_bbox_targets=True, overwrite_bboxes=False)),
            [1],
        )

    def test_bbox_target_reasons(self):
        elements = [
            {"type": "obj", "desc": "A repeating background pattern."},
            {"type": "obj", "bbox": [1, 2, 3, 4], "desc": "A chair."},
            {"type": "misc", "desc": "A label."},
        ]

        indices, reasons = bbox_target_indices_with_reasons(
            elements,
            CaptioningSettings(filter_bbox_targets=True, overwrite_bboxes=False),
        )

        self.assertEqual(indices, [])
        self.assertEqual(reasons[0], "filtered as vague/ambient")
        self.assertEqual(reasons[1], "existing bbox kept")
        self.assertEqual(reasons[2], "not an obj/text element")

    def test_legacy_caption_model_for_bboxes_flag_is_ignored(self):
        settings = CaptioningSettings(use_caption_model_for_bboxes=True, caption_model="shared-model")
        self.assertFalse(settings.use_caption_model_for_bboxes)
        self.assertNotEqual(runtime_config_for_task(settings, "bbox").api_model, "shared-model")

    def test_default_profile_is_downloadable_local_model(self):
        config = runtime_config_for_task(CaptioningSettings(), "caption")
        self.assertEqual(config.hf_repo, "unsloth/Qwen2.5-VL-7B-Instruct-GGUF")
        self.assertEqual(config.model_filename, "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf")

    def test_loads_profiles_from_json_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "id": "local-caption",
                                "label": "Local Caption",
                                "tasks": ["caption"],
                                "kind": "local",
                                "api_model": "local-caption",
                                "mmproj_repo": "other/projector-repo",
                                "local_model_path": "C:/models/model.gguf",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profiles = load_model_profiles(path)

        self.assertEqual(profiles["caption"][0].id, "local-caption")
        self.assertEqual(profiles["caption"][0].mmproj_repo, "other/projector-repo")
        self.assertNotEqual(profiles["bbox"][0].id, "local-caption")
        self.assertEqual(profiles["caption"][-2].id, "custom-hf")
        self.assertEqual(profiles["caption"][-1].id, "custom-local")

    def test_migrates_legacy_local_model_server_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "id": "local-model",
                                "label": "Existing server alias: local-model",
                                "tasks": ["caption"],
                                "kind": "server",
                                "api_model": "local-model",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profile = load_model_profiles(path)["caption"][0]

        self.assertEqual(profile.label, "Existing server: discover or enter model ID")
        self.assertEqual(profile.api_model, "")

    def test_loads_partial_prompt_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "prompts"
            folder.mkdir()
            (folder / "bbox_system.txt").write_text("custom bbox system", encoding="utf-8")
            prompts = load_prompts(folder)

        self.assertEqual(prompts["bbox_system"], "custom bbox system")
        self.assertIn("{targets_json}", prompts["bbox_user"])
        self.assertIn("{instructions}", prompts["json_refine_user"])
        self.assertIn("plain_caption_system", prompts)
        self.assertIn("{person_name}", prompts["person_caption_user"])

    def test_writes_default_prompt_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "prompts"
            written = write_default_prompts(folder)

            self.assertTrue((written / "bbox_user.txt").exists())
            self.assertTrue((written / "text_to_json_user.txt").exists())
            self.assertTrue((written / "json_refine_user.txt").exists())
            self.assertTrue((written / "person_caption_user.txt").exists())

    def test_prompt_placeholder_errors_are_actionable(self):
        with self.assertRaises(AutoCaptionError):
            format_prompt("{missing}", present="x")

    def test_person_plain_caption_uses_name_prompt_and_single_line_output(self):
        settings = CaptioningSettings(
            caption_model="vision-model",
            plain_caption_mode=PLAIN_CAPTION_MODE_PERSON,
            person_caption_name="Alex",
        )
        with patch("ideogram_captioner.llm_captioning.chat_vision", return_value='"Alex is seated indoors.\nA lamp is nearby."') as chat:
            caption = generate_plain_caption(settings, Path("sample.png"))

        self.assertEqual(caption, "Alex is seated indoors. A lamp is nearby.")
        request = chat.call_args.kwargs
        self.assertIn("known people", request["system"])
        self.assertIn("Provided person name: Alex", request["user"])

    def test_person_plain_caption_requires_name(self):
        settings = CaptioningSettings(caption_model="vision-model", plain_caption_mode=PLAIN_CAPTION_MODE_PERSON)

        with self.assertRaises(AutoCaptionError):
            generate_plain_caption(settings, Path("sample.png"))

    def test_json_directive_is_system_side(self):
        prompts = load_prompts(Path("__missing_prompts_folder__"))
        settings = CaptioningSettings(creative_json=True)
        system = json_system_prompt(prompts, "text_to_json_system", settings)

        self.assertIn("Expansion policy", system)
        self.assertNotIn("{directive}", prompts["text_to_json_user"])
        self.assertNotIn("{directive}", prompts["image_to_json_user"])

    def test_json_refinement_requires_instructions(self):
        with self.assertRaises(AutoCaptionError):
            generate_json_refinement(
                CaptioningSettings(caption_model="vision-model"),
                Path("sample.png"),
                {"high_level_description": "A sign"},
                "",
                "",
            )

    def test_json_refinement_uses_image_context_and_preserves_missing_bboxes(self):
        raw = json.dumps(
            {
                "high_level_description": "A woman seated beside a window.",
                "style_description": {
                    "aesthetics": "natural",
                    "lighting": "window light",
                    "photo": "portrait lens",
                    "medium": "photograph",
                },
                "compositional_deconstruction": {
                    "background": "room",
                    "elements": [{"type": "obj", "desc": "A woman wearing a red jacket, seated in profile."}],
                },
            }
        )
        caption = {
            "high_level_description": "A woman.",
            "style_description": {
                "aesthetics": "natural",
                "lighting": "soft",
                "photo": "",
                "medium": "photograph",
            },
            "compositional_deconstruction": {
                "background": "room",
                "elements": [{"type": "obj", "bbox": [100, 200, 500, 700], "desc": "A woman."}],
            },
        }

        with patch("ideogram_captioner.llm_captioning.chat_vision", return_value=raw) as chat, patch(
            "ideogram_captioner.llm_captioning.image_exif_context",
            return_value={"camera_model": "EOS R5"},
        ):
            refined = generate_json_refinement(
                CaptioningSettings(caption_model="vision-model"),
                Path("sample.png"),
                caption,
                "text caption source",
                "Add clothing and pose details to people.",
            )

        self.assertEqual(
            refined["compositional_deconstruction"]["elements"][0]["bbox"],
            [100, 200, 500, 700],
        )
        request = chat.call_args.kwargs["user"]
        self.assertIn("Add clothing and pose details", request)
        self.assertIn("text caption source", request)
        self.assertIn('"high_level_description": "A woman."', request)
        self.assertIn('"camera_model":"EOS R5"', request)

    def test_extracts_supported_image_exif_context(self):
        from PIL import ExifTags, Image

        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.jpg"
            exif = Image.Exif()
            exif[ExifTags.Base.Make] = "Canon"
            exif[ExifTags.Base.Model] = "EOS R5"
            exif[ExifTags.Base.LensModel] = "RF 50mm F1.2"
            exif[ExifTags.Base.FNumber] = (28, 10)
            exif[ExifTags.Base.ExposureTime] = (1, 125)
            exif[ExifTags.Base.FocalLength] = (50, 1)
            exif[ExifTags.Base.ISOSpeedRatings] = 400
            Image.new("RGB", (8, 8), "white").save(image_path, exif=exif)

            context = image_exif_context(image_path)

        self.assertEqual(context["camera_make"], "Canon")
        self.assertEqual(context["camera_model"], "EOS R5")
        self.assertEqual(context["lens_model"], "RF 50mm F1.2")
        self.assertEqual(context["f_stop"], "f/2.8")
        self.assertEqual(context["exposure_time"], "1/125s")
        self.assertEqual(context["focal_length"], "50mm")
        self.assertEqual(context["iso"], "400")

    def test_append_image_exif_context_is_noop_without_supported_exif(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            image_path.write_bytes(b"not an image")

            self.assertEqual(append_image_exif_context("Describe.", image_path), "Describe.")

    def test_json_from_image_sends_exif_metadata_when_available(self):
        raw = json.dumps(
            {
                "high_level_description": "A studio portrait.",
                "style_description": {
                    "aesthetics": "clean",
                    "lighting": "soft",
                    "photo": "shallow depth of field",
                    "medium": "photograph",
                },
                "compositional_deconstruction": {"background": "studio", "elements": []},
            }
        )

        with patch("ideogram_captioner.llm_captioning.chat_vision", return_value=raw) as chat, patch(
            "ideogram_captioner.llm_captioning.image_exif_context",
            return_value={"f_stop": "f/2.8", "focal_length": "50mm", "iso": "400", "exposure_time": "1/125s"},
        ):
            generate_json_from_image(CaptioningSettings(caption_model="vision-model"), Path("sample.jpg"))

        request = chat.call_args.kwargs["user"]
        self.assertIn("Image EXIF metadata", request)
        self.assertIn('"f_stop":"f/2.8"', request)
        self.assertIn('"focal_length":"50mm"', request)
        self.assertIn('"iso":"400"', request)
        self.assertIn('"exposure_time":"1/125s"', request)

    def test_custom_local_profile_uses_selected_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model = folder / "model.gguf"
            mmproj = folder / "mmproj.gguf"
            model.write_text("x", encoding="utf-8")
            mmproj.write_text("x", encoding="utf-8")
            settings = CaptioningSettings(
                caption_profile_id="custom-local",
                caption_model="local-caption",
                caption_local_model_path=str(model),
                caption_local_mmproj_path=str(mmproj),
            )
            config = runtime_config_for_task(settings, "caption")
            assets = ensure_model_assets(settings, "caption")

        self.assertEqual(config.kind, "local")
        self.assertEqual(assets.model_path, model)
        self.assertEqual(assets.mmproj_path, mmproj)

    def test_parses_server_host_port(self):
        self.assertEqual(server_host_port("http://127.0.0.1:8000/v1"), ("127.0.0.1", 8000))
        self.assertEqual(server_host_port("https://example.test/v1"), ("example.test", 443))

    def test_parses_server_model_ids(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"data":[{"id":"qwen3vl"},{"id":"local-model"}]}'

        with patch("urllib.request.urlopen", return_value=Response()):
            self.assertEqual(server_model_ids("http://127.0.0.1:8000/v1"), {"qwen3vl", "local-model"})

    def test_resolve_server_model_id_keeps_exact_match(self):
        self.assertEqual(
            resolve_server_model_id("model-b", {"model-a", "model-b"}, auto_select_single=True),
            "model-b",
        )

    def test_resolve_server_model_id_auto_selects_only_model(self):
        self.assertEqual(
            resolve_server_model_id("local-model", {"lm-studio/qwen-vision"}, auto_select_single=True),
            "lm-studio/qwen-vision",
        )
        self.assertEqual(
            resolve_server_model_id("", {"lm-studio/qwen-vision"}, auto_select_single=True),
            "lm-studio/qwen-vision",
        )

    def test_resolve_server_model_id_requires_choice_for_multiple_models(self):
        with self.assertRaisesRegex(AutoCaptionError, "model-a, model-b"):
            resolve_server_model_id("local-model", {"model-b", "model-a"}, auto_select_single=True)
        with self.assertRaisesRegex(AutoCaptionError, "No API model ID is selected"):
            resolve_server_model_id("", {"model-b", "model-a"}, auto_select_single=True)

    def test_resolve_server_model_id_allows_manual_id_when_server_reports_none(self):
        self.assertEqual(resolve_server_model_id("manual-model", set()), "manual-model")
        with self.assertRaisesRegex(AutoCaptionError, "none reported"):
            resolve_server_model_id("", set(), auto_select_single=True)

    def test_builds_llama_server_command(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            server = folder / "llama-server.exe"
            model = folder / "model.gguf"
            mmproj = folder / "mmproj.gguf"
            server.write_text("x", encoding="utf-8")
            model.write_text("x", encoding="utf-8")
            mmproj.write_text("x", encoding="utf-8")

            settings = CaptioningSettings(
                llama_server_path=str(server),
                base_url="http://127.0.0.1:8111/v1",
                caption_model="caption-model",
                llama_extra_args="--no-webui",
            )
            command = build_llama_server_command(settings, "caption", ModelAssets(model, mmproj))

        self.assertIn("llama-server.exe", command)
        self.assertIn("-m", command)
        self.assertIn("--mmproj", command)
        self.assertIn("--port 8111", command)
        self.assertIn("--alias caption-model", command)
        self.assertIn("-b 2048", command)
        self.assertIn("-ub 2048", command)
        self.assertIn("--reasoning off", command)

    def test_can_limit_llama_reasoning_budget_when_thinking_is_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            server = folder / "llama-server.exe"
            model = folder / "model.gguf"
            server.write_text("x", encoding="utf-8")
            model.write_text("x", encoding="utf-8")

            settings = CaptioningSettings(
                llama_server_path=str(server),
                disable_thinking=False,
            )
            command = build_llama_server_command(settings, "caption", ModelAssets(model, None))

        self.assertNotIn("--reasoning off", command)
        self.assertIn("--reasoning-budget 2048", command)

    def test_can_leave_llama_reasoning_unrestricted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            server = folder / "llama-server.exe"
            model = folder / "model.gguf"
            server.write_text("x", encoding="utf-8")
            model.write_text("x", encoding="utf-8")

            settings = CaptioningSettings(
                llama_server_path=str(server),
                disable_thinking=False,
                llama_reasoning_budget=-1,
            )
            command = build_llama_server_command(settings, "caption", ModelAssets(model, None))

        self.assertNotIn("--reasoning off", command)
        self.assertNotIn("--reasoning-budget", command)

    def test_disable_thinking_prefixes_qwen_no_think_directive(self):
        self.assertEqual(request_user_prompt(CaptioningSettings(), "Describe this."), "/no_think\n\nDescribe this.")
        self.assertEqual(request_user_prompt(CaptioningSettings(), "/no_think\nDescribe this."), "/no_think\nDescribe this.")
        self.assertEqual(
            request_user_prompt(CaptioningSettings(disable_thinking=False), "Describe this."),
            "Describe this.",
        )

    def test_strips_thinking_tags_from_model_output(self):
        self.assertEqual(strip_thinking_output("<think>hidden</think>\nVisible caption."), "Visible caption.")

    def test_debug_log_captures_thinking_and_token_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = CaptioningSettings(
                models_dir=temp,
                caption_model="debug-model",
                disable_thinking=False,
                debug_llm_output=True,
            )
            client = FakeClient(
                [
                    FakeResponse(
                        "<think>hidden chain</think>\nVisible caption.",
                        model="debug-model",
                        usage={"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
                        reasoning_content="separate hidden chain",
                    )
                ]
            )
            with patch("ideogram_captioner.llm_captioning._make_openai_client", return_value=client):
                result = chat_text(settings, "debug-model", "system", "user", 100)

            self.assertEqual(result, "Visible caption.")
            logs = list((Path(temp) / "llm_debug").glob("*.json"))
            self.assertEqual(len(logs), 1)
            data = json.loads(logs[0].read_text(encoding="utf-8"))
            self.assertEqual(data["usage"]["total_tokens"], 17)
            self.assertEqual(data["thinking_blocks"], ["hidden chain"])
            self.assertEqual(data["reasoning_content"], "separate hidden chain")
            self.assertEqual(data["visible_content"], "Visible caption.")

    def test_text_chat_warns_when_thinking_returns_no_visible_output(self):
        client = FakeClient([FakeResponse("", finish_reason="length", model="caption-model")])

        with patch("ideogram_captioner.llm_captioning._make_openai_client", return_value=client):
            with self.assertRaises(AutoCaptionError) as raised:
                chat_text(
                    CaptioningSettings(disable_thinking=False),
                    model="caption-model",
                    system="system",
                    user="Describe this.",
                    max_tokens=32,
                )

        self.assertIn("Thinking/reasoning is enabled", str(raised.exception))
        self.assertIn("finish_reason=length", str(raised.exception))
        self.assertIn("Thinking token budget", str(raised.exception))
        self.assertEqual(len(client.completions.requests), 1)
        self.assertEqual(client.completions.requests[0]["messages"][1]["content"], "Describe this.")

    def test_vision_chat_uses_image_first_then_text_first_fallback(self):
        client = FakeClient(
            [
                FakeResponse("", finish_reason="length", model="vision-model"),
                FakeResponse("Visible caption.", model="vision-model"),
            ]
        )

        with patch("ideogram_captioner.llm_captioning._make_openai_client", return_value=client), patch(
            "ideogram_captioner.llm_captioning.image_to_data_url",
            return_value="data:image/png;base64,abc",
        ):
            result = chat_vision(
                CaptioningSettings(disable_thinking=False),
                model="vision-model",
                image_path=Path("sample.png"),
                system="system",
                user="Describe this image.",
                max_tokens=32,
            )

        self.assertEqual(result, "Visible caption.")
        self.assertEqual(len(client.completions.requests), 2)
        first_user_content = client.completions.requests[0]["messages"][1]["content"]
        retry_user_content = client.completions.requests[1]["messages"][1]["content"]
        self.assertEqual(first_user_content[0]["type"], "image_url")
        self.assertEqual(first_user_content[1]["text"], "Describe this image.")
        self.assertEqual(retry_user_content[0]["text"], "Describe this image.")
        self.assertEqual(retry_user_content[1]["type"], "image_url")

    def test_vision_chat_warns_when_thinking_returns_no_visible_output(self):
        client = FakeClient(
            [
                FakeResponse("", finish_reason="length", model="vision-model"),
                FakeResponse("", finish_reason="length", model="vision-model"),
            ]
        )

        with patch("ideogram_captioner.llm_captioning._make_openai_client", return_value=client), patch(
            "ideogram_captioner.llm_captioning.image_to_data_url",
            return_value="data:image/png;base64,abc",
        ):
            with self.assertRaises(AutoCaptionError) as raised:
                chat_vision(
                    CaptioningSettings(disable_thinking=False),
                    model="vision-model",
                    image_path=Path("sample.png"),
                    system="system",
                    user="Describe this image.",
                    max_tokens=32,
                )

        self.assertIn("Thinking/reasoning is enabled", str(raised.exception))
        self.assertIn("finish_reason=length", str(raised.exception))
        self.assertIn("Context size", str(raised.exception))

    def test_safe_repo_dir(self):
        self.assertEqual(safe_repo_dir("org/model name"), "org__model__name")


if __name__ == "__main__":
    unittest.main()
