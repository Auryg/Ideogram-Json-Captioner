import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ideogram_captioner.gui import (
    AI_IMAGE_STATE_ACTIVE,
    AI_IMAGE_STATE_DONE,
    CAPTION_FILTER_BOTH,
    CAPTION_FILTER_JSON,
    CAPTION_FILTER_ORIGINAL,
    COPY_MODE_BOTH,
    COPY_MODE_JSON,
    COPY_MODE_TEXT,
    IMAGE_LIST_AI_ACTIVE_BG,
    IMAGE_LIST_AI_DONE_BG,
    IMAGE_LIST_AI_QUEUED_BG,
    IMAGE_LIST_BG,
    IMAGE_SORT_JSON_MISSING,
    IMAGE_SORT_TEXT_MISSING,
    CaptioningPreferencesDialog,
    CaptionEditorApp,
)
from ideogram_captioner.schema import default_caption
from ideogram_captioner.store import CaptionStore


class FakeVar:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class FakeButton:
    def __init__(self) -> None:
        self.config: dict[str, str] = {}

    def configure(self, **kwargs: str) -> None:
        self.config.update(kwargs)


class FakeCombo:
    def __init__(self) -> None:
        self.config: dict[str, object] = {}

    def configure(self, **kwargs: object) -> None:
        self.config.update(kwargs)


class FakeListbox:
    def __init__(self, selection: tuple[int, ...] = ()) -> None:
        self.selection = set(selection)
        self.activated: int | None = None
        self.seen: int | None = None
        self.anchor: int | None = None
        self.xview: float | None = None
        self.yview_top = 0.0
        self.configs: dict[int, dict[str, str]] = {}

    def curselection(self) -> tuple[int, ...]:
        return tuple(sorted(self.selection))

    def nearest(self, y: int) -> int:
        return int(y)

    def bbox(self, index: int) -> tuple[int, int, int, int]:
        return (0, int(index), 10, 1)

    def selection_clear(self, _first: int, _last: str) -> None:
        self.selection.clear()

    def selection_set(self, first: int, last: int | str | None = None) -> None:
        if last is None:
            self.selection.add(int(first))
            return
        if last == "end":
            raise AssertionError("FakeListbox needs an explicit item count for selection_set(..., 'end').")
        for index in range(int(first), int(last) + 1):
            self.selection.add(index)

    def activate(self, index: int) -> None:
        self.activated = int(index)

    def see(self, index: int) -> None:
        self.seen = int(index)

    def xview_moveto(self, fraction: float) -> None:
        self.xview = float(fraction)

    def yview(self) -> tuple[float, float]:
        return (self.yview_top, min(1.0, self.yview_top + 0.25))

    def yview_moveto(self, fraction: float) -> None:
        self.yview_top = float(fraction)

    def selection_anchor(self, index: int) -> None:
        self.anchor = int(index)

    def itemconfigure(self, index: int, **kwargs: str) -> None:
        self.configs.setdefault(int(index), {}).update(kwargs)


class FakeEvent:
    def __init__(self, y: int, state: int = 0) -> None:
        self.y = y
        self.state = state


class FakeText:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self, _start: str, _end: str) -> str:
        return self.value


class GuiSelectionTests(unittest.TestCase):
    def make_app(self, images: list[Path], selection: tuple[int, ...], current_index: int) -> CaptionEditorApp:
        app = object.__new__(CaptionEditorApp)
        app.images = images
        app.image_list = FakeListbox(selection)
        app.current_index = current_index
        app.loading_list = False
        app.pending_image_select_job = None
        app.last_image_click_index = None
        app.image_selection_anchor_path = None
        app.ai_image_states = {}
        app.after_idle = lambda callback: "job"
        app.after_cancel = lambda _job: None
        return app

    def test_current_image_selection_paths_deduplicates_selected_indices(self) -> None:
        images = [Path("a.png"), Path("b.png"), Path("c.png")]
        app = self.make_app(images, (0, 2), 0)

        self.assertEqual(app.current_image_selection_paths(), [Path("a.png"), Path("c.png")])

    def test_populate_image_selection_restores_paths_after_reorder(self) -> None:
        images = [Path("c.png"), Path("a.png"), Path("b.png")]
        app = self.make_app(images, (), 2)

        app.populate_image_selection(selected_paths=[Path("a.png"), Path("c.png")])

        self.assertEqual(app.image_list.curselection(), (0, 1))
        self.assertEqual(app.image_list.activated, 2)
        self.assertEqual(app.image_list.seen, 2)

    def test_populate_image_selection_falls_back_to_current_when_paths_disappear(self) -> None:
        images = [Path("a.png"), Path("b.png"), Path("c.png")]
        app = self.make_app(images, (), 1)

        app.populate_image_selection(selected_paths=[Path("missing.png")])

        self.assertEqual(app.image_list.curselection(), (1,))
        self.assertEqual(app.image_list.activated, 1)
        self.assertEqual(app.image_list.seen, 1)

    def test_populate_image_selection_can_preserve_list_viewport(self) -> None:
        images = [Path("a.png"), Path("b.png"), Path("c.png")]
        app = self.make_app(images, (), 1)

        app.populate_image_selection(reveal_current=False)

        self.assertEqual(app.image_list.curselection(), (1,))
        self.assertEqual(app.image_list.activated, 1)
        self.assertIsNone(app.image_list.seen)
        self.assertEqual(app.image_list.xview, 0.0)

    def test_horizontal_image_list_scroll_is_blocked(self) -> None:
        app = self.make_app([Path("a.png")], (), 0)

        self.assertEqual(app.block_image_list_horizontal_scroll(), "break")
        self.assertEqual(app.image_list.xview, 0.0)

    def test_image_list_vertical_scroll_can_be_restored(self) -> None:
        app = self.make_app([Path("a.png")], (), 0)
        app.image_list.yview_top = 0.6
        scroll = app.image_list_vertical_scroll()

        app.image_list.yview_top = 0.0
        app.restore_image_list_vertical_scroll(scroll)

        self.assertEqual(app.image_list.yview_top, 0.6)

    def test_ai_image_state_colors_queue_active_done_and_clear(self) -> None:
        app = self.make_app([Path("a.png"), Path("b.png"), Path("c.png")], (), 0)

        app.set_ai_image_queue([Path("a.png"), Path("b.png"), Path("c.png")])
        app.set_ai_image_state(Path("b.png"), AI_IMAGE_STATE_ACTIVE)
        app.set_ai_image_state(Path("b.png"), AI_IMAGE_STATE_DONE)
        app.set_ai_image_state(Path("c.png"), None)
        app.finish_ai_image_states()

        self.assertEqual(app.image_list.configs[0]["background"], IMAGE_LIST_BG)
        self.assertEqual(app.image_list.configs[1]["background"], IMAGE_LIST_AI_DONE_BG)
        self.assertEqual(app.image_list.configs[2]["background"], IMAGE_LIST_BG)

        app.set_ai_image_state(Path("b.png"), AI_IMAGE_STATE_ACTIVE)

        self.assertEqual(app.image_list.configs[0]["background"], IMAGE_LIST_BG)
        self.assertEqual(app.image_list.configs[1]["background"], IMAGE_LIST_AI_ACTIVE_BG)

        app.set_ai_image_queue([Path("a.png"), Path("c.png")])

        self.assertEqual(app.image_list.configs[0]["background"], IMAGE_LIST_AI_QUEUED_BG)
        self.assertEqual(app.image_list.configs[1]["background"], IMAGE_LIST_BG)
        self.assertEqual(app.image_list.configs[2]["background"], IMAGE_LIST_AI_QUEUED_BG)

    def test_ai_status_message_prefixes_current_filename(self) -> None:
        event = {"message": "Checking caption assets...", "image_name": "sample.png"}

        self.assertEqual(
            CaptionEditorApp.ai_status_message(event),
            "sample.png: Checking caption assets...",
        )
        self.assertEqual(
            CaptionEditorApp.ai_status_message({"message": "sample.png: retrying previous failure.", "image_name": "sample.png"}),
            "sample.png: retrying previous failure.",
        )

    def test_shift_click_selects_range_from_saved_anchor(self) -> None:
        images = [Path("a.png"), Path("b.png"), Path("c.png"), Path("d.png"), Path("e.png")]
        app = self.make_app(images, (1,), 1)
        app.image_selection_anchor_path = Path("b.png")

        result = app.remember_image_list_click(FakeEvent(y=4, state=0x0001))

        self.assertEqual(result, "break")
        self.assertEqual(app.image_list.curselection(), (1, 2, 3, 4))
        self.assertEqual(app.image_list.activated, 4)
        self.assertEqual(app.image_list.seen, 4)
        self.assertEqual(app.image_list.anchor, 1)

    def test_caption_filter_paths_deduplicates_same_caption_source(self) -> None:
        image = Path("sample.png")

        self.assertEqual(
            CaptionEditorApp.caption_filter_paths_for_image(image, ".txt", ".txt", CAPTION_FILTER_BOTH),
            [Path("sample.txt")],
        )

    def test_caption_filter_matches_json_or_original_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.png"
            image.write_bytes(b"x")
            image.with_suffix(".json").write_text('{"high_level_description":"bright red sign"}', encoding="utf-8")
            image.with_suffix(".txt").write_text("plain caption with blue paint", encoding="utf-8")

            self.assertTrue(
                CaptionEditorApp.image_matches_caption_filter(
                    image, "red sign", ".json", ".txt", CAPTION_FILTER_JSON
                )
            )
            self.assertFalse(
                CaptionEditorApp.image_matches_caption_filter(
                    image, "blue paint", ".json", ".txt", CAPTION_FILTER_JSON
                )
            )
            self.assertTrue(
                CaptionEditorApp.image_matches_caption_filter(
                    image, "blue paint", ".json", ".txt", CAPTION_FILTER_ORIGINAL
                )
            )
            self.assertTrue(
                CaptionEditorApp.image_matches_caption_filter(
                    image, "bright red", ".json", ".txt", CAPTION_FILTER_BOTH
                )
            )

    def test_missing_json_caption_mode_filters_to_missing_or_blank_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            captioned = folder / "captioned.png"
            blank = folder / "blank.png"
            empty = folder / "empty.png"
            missing_a = folder / "missing-a.png"
            missing_b = folder / "missing-b.png"
            for image in (captioned, blank, empty, missing_a, missing_b):
                image.write_bytes(b"x")
            captioned.with_suffix(".json").write_text('{"high_level_description":"done"}', encoding="utf-8")
            blank.with_suffix(".json").write_text("{}", encoding="utf-8")
            empty.with_suffix(".json").write_text("   ", encoding="utf-8")

            app = object.__new__(CaptionEditorApp)
            app.all_images = [missing_b, captioned, empty, blank, missing_a]
            app.images = list(app.all_images)
            app.current_index = -1
            app.caption_filter_matches = None
            app.caption_extension_var = FakeVar(".json")
            app.original_extension_var = FakeVar(".txt")
            app.image_sort_var = FakeVar(IMAGE_SORT_JSON_MISSING)

            app.apply_image_sort(preserve_current=False)

            self.assertEqual(app.images, [blank, empty, missing_a, missing_b])

    def test_missing_text_caption_mode_filters_to_images_without_text_caption_or_blank_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            captioned = folder / "captioned.png"
            blank = folder / "blank.png"
            missing_a = folder / "missing-a.png"
            missing_b = folder / "missing-b.png"
            for image in (captioned, blank, missing_a, missing_b):
                image.write_bytes(b"x")
            captioned.with_suffix(".txt").write_text("plain text", encoding="utf-8")
            blank.with_suffix(".txt").write_text("  ", encoding="utf-8")

            app = object.__new__(CaptionEditorApp)
            app.all_images = [missing_b, captioned, blank, missing_a]
            app.images = list(app.all_images)
            app.current_index = -1
            app.caption_filter_matches = None
            app.caption_extension_var = FakeVar(".json")
            app.original_extension_var = FakeVar(".txt")
            app.image_sort_var = FakeVar(IMAGE_SORT_TEXT_MISSING)

            app.apply_image_sort(preserve_current=False)

            self.assertEqual(app.images, [blank, missing_a, missing_b])

    def test_blank_json_caption_form_is_not_saved_as_new_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.png"
            image.write_bytes(b"x")

            app = object.__new__(CaptionEditorApp)
            app.store = CaptionStore(folder, ".json")
            app.images = [image]
            app.current_index = 0
            app.current_caption = default_caption()
            app.dirty = True
            app.original_dirty = False
            app.autosave_job = None
            app.original_autosave_job = None
            app.image_list = FakeListbox((0,))
            app.status_var = FakeVar("")
            app.sync_caption_from_form = lambda: None

            app.save_current()

            self.assertFalse(image.with_suffix(".json").exists())
            self.assertFalse(app.dirty)
            self.assertEqual(app.status_var.get(), "Blank JSON caption not saved.")

    def test_json_caption_style_fields_infers_mode_and_medium(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            caption = folder / "sample.json"
            caption.write_text(
                '{"style_description":{"_mode":"art_style","medium":"painting","photo":"ignored","art_style":"oil"}}',
                encoding="utf-8",
            )

            self.assertEqual(CaptionEditorApp.json_caption_style_fields(caption), ("art_style", "painting"))

    def test_image_matches_style_filter_uses_json_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.png"
            image.write_bytes(b"x")
            image.with_suffix(".json").write_text(
                '{"style_description":{"photo":"85mm lens","medium":"Photograph"}}',
                encoding="utf-8",
            )

            self.assertTrue(CaptionEditorApp.image_matches_style_filter(image, ".json", "photo", "photograph"))
            self.assertTrue(CaptionEditorApp.image_matches_style_filter(image, ".json", "", "photograph"))
            self.assertFalse(CaptionEditorApp.image_matches_style_filter(image, ".json", "art_style", "photograph"))
            self.assertFalse(CaptionEditorApp.image_matches_style_filter(image, ".json", "photo", "painting"))

    def test_excel_tsv_row_quotes_cells_with_excel_delimiters(self) -> None:
        self.assertEqual(
            CaptionEditorApp.excel_tsv_row(['{"a":"b"}', 'caption with "quotes"\nand tab\tinside']),
            '"{""a"":""b""}"\t"caption with ""quotes""\nand tab\tinside"',
        )

    def make_clipboard_app(self) -> CaptionEditorApp:
        app = object.__new__(CaptionEditorApp)
        app.images = [Path("sample.png")]
        app.current_index = 0
        app.current_caption = {"high_level_description": "json caption"}
        app.original_text = FakeText('text caption with "quotes"')
        app.status_var = FakeVar("")
        app.copy_mode_var = FakeVar(COPY_MODE_BOTH)
        app.sync_caption_from_form = lambda: None
        app.clipboard_value = ""
        app.clipboard_clear = lambda: setattr(app, "clipboard_value", "")
        app.clipboard_append = lambda value: setattr(app, "clipboard_value", value)
        return app

    def test_copy_json_caption_to_clipboard(self) -> None:
        app = self.make_clipboard_app()

        app.copy_json_caption_to_clipboard()

        self.assertIn('"high_level_description":"json caption"', app.clipboard_value)
        self.assertEqual(app.status_var.get(), "Copied JSON caption to clipboard.")

    def test_copy_text_caption_to_clipboard(self) -> None:
        app = self.make_clipboard_app()

        app.copy_text_caption_to_clipboard()

        self.assertEqual(app.clipboard_value, 'text caption with "quotes"')
        self.assertEqual(app.status_var.get(), "Copied text caption to clipboard.")

    def test_copy_caption_pair_to_clipboard_uses_excel_ready_columns(self) -> None:
        app = self.make_clipboard_app()

        app.copy_caption_pair_to_clipboard()

        self.assertIn("\t", app.clipboard_value)
        self.assertTrue(app.clipboard_value.startswith('"text caption with ""quotes"""\t'))
        self.assertEqual(app.status_var.get(), "Copied text and JSON captions for Excel.")

    def test_copy_selected_caption_to_clipboard_uses_json_mode(self) -> None:
        app = self.make_clipboard_app()
        app.copy_mode_var.set(COPY_MODE_JSON)

        app.copy_selected_caption_to_clipboard()

        self.assertIn('"high_level_description":"json caption"', app.clipboard_value)
        self.assertNotIn("\t", app.clipboard_value)
        self.assertEqual(app.status_var.get(), "Copied JSON caption to clipboard.")

    def test_copy_selected_caption_to_clipboard_uses_text_mode(self) -> None:
        app = self.make_clipboard_app()
        app.copy_mode_var.set(COPY_MODE_TEXT)

        app.copy_selected_caption_to_clipboard()

        self.assertEqual(app.clipboard_value, 'text caption with "quotes"')
        self.assertEqual(app.status_var.get(), "Copied text caption to clipboard.")

    def test_successful_copy_flashes_copy_button_feedback(self) -> None:
        app = self.make_clipboard_app()
        app.copy_button = FakeButton()
        app.copy_feedback_job = None
        scheduled: list[tuple[int, object]] = []
        app.after = lambda delay, callback: scheduled.append((delay, callback)) or "copy-feedback"
        app.after_cancel = lambda _job: None

        app.copy_text_to_clipboard("copied value", "Copied value.")

        self.assertEqual(app.copy_button.config["text"], "Copied")
        self.assertEqual(app.copy_button.config["style"], "Accent.TButton")
        self.assertEqual(app.copy_feedback_job, "copy-feedback")
        self.assertEqual(scheduled[0][0], 900)

        scheduled[0][1]()

        self.assertEqual(app.copy_button.config["text"], "Copy")
        self.assertEqual(app.copy_button.config["style"], "TButton")
        self.assertIsNone(app.copy_feedback_job)

    def test_server_discovery_populates_choices_and_selects_only_model(self) -> None:
        dialog = object.__new__(CaptioningPreferencesDialog)
        dialog.base_url_var = FakeVar("http://127.0.0.1:1234/v1")
        dialog.api_key_var = FakeVar("key")
        dialog.runtime_mode_var = FakeVar("Connect to existing server")
        dialog.caption_model_var = FakeVar("local-model")
        dialog.bbox_model_var = FakeVar("local-model")
        caption_combo = FakeCombo()
        bbox_combo = FakeCombo()
        dialog.model_id_combos = {"caption": caption_combo, "bbox": bbox_combo}

        with (
            patch("ideogram_captioner.gui.server_model_ids", return_value={"lm-studio/qwen-vision"}) as discover,
            patch("ideogram_captioner.gui.messagebox.showinfo") as showinfo,
        ):
            dialog._test_server()

        discover.assert_called_once_with("http://127.0.0.1:1234/v1", "key", timeout=5.0)
        self.assertEqual(caption_combo.config["values"], ["lm-studio/qwen-vision"])
        self.assertEqual(bbox_combo.config["values"], ["lm-studio/qwen-vision"])
        self.assertEqual(dialog.caption_model_var.get(), "lm-studio/qwen-vision")
        self.assertEqual(dialog.bbox_model_var.get(), "lm-studio/qwen-vision")
        showinfo.assert_called_once()


if __name__ == "__main__":
    unittest.main()
