# Ideogram JSON Captioner

A local desktop editor and generator for image-caption pairs in Ideogram 4 structured JSON
format or regular text captions. Use it to generate either regular captions or JSON captions, review image datasets, repair generated captions, edit source
caption text, and draw or adjust object/text bounding boxes.  You can also generate JSON captions based on your already existing and vetted captions.

The app runs locally by default. Auto-captioning sends requests only to the OpenAI-compatible endpoint
you configure in Preferences, such as a local llama.cpp, LM Studio, vLLM, or
Ollama-compatible server.

![Ideogram JSON Captioner screenshot](ideogramCaptionerScreenshot.png)

## Features

- Open a folder of images and step through them with keyboard shortcuts.
- Edit Ideogram JSON fields for high-level description, style, background,
  elements, rendered text, bounding boxes, and color palettes.
- Draw, move, resize, delete, and numerically edit bounding boxes.
- Keep editable text captions, such as `.txt` or `.original`, separate from
  structured JSON captions.
- Generate text captions, JSON captions from text, JSON captions from images,
  JSON refinements, and bounding boxes with a local or existing
  vision-language model server or local Ultralytics detector.
- Batch auto-caption selected images, retry failed captions, and undo the last
  auto-captioning job.
- Sort and filter the image list by name, modified date, missing/blank JSON/text captions, or
  failed auto-captioning jobs.
- Filter by caption text, structured JSON mode (`photo` or `art_style`), and
  structured JSON medium.
- Copy the current image into an `edit` subfolder for later external editing.

## Install

For most Windows users, download `IdeogramCaptioner.exe` from the GitHub
Releases page and run it.

To run from source instead:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ideogram_captioner
```

You can also start the app with:

```powershell
python run_captioner.py
```

## Basic Use

### Manual Captioning: 
1. Open a folder that contains images.
2. Choose the JSON caption extension to edit, usually `.json`.
3. Choose a text caption extension if you also want to edit source captions,
   such as `.txt` or `.original`.
4. Select an image, edit the fields, then use `Save` or `Enter` to save and move
   to the next image.

The app will also load image metadata as either Ideogram JSON or regular text
captions. This supports ComfyUI metadata, ComfyUI Ideogram Prompt Builder
nodes, direct Ideogram JSON in PNG/EXIF fields, and Automatic1111/Forge/Forge
NEO `parameters` prompt metadata.
   
### Automatic Captioning: 
If you don't already have an OpenAI-compatible server, the easiest way to get automatic captioning working is to grab llama.cpp from their releases section - https://github.com/ggml-org/llama.cpp/releases - and be sure to grab the CUDA .dlls and put them in the same folder as llama-server.exe if you're using an Nvidia card, otherwise it will probably be quite slow.
After you've done so, open the preferences and select where you've put llama-server.exe.  From there, just select what image(s) you want to caption and select the appropriate button from the Auto Captioning section.

For automatic captioning, I would probably recommend doing a regular text caption first (if you don't have one already), and then creating the JSON caption from that, but making them directly can also work.  You will definitely want to alter the prompts sent to the LLM as they've very basic - do a few images as a test to see what you need to specify.


## Keyboard Shortcuts

- `Enter`: save and move to the next image.
- `Shift+Enter`: insert a newline in a text field.
- `Tab` / `Shift+Tab`: move forward or backward between fields.
- Arrow keys: navigate images when focus is not inside an input.
- `Ctrl+Up` / `Ctrl+Down`: navigate images even when an input has focus.
- `Ctrl+S`: save.
- `Ctrl+A` in the image list: select all images for batch auto-captioning.
- `Esc`: cancel eyedropper mode or exit fullscreen.
- `F11`: toggle fullscreen.

## Auto Captioning

Open `Preferences` to choose the captioning runtime, model profile, endpoint,
model folder, prompt behavior, and bbox settings.

The default runtime is `Local llama.cpp`. In this mode, the app downloads the
selected Hugging Face GGUF model files if needed, starts `llama-server.exe`, and
sends OpenAI-compatible requests to `http://127.0.0.1:8000/v1`.

For the local runtime, put `llama-server.exe` in one of these locations:

- Beside `IdeogramCaptioner.exe` or the source checkout.
- In a `tools` folder beside the app.
- Anywhere else selected in Preferences.

The app reuses an already-running endpoint when one is available. Use
`Connect to existing server` for LM Studio, llama.cpp, vLLM, Ollama bridges, or
another OpenAI-compatible server you started yourself.

The right-side `Auto Captioning` buttons do the following:

- `Text Caption`: creates or replaces the plain text caption.
- `JSON from Text`: converts the text caption into Ideogram JSON.
- `JSON from Image`: creates Ideogram JSON directly from the image.
- `Refine JSON`: revises existing structured JSON from the image, the current
  JSON, the text caption, and custom instructions entered at run
  time.
- `Add/redo BBoxes`: localizes existing JSON elements with the selected bbox
  backend.
- `Retry Failed`: reruns images that have failed auto-captioning markers.
- `Clear Failed`: removes failed markers without rerunning the model.
- `Undo AI`: restores JSON/text captions from before the last auto-captioning job.

Multiple images can be selected with Shift, Ctrl, or `Ctrl+A`. The app asks for
confirmation before running an auto-captioning job on more than one image.

Image-based JSON operations automatically send a compact EXIF summary when the
image has supported camera fields such as camera/lens model, f-stop, focal
length, ISO, exposure time, exposure bias, metering, flash, or white balance.
GPS and other broad metadata are not sent.

The bbox backend can be changed in `Preferences` -> `Pipeline`:

- `VLM grounding`: sends existing JSON elements to the selected bbox VLM and is
  best for arbitrary detailed element descriptions.
- `YOLOE-26 prompts`: runs Ultralytics locally, defaults to
  `yoloe-26l-seg.pt`, and uses each JSON element description as an
  open-vocabulary text prompt. 
  
Ultralytics weights are downloaded by Ultralytics on first use. The
`Ultralytics image size` preference defaults to `1024`; set it to `0` to use
Ultralytics' own default, currently `640`.

## Failed Captions

When auto-captioning cannot complete, the image is marked with a local
`*.caption_failed.json` sidecar. This lets the app show failed items, retry them
later, or clear the failed state.

JSON parse failures get a repair pass before they are marked failed. If the
repair pass still fails, retry the failed images with a larger or more reliable
model, a larger context size, or a lower reasoning budget.

Failure markers are local working files and are ignored by git.

## Model Profiles

Model profiles live in `captioner_model_profiles.json` beside the app or source
checkout. This file is local and ignored by git so your model choices, local
paths, and experiments are not overwritten by pulls or uploaded by accident.

Use `Preferences` -> `Models` -> `Open Profiles File` to create or edit the
local profile file. The tracked `captioner_model_profiles.example.json` file is
the default seed.

A downloadable Hugging Face profile looks like this:

```json
{
  "id": "my-qwen-profile",
  "label": "Download: My Qwen model",
  "tasks": ["caption", "bbox"],
  "kind": "hf",
  "api_model": "my-qwen",
  "hf_repo": "org/repo-name",
  "model_filename": "model-file.gguf",
  "mmproj_filename": "mmproj-file.gguf"
}
```

`tasks` can be `["caption"]`, `["bbox"]`, or both. Supported profile kinds are:

- `hf`: download GGUF files from Hugging Face.
- `local`: use fixed `local_model_path` and optional `local_mmproj_path` values.
- `server`: skip downloads and local launch; send requests using `api_model` as
  the existing server model name.

The profile dropdown also includes one-off custom choices for Hugging Face GGUF
downloads and local GGUF files already on your computer.

## Prompt Overrides

Prompt overrides live in the ignored `captioner_prompts/` folder. Use
`Preferences` -> `Pipeline` -> `Open Prompts Folder` to create or open it.

The tracked `captioner_prompts.example/` folder shows the supported filenames.
You only need to copy the prompts you want to override; missing files fall back
to the built-in defaults.

Keep these placeholders if you edit the matching prompt:

- `text_to_json_user.txt`: `{caption}`
- `json_refine_user.txt`: `{instructions}`, `{source_caption}`, `{caption_json}`
- `bbox_user.txt`: `{context_json}`, `{targets_json}`

`creative_directive.txt` or `faithful_directive.txt` is appended to the JSON
system prompt automatically.

## Troubleshooting

- `Connection error`: the configured endpoint is not reachable, the local server
  failed to start, or the model process crashed. Check Preferences, confirm the
  server URL, and inspect any `server_logs/` output.
- `finish_reason=length` or empty assistant text: increase the llama context
  size, increase output tokens, reduce reasoning budget, or use a smaller image
  or model profile.
- JSON errors after generation: use the failed-caption filter and `Retry Failed`
  with a stronger model or larger context.
- Missing VLM bbox output: confirm the selected model profile supports vision
  and bbox tasks, and that the model has access to its `mmproj` file if
  required.
- Missing YOLOE-26 bbox output: lower the Ultralytics confidence threshold,
  increase `Ultralytics image size` for small objects, simplify overly long
  element descriptions, or try the largest YOLOE-26 model,
  `yoloe-26x-seg.pt`.
- Local model download problems: verify the Hugging Face repo id, filename, and
  network access, or use a local profile with files already on disk.

## License

MIT License. See [LICENSE](LICENSE).
