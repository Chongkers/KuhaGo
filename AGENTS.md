# AGENTS.md — Kuha Architecture & Dev Directives

## 1. Project Context
* **App:** Kuha — local AI organizer for event photos and videos, for **everyone**.
  * **Personal mode:** anyone with phone photos/videos from a birthday, Noche Buena, trip, graduation, etc.
  * **Pro mode:** freelance photographers/videographers (RAW, gig tracking, editor integration).
  * One engine, one codebase. Pro mode = Personal mode + extra panels.
* **Target:** Hackathon MVP, **13-hour build**. Feature freeze at hour 9.
* **Platform:** Desktop, run as a local web app (no native packaging, no phone app).
* **Stack:**
  * Backend: Python 3.11, FastAPI, SQLite (`sqlite3` or SQLModel), PyTorch, `sse-starlette`.
  * ML/IO: `open_clip_torch`, `silero-vad`, `rawpy`, `pillow-heif`, `opencv-python-headless`, `imagehash`, `Pillow`, `numpy`, FFmpeg (subprocess), ExifTool (subprocess).
  * Frontend: React 19, Vite, TypeScript, Tailwind CSS.
* **Guardrails:** Zero cloud AI, zero network calls at runtime (only `scripts/download_models.py` may download). Never modify or move original media except copying in the user-confirmed Export step.

---

## 2. Demo Flow (everything serves this)
1. **Personal (≈45 s):** New Event → Birthday template → organize a folder of phone photos/videos → moments appear → missed-moment check ("No group photo") → Highlights → search "lola with the cake" → Export.
2. **Pro (≈60 s):** Switch to Pro → Debut gig → add example photos → ingest RAW + video → A-roll/B-roll → missed moments → write XMP and show keywords in darktable/RapidRAW → finals matching ("3 selects not yet edited").
3. Wi-Fi visibly off the whole time.

**Priority if time runs short:**
Missed-moment check > Ingest/classify > Export > Mode switch + personal templates > Highlights > Talking/scenery (A/B) > XMP (Pro) > Search > Finals matching (Pro).
Never cut the missed-moment check.

---

## 3. Modes & UI Language
* Mode is chosen on first launch ("I'm organizing my own photos" / "I shoot for clients") and switchable in Settings. Stored per event (`events.mode`).
* **Personal mode uses plain language.** Never show jargon there. Terminology map (single source: `frontend/src/lib/terms.ts`):

| Internal | Personal mode | Pro mode |
|---|---|---|
| event | Event | Gig |
| shot_items | Moments checklist | Shot list |
| refs | Example photos | References |
| roll A / B | Talking clips / Scenery clips | A-roll / B-roll |
| is_select | Favorites | Selects |
| coverage | Missed moments | Coverage |
| export | Export folders | Archive |

* Personal mode hides: fee, paid, client, XMP, finals, RAW-specific settings.

---

## 4. Repository Layout
```
backend/
  app/
    main.py            # FastAPI app, CORS for localhost:5173, router registration
    config.py          # thresholds, paths, device selection (cuda > mps > cpu)
    db.py              # SQLite schema + helpers
    schemas.py         # Pydantic request/response models
    routes/            # events.py, shot_items.py, ingest.py, coverage.py, highlights.py,
                       # search.py, finals.py, xmp.py, export.py
  ml/
    clip_engine.py     # singleton model load, embed_images(), embed_texts()
    vad.py             # speech_ratio(audio_path) -> float
    classifier.py      # classify_moments()
    coverage.py        # compute_coverage()
    highlights.py      # group_duplicates(), pick_highlights()
    search.py          # search(query)
    matcher.py         # match_finals() (Pro)
  media/
    previews.py        # RAW/JPEG/HEIC/PNG -> preview JPEG (cached), EXIF capture time
    video.py           # ffmpeg frame + audio extraction
    xmp.py             # read ratings, write keywords via ExifTool (Pro)
    export.py          # plan + execute folder copy (personal & pro layouts)
  templates/           # birthday, christmas, graduation, trip, reunion, debut, wedding, binyag (.json)
  scripts/
    download_models.py # only script allowed to use the network
    eval.py            # metrics vs. labels CSV
  requirements.txt
frontend/
  src/
    api/client.ts      # typed fetch wrappers + SSE helper
    lib/terms.ts       # mode-aware terminology
    pages/             # Welcome.tsx (mode pick), Events.tsx, EventDetail.tsx, Settings.tsx
    components/        # MomentChecklist, ExampleDrop, IngestProgress, MomentGrid, MissedMoments,
                       # Highlights, SearchBar, ClipsView, FinalsView (Pro), GigFields (Pro)
data/                  # gitignored: previews, embeddings, sqlite db, test sets
```
Do not name any package `io` (shadows stdlib).

---

## 5. Data Model (SQLite)
```
events(id, mode['personal'|'pro'], name, event_type, date, venue,
       client, fee, paid, status)                     -- client/fee/paid/status: Pro only, nullable
shot_items(id, event_id, moment, prompt, required)
refs(id, shot_item_id, image_path, emb_index)
media(id, event_id, path, kind['photo'|'video'], preview_path, emb_index, taken_at,
      moment, confidence, roll['A'|'B'|NULL], speech_ratio, sharpness,
      dup_group, is_highlight, rating, is_select)
finals(id, event_id, path, matched_media_id, match_score)   -- Pro only
```
* Embeddings: L2-normalized float32, stored per event in `data/events/{event_id}/emb.npy`; `emb_index` is the row.
* Video media: embedding = mean of its frame embeddings (re-normalized).
* Pro `status`: `inquiry | booked | shot | editing | delivered`.

---

## 6. Core Subsystems

### Agent 1: Ingest (`media/previews.py`, `media/video.py`, `routes/ingest.py`)
* Register HEIC support once at startup: `pillow_heif.register_heif_opener()`.
* **Phone/JPEG/HEIC/PNG:** open with Pillow, apply EXIF orientation (`ImageOps.exif_transpose`).
* **RAW (Pro):** `rawpy.imread(p).extract_thumb()`; JPEG → bytes, BITMAP → Pillow. Fallback: `exiftool -b -PreviewImage`.
* Resize preview to max 1024 px; cache in `data/events/{id}/previews/`. Read `DateTimeOriginal` into `taken_at` (fallback: file mtime).
* **Video (MP4/MOV):** `ffmpeg -i in -vf fps=0.5,scale=448:-1 frames/%04d.jpg`; audio `ffmpeg -i in -ac 1 -ar 16000 audio.wav`.
* Ingest accepts multiple folders (e.g., several family members' phone dumps).
* Background job; SSE progress `{stage, done, total}`. Skip and log unreadable files; never crash the job.

### Agent 2: CLIP Engine (`ml/clip_engine.py`)
* `open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')`.
* **Singleton**, loaded once at startup. `torch.inference_mode()`. Batch size 32 (lower on CPU if memory-bound).
* `embed_images(paths) -> np.ndarray[N,512]`, `embed_texts(texts) -> np.ndarray[M,512]`, both L2-normalized.

### Agent 3: Moment Classifier (`ml/classifier.py`)
* Per moment `m`: `T_m` = mean text embedding of its prompts; `R_m` = mean embedding of its example photos (if any). Both normalized.
* `score(x, m) = α·cos(x, T_m) + (1−α)·cos(x, R_m)`; `α = 0.4` if examples exist, else `1.0`.
* `moment = argmax`; `confidence = top1 − top2`. `"Unsorted"` if `top1 < τ_score` or `confidence < τ_margin`.
* Contract: `classify_moments(emb: np.ndarray, items: list[ShotItem]) -> list[tuple[str, float]]`.

### Agent 4: Missed Moments / Coverage (`ml/coverage.py`)
* For each shot item: `best = max score(x, m)` over all event media; `count` = media assigned to `m`.
* `covered = count > 0 or best >= τ_cov`.
* Return `[{moment, covered, count, best_score, candidates: top-3 media ids}]`, missed items first.

### Agent 5: Highlights (`ml/highlights.py`)
* **Sharpness:** variance of Laplacian on grayscale preview (`cv2.Laplacian(...).var()`), normalized per event (percentile rank).
* **Duplicate groups:** photos within the same moment, `|taken_at` difference| ≤ 10 s, and (`pHash Hamming ≤ 8` or `cos ≥ 0.95`) → same `dup_group` (union-find).
* **Highlights:** per moment, keep the sharpest photo per `dup_group`, drop items below `SHARPNESS_MIN_PCT`, take top `HIGHLIGHTS_PER_MOMENT` (default 5).
* This is lightweight personal curation, not pro culling. Do not add eye/blink detection.

### Agent 6: Talking vs. Scenery Clips (`ml/vad.py`)
* `speech_ratio = speech_seconds / duration` via Silero VAD.
* `talk_score = cos(video_emb, mean(embed_texts(["a person talking to the camera", "a person giving a speech"])))`.
* `roll = 'A' if speech_ratio >= 0.35 and talk_score >= τ_talk else 'B'`. UI label comes from `terms.ts`.

### Agent 7: Search (`ml/search.py`)
* `search(event_id | None, query, k=24)`: `embed_texts([query])` → cosine against stored embeddings → top-k media. Event-scoped by default; optional across all events.

### Agent 8: Finals Matcher — Pro only (`ml/matcher.py`)
* Selects = media with `rating >= SELECT_MIN_RATING` (default 3).
* For each final: top-5 selects by CLIP cosine → lowest pHash Hamming. Accept if `hamming <= 12` or `cos >= 0.92`.
* Output: matched pairs, unmatched finals, selects with no final ("not yet edited").

### Agent 9: XMP (Pro) & Export (`media/xmp.py`, `media/export.py`)
* Read ratings: `exiftool -j -XMP:Rating <files>` in one batched call.
* Sidecar path by mode: `lightroom` → `IMG_0001.xmp`; `darktable` → `IMG_0001.CR3.xmp`.
* Back up existing sidecars to `.kuha_backup/` before writing.
* Write **only** keyword tags: `-XMP-dc:Subject+=<kw>` and `-XMP-lr:HierarchicalSubject+=Kuha|<Event>|<Moment>`, plus `A-roll`/`B-roll`. Use `-overwrite_original`. Never rewrite the whole sidecar (darktable stores edit history there).
* **Export:** `plan_export(event_id, dest)` returns the full copy list (dry run); `execute_export()` copies with `shutil.copy2`. Copy, never move. Sanitize folder names for Windows (`<>:"/\|?*`).
  * Personal layout: `{date}_{name}/Highlights/`, `Moments/<moment>/`, `Videos/Talking/`, `Videos/Scenery/`, `Unsorted/`.
  * Pro layout: `{date}_{client}_{event}/01_RAW/<moment>/`, `02_SELECTS/`, `03_FINALS/`, `04_VIDEO/A-ROLL/`, `04_VIDEO/B-ROLL/`, `05_REFERENCES/`.

---

## 7. Event Templates (`templates/*.json`)
```json
{ "event_type": "birthday", "modes": ["personal", "pro"],
  "moments": [
    { "moment": "Blowing the Candles", "required": true,
      "prompts": ["a person blowing out candles on a birthday cake",
                  "a birthday cake with lit candles"] }
  ] }
```
* Prompts describe what is **visible**, not cultural names (CLIP doesn't know "18 Roses" or "Noche Buena").
* **Birthday:** Arrival/Guests, Blowing the Candles, Cake, Food/Handaan, Games, Gift Opening, Group Photo, Decorations.
* **Christmas / Noche Buena:** Food on the Table, Family Group Photo, Gift Exchange, Christmas Tree/Parol, Kids Opening Gifts, Mass/Church.
* **Graduation:** Marching, Receiving Diploma, Toga Portrait, Family with Graduate, Celebration Meal.
* **Trip / Outing:** Travel/Transport, Scenery, Beach/Pool, Food, Group Photo, Selfies.
* **Family Reunion:** Group Photo, Elders/Lolo & Lola, Kids Playing, Food, Games.
* **Debut:** Grand Entrance, 18 Roses, 18 Candles, 18 Treasures/Gifts, Cotillion, Cake, Speech/Toast, Family Formals, Details/Decor.
* **Wedding:** Bride Prep, Groom Prep, Processional, Vows, Veil & Cord, Ring Exchange, Kiss, Family Formals, Reception Entrance, Cake, Details.
* **Binyag:** Arrival, Baptism Rite (water), Godparents, Candle, Family Formals, Reception, Details.

---

## 8. API Contract
```
GET/POST /events               PATCH /events/{id}
POST /events/{id}/template     {event_type}
GET  /events/{id}/shot_items   POST /shot_items/{id}/refs (multipart)
POST /events/{id}/ingest       {folders: [...]}   → {job_id}
GET  /jobs/{job_id}/events     (SSE progress)
GET  /events/{id}/media?moment=&roll=&highlights=
GET  /media/{id}/preview       (serves cached JPEG)
GET  /events/{id}/coverage
GET  /events/{id}/highlights
GET  /search?q=&event_id=
POST /events/{id}/export       {dest, dry_run}
POST /events/{id}/finals       {folder}           (Pro)
POST /events/{id}/xmp          {mode: 'lightroom'|'darktable'}   (Pro)
```
Folder paths are typed/picked by the user (local app; originals are never uploaded anywhere).

---

## 9. Evaluation (`scripts/eval.py`)
* Inputs: `labels.csv` per test set (`filename, moment, roll, dup_group`) — one **personal** set (phone photos from a real family event) and one **pro** set (RAW + video).
* Report: moment accuracy (descriptions only vs. with example photos) per set, missed-moment detection, duplicate grouping accuracy, search top-5 hit rate (`queries.csv`), talking/scenery accuracy, finals match accuracy, photos/sec, hardware info.
* Write `results.md`. **Never hand-edit or invent numbers.** README metrics come only from this output.

---

## 10. Coding Agent Constraints & Guardrails
1. **Token efficiency:** When proposing edits, output only changed functions or localized diffs. Do not re-print unedited files.
2. **Offline at runtime:** No HTTP calls to external hosts outside `download_models.py`. No cloud AI APIs, no CDN assets in the frontend.
3. **Originals are sacred:** Never move, rename, or overwrite media files. Only sidecars (after backup, Pro) and the Export copy step write outside `data/`.
4. **Plain language in Personal mode:** All user-facing labels go through `terms.ts`. No jargon (A-roll, selects, XMP, RAW) in Personal mode.
5. **One model load:** CLIP and VAD are singletons loaded at startup, never per request.
6. **Batch everything:** Embeddings in batches; ExifTool reads/writes batched per call, not per file.
7. **No blocking requests:** Ingest, finals matching, and export run as background jobs with SSE progress.
8. **Fail soft:** A bad file is logged and skipped; jobs continue. Low confidence → `"Unsorted"`, never a forced label.
9. **Framework isolation:** Do not add Electron/Tauri packaging, a phone app, vector databases, Docker, auth, or cloud sync. SQLite + `.npy` only.
10. **Thresholds in config:** No magic numbers inside ML functions; read from `config.py`.
11. **Cross-platform paths:** Use `pathlib` everywhere; must run on Windows and macOS.
