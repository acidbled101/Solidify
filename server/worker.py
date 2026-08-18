"""
Single background worker thread that owns the resident TRELLIS.2 pipeline.

Only one worker thread exists, and it pulls job ids off a blocking queue one
at a time. That single-consumer design IS the serialization guarantee: only
one job ever touches the (single) MPS device at a time, so there is no locking
needed around `pipeline.run()` itself, and thermal throttling from concurrent
GPU use is impossible by construction.

The pipeline is a lazy singleton (loaded once, ~100s, kept resident) so the
model-load cost is paid at most once for the whole server lifetime.
"""

# `import trellis_core` MUST come before torch is imported (directly or via
# trellis_core.pipeline), because it sets PYTORCH_ENABLE_MPS_FALLBACK and the
# sparse backends env vars, and appends the repo root to sys.path.
import trellis_core  # noqa: F401,E402  (must be first trellis-related import)

import logging
import re
# Module level, deliberately. This used to be imported inside _run_job, which
# made `os` a LOCAL of that function -- so the preview callbacks defined above
# it closed over an unbound name and raised NameError the moment they ran,
# silently, because the callers swallowed exceptions. Previews never appeared.
import os
import queue
import threading
import time
import traceback

from . import config, db
from .jobs import Job, JobStore, JobStatus

log = logging.getLogger("trellis.worker")

# Global FIFO of job ids awaiting processing.
job_queue: "queue.Queue[str]" = queue.Queue()

# Lazy pipeline singleton, guarded by a lock so a warm-up thread and the
# worker thread can't both trigger a load.
_pipeline = None
_pipeline_lock = threading.Lock()

# True once the LoRA adapter has been wrapped around the 512 flow model and its
# weights loaded. False means every job runs as "base" regardless of what was
# requested -- see _attach_adapter.
_adapter_ready = False


def is_pipeline_loaded() -> bool:
    return _pipeline is not None


def adapter_available() -> bool:
    return _adapter_ready


def queue_depth() -> int:
    return job_queue.qsize()


def get_or_load_pipeline():
    """Return the resident pipeline, loading it once if needed (thread-safe)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            from trellis_core.pipeline import load_pipeline

            log.info("Loading pipeline %s on %s ...", config.MODEL_ID, config.DEVICE)
            t0 = time.time()
            _pipeline = load_pipeline(config.MODEL_ID, device=config.DEVICE)
            log.info("Pipeline loaded in %.0fs", time.time() - t0)
            _attach_adapter(_pipeline)
    return _pipeline


def _attach_adapter(pipeline) -> None:
    """Wrap the 512 shape-SLat flow model in LoRA and load the fine-tuned weights.

    Done once, at load. The adapter is left DISABLED here; each job turns it on
    or off in _run_job. Because the low-rank branch is additive and B is
    zero-initialised at construction, "disabled" is the stock model exactly, not
    an approximation of it -- which is what makes offering "base" honest.

    Only shape_slat_flow_model_512 is wrapped. The 1024 cascade model has
    identical layer shapes and would swallow these weights silently, but it was
    never trained with them.

    A missing or unreadable adapter must NOT stop the server from booting: a lab
    machine serving the untrained model is a far better failure than a lab
    machine serving nothing. It degrades to base and says so loudly.
    """
    global _adapter_ready
    try:
        import torch
        from trellis_core import lora

        if not config.ADAPTER_PATH.exists():
            log.warning("No adapter at %s -- every job will run the BASE model.",
                        config.ADAPTER_PATH)
            return

        flow = pipeline.models["shape_slat_flow_model_512"]
        adapted = lora.apply_lora(flow, rank=config.ADAPTER_RANK,
                                  alpha=config.ADAPTER_ALPHA)
        state = torch.load(config.ADAPTER_PATH, map_location="cpu")
        loaded = lora.load_lora_state_dict(flow, state["lora"])
        lora.set_lora_enabled(flow, False)

        if loaded != len(state["lora"]):
            log.warning("Adapter mismatch: %d/%d tensors matched a module. "
                        "Falling back to BASE.", loaded, len(state["lora"]))
            lora.set_lora_enabled(flow, False)
            return

        _adapter_ready = True
        log.info("Adapter loaded: %s (%d modules, %d tensors, step %s)",
                 config.ADAPTER_PATH.name, len(adapted), loaded,
                 state.get("step", "?"))
    except Exception:
        log.exception("Could not attach the LoRA adapter -- serving BASE only.")


def _set_variant(pipeline, variant: str) -> str:
    """Enable or disable the adapter for the job about to run.

    Mutating shared module state per job is only safe because worker_loop
    processes exactly one job at a time on a single thread. If that ever becomes
    concurrent, this needs a lock or a per-request copy of the model.

    Returns the variant actually used, which is "base" whenever the adapter is
    unavailable regardless of what was asked for.
    """
    from trellis_core import lora

    if not _adapter_ready:
        return "base"
    flow = pipeline.models["shape_slat_flow_model_512"]
    lora.set_lora_enabled(flow, variant == "tuned")
    return variant


class LiveNotes(list):
    """A notes list that reports each entry the moment it is appended.

    print-prep already narrates itself into a `notes` list, but only handed it
    over once the whole stage had finished -- so the repair ladder climbed in
    silence and the UI could say nothing more useful than "finalizing".
    Subclassing list keeps every existing `notes.append(...)` call site working
    untouched.
    """

    def __init__(self, on_note):
        super().__init__()
        self._on_note = on_note

    def append(self, item):
        super().append(item)
        try:
            self._on_note(item)
        except Exception:
            log.debug("note callback failed", exc_info=True)


# Escalation notes are tagged "[rung N/5]" by printable_v2 so the UI can show
# which repair is being attempted without parsing prose. Rung 1 (accept the mesh
# as-is) emits nothing, because the common case is that it simply works.
_RUNG_RE = re.compile(r"^\[rung (\d+)/(\d+)\]\s*(.*)", re.S)


def _run_printable(pipeline: str, *, glb_path: str, output_prefix: str, notes=None):
    """Run the requested print-prep pipeline; degrade to v1 rather than fail.

    v2/v3 need optional extras (pymeshlab, manifold3d, meshlib). If they are not
    installed on this machine, a job must still produce a printable file -- so an
    ImportError falls back to v1, which needs only trimesh. Any OTHER exception
    propagates: a genuine geometry failure should surface as a job error, not be
    silently papered over with a worse result.
    """
    if pipeline in ("v2", "v3"):
        try:
            from trellis_core.printprep.printable_v2 import run_make_printable_v2

            return run_make_printable_v2(
                glb_path=glb_path,
                output_prefix=output_prefix,
                target_faces=config.PRINTABLE_TARGET_FACES,
                overhang_angle=config.PRINTABLE_OVERHANG_ANGLE,
                solid_infill=config.PRINTABLE_SOLID_INFILL,
                repair_backend="meshlib" if pipeline == "v3" else "pymeshlab",
                notes=notes,
            )
        except ImportError as e:
            log.warning(
                "Print-prep %s unavailable (%s); falling back to v1. "
                "Install the extras with: pip install -e \".[postproc-v2]\"", pipeline, e,
            )

    from trellis_core.printprep.printable import run_make_printable

    return run_make_printable(
        glb_path=glb_path,
        output_prefix=output_prefix,
        target_faces=config.PRINTABLE_TARGET_FACES,
        overhang_angle=config.PRINTABLE_OVERHANG_ANGLE,
        solid_infill=config.PRINTABLE_SOLID_INFILL,
    )


def _run_job(store: JobStore, job_id: str) -> None:
    """Run one job through all stages. May raise -- worker_loop catches it."""
    # Imports kept local so this module is importable even before torch exists,
    # and so warm-up failures surface at the right stage.
    from PIL import Image as PILImage
    from trellis_core.pipeline import run_generation, WatchdogError

    job: Job = store.get(job_id)
    params = job.params
    output_dir = job.output_dir
    glb_path = f"{output_dir}/model.glb"
    obj_path = f"{output_dir}/model.obj"

    # --- Stage: LOADING_MODEL (only visible if not yet warm) ---------------
    if not is_pipeline_loaded():
        store.update(job_id, status=JobStatus.LOADING_MODEL)
        t0 = time.time()
        pipeline = get_or_load_pipeline()
        _record_timing(store, job_id, "load_model", time.time() - t0)
    else:
        pipeline = get_or_load_pipeline()

    # --- Stage: PREPROCESSING ---------------------------------------------
    store.update(job_id, status=JobStatus.PREPROCESSING)
    t0 = time.time()
    try:
        img = PILImage.open(job.input_image_path)
        img.load()  # force decode now so a corrupt image fails here, cleanly
        img = img.convert("RGBA")
    except Exception as e:
        # Corrupt / unreadable image: record a clean, specific error and stop
        # (this is a normal user error, not a server bug).
        store.update(
            job_id,
            status=JobStatus.ERROR,
            error={
                "type": "exception",
                "message": f"Could not read the uploaded image: {e}",
                "hint": "Please upload a valid image file (JPEG, PNG, etc.).",
                "traceback": traceback.format_exc(),
            },
        )
        return
    _record_timing(store, job_id, "preprocess", time.time() - t0)

    # --- Stage: GENERATING -------------------------------------------------
    # Pick the weights first: this flips the LoRA adapter on the already-loaded
    # flow model, so it must happen before run_generation, not inside it.
    variant = _set_variant(pipeline, params.get("model_variant", config.MODEL_VARIANT))
    if variant != params.get("model_variant", config.MODEL_VARIANT):
        log.warning("Job %s asked for %r but the adapter is unavailable; ran base.",
                    job_id, params.get("model_variant"))
    store.update(job_id, params={**params, "model_variant_used": variant})
    log.info("Job %s generating with the %s model", job_id, variant)

    store.update(job_id, status=JobStatus.GENERATING, progress=None, previews=[])
    t0 = time.time()

    # Real progress + in-flight shape previews. The callbacks run on this
    # thread, inside the sampling loop, so they must be cheap and must never
    # raise -- progress.instrumented already swallows callback failures, and
    # store.update is a dict write under a lock.
    from trellis_core import progress as progress_mod

    def on_step(stage, step, total, overall):
        store.update(job_id, progress={
            "stage": stage, "step": step, "total": total, "overall": overall,
        })

    def _add_preview(name):
        job_now = store.get(job_id)
        if name not in job_now.previews:
            store.update(job_id, previews=list(job_now.previews) + [name])

    def on_preview(path, step):
        _add_preview(os.path.basename(path))

    def on_structure(coords, grid):
        name = "preview_structure.glb"
        if progress_mod.write_voxel_preview(coords, os.path.join(output_dir, name), grid=grid):
            _add_preview(name)

    # How many sampler progress bars this pipeline_type will produce. Measured,
    # not assumed: a 1024_cascade run reports three (structure, then the shape
    # sampler twice). Used only to keep the overall fraction monotonic.
    ptype = params.get("pipeline_type", config.DEFAULT_PIPELINE_TYPE)
    expected_bars = 3 if "cascade" in ptype else 2

    try:
        with progress_mod.instrumented(
            pipeline,
            on_step=on_step,
            on_preview=on_preview,
            on_structure=on_structure,
            expected_bars=expected_bars,
            preview_at=config.PREVIEW_AT,
            preview_resolution=config.PREVIEW_RESOLUTION,
            preview_dir=output_dir,
        ):
            gen = run_generation(
                pipeline,
                img,
                seed=int(params.get("seed", config.DEFAULT_SEED)),
                pipeline_type=params.get("pipeline_type", config.DEFAULT_PIPELINE_TYPE),
                target_faces=int(params.get("target_faces", config.DEFAULT_TARGET_FACES)),
                texture_size=int(params.get("texture_size", config.DEFAULT_TEXTURE_SIZE)),
                no_texture=config.NO_TEXTURE,
                skip_texture=bool(params.get("skip_texture", config.SKIP_TEXTURE_BY_DEFAULT)),
                out_glb_path=glb_path,
                out_obj_path=obj_path,
            )
    except WatchdogError as e:
        # The known macOS GPU-watchdog signature. Its str() IS the full,
        # multi-line, numbered workaround text the CLI prints; surface it
        # verbatim as the hint so the user gets actionable guidance, not a
        # stack trace. Caught here, BEFORE the generic handler in worker_loop.
        store.update(
            job_id,
            status=JobStatus.ERROR,
            error={
                "type": "watchdog",
                "message": "Generation failed: the GPU watchdog likely killed the decoder.",
                "hint": str(e),
                "traceback": None,
            },
        )
        return
    _record_timing(store, job_id, "generation", time.time() - t0)

    # Track which output files actually exist, in a stable display order.
    file_specs = [
        ("model_glb", "Geometry (GLB)", "model.glb"),
        ("model_obj", "Geometry (OBJ)", "model.obj"),
    ]

    # --- Stage: MAKING_PRINTABLE (unless skipped) --------------------------
    diagnostics = None
    fidelity = None
    if not params.get("skip_printable"):
        store.update(job_id, status=JobStatus.MAKING_PRINTABLE, repair_log=[],
                     progress={"stage": "repair", "step": 1, "total": 5, "overall": 0.0,
                               "detail": "Checking whether the mesh is already a valid solid"})
        t0 = time.time()

        def on_note(text):
            """Narrate the repair ladder as it climbs.

            Escalation notes carry a "[rung N/5]" tag; everything else is
            progress chatter from inside a rung. Both go to the feed, but only a
            tagged note advances the bar -- otherwise a chatty rung would look
            like progress it has not made.
            """
            job_now = store.get(job_id)
            fields = {"repair_log": (list(job_now.repair_log) + [text])[-40:]}
            m = _RUNG_RE.match(text)
            if m:
                rung, total = int(m.group(1)), int(m.group(2))
                fields["progress"] = {
                    "stage": "repair", "step": rung, "total": total,
                    "overall": (rung - 1) / total, "detail": m.group(3),
                }
                log.info("Job %s repair rung %d/%d", job_id, rung, total)
            store.update(job_id, **fields)

        repair_notes = LiveNotes(on_note)
        printable = _run_printable(
            params.get("printable_pipeline") or config.PRINTABLE_PIPELINE,
            glb_path=glb_path,
            output_prefix=f"{output_dir}/model_printable",
            notes=repair_notes,
        )
        _record_timing(store, job_id, "make_printable", time.time() - t0)
        diagnostics = printable.diagnostics
        fidelity = printable.fidelity
        file_specs.append(("printable_glb", "Print-ready (GLB)", "model_printable.glb"))
        file_specs.append(("printable_stl", "Print-ready (STL)", "model_printable.stl"))

    # --- Assemble result (only files that were actually written) -----------
    files = [
        {"kind": kind, "label": label, "filename": name}
        for kind, label, name in file_specs
        if os.path.exists(os.path.join(output_dir, name))
    ]
    # Prefer the printable glb for preview; fall back to the raw glb.
    preview = next((f["filename"] for f in files if f["kind"] == "printable_glb"), None)
    if preview is None:
        preview = next((f["filename"] for f in files if f["kind"] == "model_glb"), None)

    result = {
        "files": files,
        "preview_filename": preview,
        "mesh_stats": {"vertices": gen.vertex_count, "faces": gen.face_count},
        "used_metal_bake": gen.used_metal_bake,
        "watertight": getattr(printable, "watertight", None) if not params.get("skip_printable") else None,
    }

    job = store.update(
        job_id,
        status=JobStatus.DONE,
        result=result,
        diagnostics=diagnostics,
        fidelity=fidelity,
    )

    # Persist to the shared, on-disk archive (Library) so completed models
    # survive restarts. Wrapped so a DB hiccup can never break generation --
    # the worker's core invariant is that a finished job stays finished.
    try:
        title = os.path.splitext(os.path.basename(job.input_image_path))[0] or "model"
        db.upsert_model(
            job_id=job_id,
            operator=getattr(job, "operator", "anonymous"),
            title=title,
            created_at=job.created_at,
            duration_seconds=round(job.updated_at - job.created_at, 1),
            params=params,
            vertices=gen.vertex_count,
            faces=gen.face_count,
            watertight=result.get("watertight"),
            files=files,
            preview_filename=preview,
            status="done",
        )
    except Exception:  # noqa: BLE001
        log.exception("Failed to persist model %s to the library DB", job_id)


def _record_timing(store: JobStore, job_id: str, key: str, seconds: float) -> None:
    job = store.get(job_id)
    timings = dict(job.timings)
    timings[key] = round(seconds, 2)
    store.update(job_id, timings=timings)


def _failure_hint(store: JobStore, job_id: str, exc: Exception) -> str:
    """Explain a failure using what the repair ladder actually recorded.

    The one failure this pipeline produces in practice is a mesh no repair could
    close, and by then the ladder has already written down every method it tried
    and why each was abandoned. Replaying that is far more use than a generic
    apology -- and it tells the user the one thing they can act on: this input
    generated geometry too broken to seal, so try another photo or angle.
    """
    generic = "An unexpected error occurred while processing this job."
    try:
        job = store.get(job_id)
    except Exception:
        return generic

    # Keep each note's OWN rung number. Renumbering them sequentially would
    # relabel "rung 5 of 5" as step 3 and quietly misreport how far the ladder
    # actually got -- which is the single most useful fact in this message.
    steps = []
    for note in job.repair_log:
        m = _RUNG_RE.match(note)
        if m:
            steps.append((int(m.group(1)), int(m.group(2)), m.group(3).strip()))

    if "Manifold3D rejected" in str(exc) or steps:
        lines = [
            "The mesh could not be turned into a closed, printable solid.",
            "",
            "Every repair in the pipeline was tried, in order:",
        ]
        lines += [f"  {r}/{t}  {text}" for r, t, text in steps] or \
                 ["  (the ladder recorded no escalation)"]
        lines += [
            "",
            "This normally means the generated geometry was unusually broken -- "
            "thin or hollow shapes are the common cause. A different photo, or a "
            "less side-on angle, usually generates a mesh that seals cleanly.",
            "",
            "The raw (unrepaired) geometry was still produced and can be "
            "downloaded, but it is not watertight and will not slice as-is.",
        ]
        return "\n".join(lines)
    return generic


def worker_loop(store: JobStore) -> None:
    """Forever pull job ids and run them. A crash in one job is caught here so
    that the worker thread itself never dies -- the next queued job still runs.
    """
    log.info("Worker loop started.")
    while True:
        job_id = job_queue.get()
        try:
            _run_job(store, job_id)
        except Exception as e:  # noqa: BLE001 -- intentional blanket guard
            # This is the critical invariant: NO exception from _run_job may
            # escape this loop. Record it on the job and move on.
            log.exception("Job %s failed", job_id)
            try:
                store.update(
                    job_id,
                    status=JobStatus.ERROR,
                    error={
                        "type": "exception",
                        "message": f"{type(e).__name__}: {e}",
                        # "An unexpected error occurred" tells the user nothing they
                        # can act on. When print-prep is what failed, the repair log
                        # holds the actual story -- which repairs were tried, why each
                        # was abandoned -- so hand that over instead of a shrug.
                        "hint": _failure_hint(store, job_id, e),
                        "traceback": traceback.format_exc(),
                    },
                )
            except Exception:  # noqa: BLE001 -- even bookkeeping must not kill the loop
                log.exception("Failed to record error for job %s", job_id)
        finally:
            job_queue.task_done()


def start_worker(store: JobStore) -> threading.Thread:
    """Spawn the single daemon worker thread."""
    thread = threading.Thread(target=worker_loop, args=(store,), name="trellis-worker", daemon=True)
    thread.start()
    return thread


def warm_pipeline_async() -> threading.Thread:
    """Eagerly load the pipeline in the background so the ~100s cold-start cost
    is paid at boot, not on the first visitor. Failures are logged, not fatal.
    """

    def _warm():
        try:
            get_or_load_pipeline()
        except Exception:  # noqa: BLE001
            log.exception("Background pipeline warm-up failed")

    thread = threading.Thread(target=_warm, name="trellis-warmup", daemon=True)
    thread.start()
    return thread
