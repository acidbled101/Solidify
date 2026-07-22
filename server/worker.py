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
import queue
import threading
import time
import traceback

from . import config
from .jobs import Job, JobStore, JobStatus

log = logging.getLogger("trellis.worker")

# Global FIFO of job ids awaiting processing.
job_queue: "queue.Queue[str]" = queue.Queue()

# Lazy pipeline singleton, guarded by a lock so a warm-up thread and the
# worker thread can't both trigger a load.
_pipeline = None
_pipeline_lock = threading.Lock()


def is_pipeline_loaded() -> bool:
    return _pipeline is not None


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
    return _pipeline


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
    store.update(job_id, status=JobStatus.GENERATING)
    t0 = time.time()
    try:
        gen = run_generation(
            pipeline,
            img,
            seed=int(params.get("seed", config.DEFAULT_SEED)),
            pipeline_type=params.get("pipeline_type", config.DEFAULT_PIPELINE_TYPE),
            target_faces=int(params.get("target_faces", config.DEFAULT_TARGET_FACES)),
            texture_size=int(params.get("texture_size", config.DEFAULT_TEXTURE_SIZE)),
            no_texture=config.NO_TEXTURE,
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
        from trellis_core.printable import run_make_printable

        store.update(job_id, status=JobStatus.MAKING_PRINTABLE)
        t0 = time.time()
        printable = run_make_printable(
            glb_path=glb_path,
            output_prefix=f"{output_dir}/model_printable",
            target_faces=config.PRINTABLE_TARGET_FACES,
            overhang_angle=config.PRINTABLE_OVERHANG_ANGLE,
            solid_infill=config.PRINTABLE_SOLID_INFILL,
        )
        _record_timing(store, job_id, "make_printable", time.time() - t0)
        diagnostics = printable.diagnostics
        fidelity = printable.fidelity
        file_specs.append(("printable_glb", "Print-ready (GLB)", "model_printable.glb"))
        file_specs.append(("printable_stl", "Print-ready (STL)", "model_printable.stl"))

    # --- Assemble result (only files that were actually written) -----------
    import os

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

    store.update(
        job_id,
        status=JobStatus.DONE,
        result=result,
        diagnostics=diagnostics,
        fidelity=fidelity,
    )


def _record_timing(store: JobStore, job_id: str, key: str, seconds: float) -> None:
    job = store.get(job_id)
    timings = dict(job.timings)
    timings[key] = round(seconds, 2)
    store.update(job_id, timings=timings)


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
                        "hint": "An unexpected error occurred while processing this job.",
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
