"""
Job model + in-memory job store with a per-job on-disk layout.

A single process holds all job state in memory (fine for personal/local use --
no database). Every job also gets a directory under `config.JOBS_DIR`:

    jobs/<job_id>/
      input/<original_filename>      uploaded image
      output/                        model.glb, model.obj,
                                     model_printable.glb, model_printable.stl
"""

import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    LOADING_MODEL = "loading_model"
    PREPROCESSING = "preprocessing"
    GENERATING = "generating"
    MAKING_PRINTABLE = "making_printable"
    DONE = "done"
    ERROR = "error"


# Human-readable text shown in the frontend for each stage.
STATUS_TEXT = {
    JobStatus.QUEUED: "Queued -- waiting for the worker...",
    JobStatus.LOADING_MODEL: "Loading the TRELLIS.2 model (one-time, ~100s)...",
    JobStatus.PREPROCESSING: "Preparing the uploaded image...",
    JobStatus.GENERATING: "Generating 3D mesh (typically 3-5 min)...",
    JobStatus.MAKING_PRINTABLE: "Repairing and preparing the mesh for printing...",
    JobStatus.DONE: "Done.",
    JobStatus.ERROR: "Failed.",
}


@dataclass
class Job:
    id: str
    status: JobStatus
    created_at: float
    updated_at: float
    params: dict
    input_image_path: str
    output_dir: str
    operator: str = "anonymous"
    result: Optional[dict] = None
    diagnostics: Optional[dict] = None
    fidelity: Optional[dict] = None
    error: Optional[dict] = None
    timings: dict = field(default_factory=dict)
    # Live sampler position: {"stage", "step", "total", "frac"}. Written from
    # the generation thread on every sampling step, read by the polling UI, so
    # the progress bar reflects where the model actually is rather than a curve
    # fitted to elapsed time.
    progress: Optional[dict] = None
    # Basenames of coarse meshes decoded mid-generation, oldest first. These are
    # also the allowlist for serving them (see get_job_file).
    previews: list = field(default_factory=list)

    def status_text(self) -> str:
        return STATUS_TEXT.get(self.status, str(self.status))

    def elapsed_seconds(self) -> float:
        end = self.updated_at if self.status in (JobStatus.DONE, JobStatus.ERROR) else time.time()
        return round(end - self.created_at, 1)

    def to_public_dict(self, debug: bool = False) -> dict:
        """Serialize for the API. Tracebacks in `error` are hidden unless debug."""
        error = None
        if self.error is not None:
            error = {
                "type": self.error.get("type"),
                "message": self.error.get("message"),
                "hint": self.error.get("hint"),
            }
            if debug and self.error.get("traceback"):
                error["traceback"] = self.error["traceback"]

        return {
            "job_id": self.id,
            "status": self.status.value,
            "status_text": self.status_text(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed_seconds": self.elapsed_seconds(),
            "timings": self.timings,
            "progress": self.progress,
            "previews": list(self.previews),
            "params": self.params,
            "result": self.result,
            "diagnostics": self.diagnostics,
            "fidelity": self.fidelity,
            "error": error,
        }


class JobStore:
    """Thread-safe in-memory store of jobs, keyed by job id."""

    def __init__(self, jobs_dir: Optional[Path] = None):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._jobs_dir = Path(jobs_dir) if jobs_dir is not None else config.JOBS_DIR

    def create(
        self, params: dict, image_bytes: bytes, filename: str, operator: str = "anonymous"
    ) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = self._jobs_dir / job_id
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(filename or "upload").name or "upload"
        input_path = input_dir / safe_name
        input_path.write_bytes(image_bytes)

        now = time.time()
        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            params=dict(params),
            input_image_path=str(input_path),
            output_dir=str(output_dir),
            operator=operator,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        """Return the job or raise KeyError (the API turns that into a 404)."""
        with self._lock:
            return self._jobs[job_id]

    def update(self, job_id: str, **fields) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()
            return job

    def list_recent(self, n: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:n]
