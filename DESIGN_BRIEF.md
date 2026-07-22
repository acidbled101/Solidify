# Design Brief — Image-to-3D Web App

> A plain description of **what this product is and what it does**, for use as
> input to a design tool. No visual/design details here (no colors, fonts, or
> layout) — just the idea, the users, the flow, and the features.

## The idea in one line

A website where you upload a single photo of an object and get back a
**ready-to-3D-print model** you can download — with all the heavy AI computation
running on a lab machine, not on the user's device.

## What it is

An **interface** (remote control) for an image-to-3D AI model. The user just
brings a photo; the system turns that photo into a watertight, print-ready 3D
model (STL/GLB) and shows a live 3D preview before download. The user never
installs anything and never configures anything — they open a link, log in, and
use it from any device (laptop, phone, tablet), anywhere.

**Important:** the actual 3D generation happens on a single lab machine (a Mac
with a GPU). The website is only the interface to it. One job is processed at a
time, and a job takes a few minutes.

## The problem it solves

Turning a real-world object into a printable 3D model normally requires
specialized software, technical setup, and know-how. This makes it as simple as
uploading a photo — so non-technical users (e.g. professors) can produce
print-ready files themselves without any tools or training.

## Who it's for

- **Primary users:** professors at the lab who want to create 3D-printable files
  from photos, quickly, from their own devices.
- **Possibly also:** students and other lab members (to be decided).
- **Admin/owner:** manages who is allowed to log in.

Users are assumed to be **non-technical**. If the experience feels like work or
requires setup, they won't use it. Simplicity and clarity are the whole point.

## The core flow (happy path)

1. **Log in** — access is restricted to approved users.
2. **Upload a photo** of an object.
3. *(Optional)* adjust a few advanced settings; otherwise use sensible defaults.
4. **Start generation** — the request is queued and sent to the lab machine.
5. **Watch progress** — the interface shows the job moving through stages
   (queued → preparing → generating → finalizing) with a sense of how far along
   it is. A job takes a few minutes.
6. **See the result** — an interactive 3D preview of the finished model appears
   (rotate/zoom), along with basic model info.
7. **Download** the print-ready file(s).
8. **Start another** whenever they want.

## Features

### Core (already exist in the app)
- **Photo upload** with basic validation (must be a real image; size limit).
- **Generate a 3D model** from the uploaded photo.
- **Print-prep step** that makes the model watertight and printable (can be
  skipped for geometry-only output).
- **Advanced options** (all optional, with defaults): random seed, mesh detail /
  target face count, generation quality/pipeline level, and a "skip print-prep"
  toggle.
- **Live progress** through named stages while the job runs.
- **Interactive 3D preview** of the finished model (rotate, zoom).
- **Downloads** of the output files (print-ready model + raw geometry).
- **Model info** after completion (e.g. vertex/face counts).
- **Print diagnostics** (e.g. whether it's watertight, overhang/thin-wall notes).
- **Clear error handling** — if something fails, show a friendly explanation and
  a one-click "retry with the same settings."
- **Queue awareness** — because only one job runs at a time, the interface should
  communicate waiting/position and reject overload gracefully.

### Planned (to be built)
- **Login / accounts** — a proper login screen, restricted to an approved list of
  users that the owner manages (add/remove people). Replaces a temporary shared
  password.
- **(Ideas, not committed):** history/gallery of a user's past generations;
  clearer print instructions or hand-off to a slicer; multi-language support
  (e.g. English/Arabic) — all TBD with the owner.

## Key states the interface needs to cover

- **Login** (and "wrong credentials").
- **Empty / ready to upload.**
- **Uploaded, ready to generate** (with the optional advanced settings).
- **Working / in progress** (staged, multi-minute).
- **Success** (3D preview + info + downloads).
- **Error** (clear message + retry).
- **Busy / queued** (someone else's job is running).
- **Warming up** (the model takes ~100s to load the first time after the server
  starts; the first job may wait for it).

## Constraints & principles

- **Zero install, zero config for the user** — just a link and a login.
- **Works from any device**, anywhere (accessed over the internet via a secure
  tunnel to the lab machine).
- **All computation is on the lab machine**; the website only sends the image and
  displays the result.
- **One job at a time**, each taking a few minutes — the design should make
  waiting feel informative and calm, not broken.
- **Non-technical audience** — favor clarity and simplicity over showing every
  knob.
