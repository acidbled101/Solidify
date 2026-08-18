"""This week's report: same document, same style, new abstract and new sections.

Reuses build_weekly_report's renderer and tex_from_content's emitter rather
than copying either, so the two reports cannot drift apart in styling: the
logos, fonts, header banner, section formatting and table styling all come from
the same code that produced the previous one.

    python report/build_weekly_report_w2.py       -> PDF + .tex

PLACEHOLDERS. Text in [[double brackets]] marks a fact only Ali can supply --
the clay printer's make and model, the workshop details, which geometries the
G-code produced, and what the slicing interface is called. They are deliberately
loud so they cannot be missed on a read-through. Search the PDF for "[[".
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_weekly_report as bwr
from tex_from_content import emit


ABSTRACT = """This week I continued the work on fine-tuning the model using LoRA, this time using a dataset of 900 samples of SLats instead of the 300 used in the previous run, and raising the amount of training to six full passes over that data. Alongside the training work, we held a workshop in the FabLab about 3D clay printing. I helped the FabLab staff run a special clay printer, and I wrote G-code by hand to create different geometries on it. I also made a web interface that can slice 3D meshes based on the specific requirements of the clay printer, since clay does not tolerate the supports, overhangs and thin walls that a normal plastic slicer will happily produce. Finally, I discussed the possibility of further work on converting the model's outputs into clay-printable results, which is a different and stricter target than the plastic printing the validation layer was built for. This report covers that work, together with the fine-tuning, post-processing and web deployment work from the previous weeks, and the background material on the TRELLIS pipeline and on flow and diffusion models."""


NEW = [
 # ---------------------------------------------------------------- 8.3-8.5
 ("h2","8.3","Scaling the Dataset to 900 Samples"),
 ("p","""This week I continued the fine-tuning with a larger dataset. The previous run used 300 curated models; this one uses 900 samples of SLats, encoded from Thingi10K models filtered the same way as before to clean, closed, manifold meshes. Of the 900, 870 are used for training and 30 are held out so that the model is always measured on objects it has not been trained on."""),
 ("p","""The other change is the amount of training rather than the amount of data. The run makes six full passes over the 900 samples, which is 5,220 training steps, where the previous larger attempt had made less than one pass before it was stopped. That distinction turned out to matter: when the 300-sample model was compared against a model trained on a larger dataset but for a shorter time, the 300-sample model won, and the most likely reason is not the data but the number of passes over it - 8.7 passes against 0.78. Six passes over 900 samples is the first run in this project that exceeds the exposure of the model that currently performs best, which is what makes it a fair test of whether more data continues to help."""),
 ("p","""The setup is otherwise unchanged from the previous run so that the comparison means something: LoRA at rank 16 applied to the attention and MLP projections of the flow model's transformer blocks, with the base weights frozen, AdamW, a learning rate that warms up and then decays on a cosine schedule, and conditioning dropout so that the unconditional branch the sampler needs does not drift. At the time of writing this report the run is still in progress, so this section reports the setup rather than the result; the measured numbers will follow in the next report."""),
 ("p","""A smaller control run was completed first, using 100 samples for the same six passes. Its held-out loss stayed flat across all six passes, moving by less than a quarter of a percent from beginning to end while the training loss fell by a factor of four. That is what fitting a training set too small to generalize from looks like, and it is the reason for moving to 900 rather than continuing at 100."""),

 # ------------------------------------------------------------------ clay
 ("h1","9","FabLab Clay Printing Workshop"),
 ("h2","9.1","The Workshop and the Clay Printer"),
 ("p","""This week the FabLab held a workshop about 3D clay printing, and I helped the FabLab staff run the special clay printer used for it. [[FILL: printer make and model, the date of the workshop, and who attended - staff, students, public]]"""),
 ("p","""Clay printing is the same idea as the plastic printing assumed everywhere else in this report, but almost none of the practical constraints carry over. Instead of melting a filament and letting it solidify on contact, the machine extrudes a soft paste through a wide nozzle from a pressurised cartridge, and the material stays soft after it is placed. Everything that makes clay difficult follows from that. The material cannot bridge a gap, so an overhang has to be self-supporting or it will sag. Support structures are not a real option, because a support printed in clay is as soft as the part it is supporting and cannot be cleanly removed afterwards. The nozzle is much wider than a plastic nozzle, so layers are thick and walls cannot be thin. Extrusion cannot be started and stopped cleanly the way a plastic printer retracts, so a toolpath that jumps around leaves marks and blobs, and a continuous spiral path is usually preferred. And because the whole part is still wet while it is being built, the lower layers carry the weight of everything above them, which limits how tall and how fast a piece can be printed."""),

 ("h2","9.2","Writing G-code for Test Geometries"),
 ("p","""I wrote G-code to create different geometries on the printer. [[FILL: which geometries - for example cylinders, cones, twisted vases, lattice or wave-patterned walls - and what each was testing]]"""),
 ("p","""Writing the toolpath directly, rather than going through a general-purpose slicer, is a reasonable way to work with a clay printer. The machine's behaviour depends on the extrusion rate being matched to the speed the head is moving at, and on the path being continuous, and both of those are easier to control when the path is generated deliberately than when it is the by-product of a slicer built for a different process. Hand-written G-code also makes it straightforward to produce the kind of geometry clay is actually good at: shapes generated as a single continuous spiral, where the wall thickness and the rate of change of radius are chosen so that each layer is supported by the one beneath it."""),
 ("fig",["fig5_clay_table.jpg","fig5_clay_piece.jpg"],[74,74],5,"Pieces printed during the workshop. Left: the workbench, with each print left on its own laser-cut plywood bat so that it can be moved off the machine without being handled, and the clay mixer and the printer behind. The forms are cylinders, tapered cones and vessels of varying profile, printed as single continuous walls. Right: a close-up of one piece. The layer lines run as one unbroken spiral rather than as separate closed loops, and the lobed profile is produced by varying the radius as the spiral climbs, which is the kind of shape clay handles well because every layer stays supported by the one beneath it. The ragged top rim is where extrusion stopped."),

 ("h2","9.3","A Web Interface for Slicing to the Clay Printer"),
 ("p","""I also made a web interface that can slice 3D meshes based on the specific requirements of the clay printer. The motivation is the same as for the SOLIDIFY front end described in Section 5: the knowledge of what the machine will and will not accept should live in the tool, not in the person operating it. A general slicer will happily generate supports, thin walls, sharp overhangs and a toolpath full of retractions, all of which are correct for plastic and wrong for clay, and it takes experience with the machine to know which of its settings to override."""),
 ("p","""The interface takes a mesh, applies the constraints the clay printer needs, and produces a toolpath the machine can run. [[FILL: what the interface is called, what it is built with, which parameters it exposes - nozzle diameter, layer height, extrusion rate, maximum overhang angle, minimum wall thickness, spiral or shelled mode - and whether it previews the toolpath before printing]]"""),
 ("p","""[[FILL: add a screenshot of the interface here. Drop the image into report/weekly_figs/ and it can be placed the same way as Figure 5.]]"""),

 ("h2","9.4","From Model Output to Clay-Printable Results"),
 ("p","""Finally, I discussed the possibility of further work on converting the model's outputs into clay-printable results, so that the pipeline described in this report could end at the clay printer rather than at a plastic one."""),
 ("p","""This is a harder target than plastic, and the difference is instructive. For plastic, the validation layer's job is almost entirely repair: make the mesh closed and manifold so that the slicer will accept it, and the printer will handle the rest, using supports where the geometry needs them. For clay, a mesh can be perfectly closed and manifold and still be impossible to print, because the constraints are about the shape itself rather than about the correctness of the surface. Overhang angle, wall thickness and how much unsupported weight sits above a given layer are all properties of the geometry the model chose to generate, and no amount of repair after the fact will change them."""),
 ("p","""That is exactly the kind of property fine-tuning can reach, which is what makes this worth pursuing rather than treating as a separate problem. The scoring function already used to evaluate the fine-tuned models measures an overhang penalty and a thickness penalty alongside the mesh-quality metrics, because those were the axes expected to respond to training in the first place. A clay-specific version of that scoring, with the thresholds set by what the machine can actually do, would give a direct way to measure whether a model's raw output is clay-printable, and therefore a direct target to fine-tune towards. Whether the improvement would be large enough to matter is an open question, and answering it needs the clay constraints written down as numbers first. [[FILL: the printer's actual limits - nozzle diameter, usable overhang angle, minimum wall thickness, maximum practical height]]"""),
]


REFS_EXTRA = [
 "Clay 3D printer used in the FabLab workshop. [[FILL: manufacturer, model, and a link to its documentation]]",
 "Clay slicing web interface (internal, lab network). [[FILL: name and URL if it is deployed]]",
]


def make_content():
    """Insert this week's material into last week's document.

    Editing a copy of CONTENT rather than rewriting it keeps every earlier
    section, figure and caption exactly as it was published, which is what
    "based on the old one" has to mean if the two reports are to be comparable
    side by side.
    """
    old = list(bwr.CONTENT)

    # Where the new subsections go: immediately after Section 8's closing
    # paragraph, which is the one ending on the 300-model sample size.
    idx_after_8 = next(i for i, it in enumerate(old)
                       if it[0] == "h1" and it[1] == "9")            # old Discussion
    out = old[:idx_after_8] + NEW

    # Everything from the old Discussion onwards, renumbered by one section.
    renumber = {"9": "10", "10": "11", "11": "12"}
    for it in old[idx_after_8:]:
        if it[0] == "h1" and it[1] in renumber:
            it = ("h1", renumber[it[1]], it[2])
        elif it[0] == "refs":
            it = ("refs", list(it[1]) + REFS_EXTRA)
        out.append(it)

    # Roadmap and conclusion have to describe the document they are actually in.
    return [_retitle(it) for it in out]


ROADMAP = """Section 2 lists the sources I studied over these past weeks. Section 3 summarizes the topics themselves, one subsection per topic. Section 4 covers my initial results, including the validation layer I built to make the model's output printable, together with the printed output and photos. Section 5 covers the deployment of the pipeline as a web service for FabLab use. Sections 6 to 8 cover the model-side work: extending the post-processing layer, an attempt at preference optimization, and fine-tuning the SLat flow model with LoRA, including this week's larger 900-sample run. Section 9 covers this week's FabLab clay printing workshop, the G-code and geometries I made for the clay printer, the web interface I built to slice meshes for it, and what converting the model's outputs into clay-printable results would involve. Section 10 is a short discussion, Section 11 covers limitations and next steps, and Section 12 concludes."""

CONCLUSION = """Over the past few weeks I reviewed the background theory behind TRELLIS: the TRELLIS pipeline itself and the flow matching, guidance, VAE, and transformer-architecture material from the MIT lecture notes. I also ran the model and built a validation layer to post-process its output into a printable form, with physical prints included above, and put that pipeline behind a web front end, SOLIDIFY, hosted on the lab machine, so that a FabLab user can upload a single photograph and get back a watertight, print-ready mesh. I extended the post-processing side with manifold3d, tried and set aside a DreamDPO-based preference approach, and fine-tuned the SLat flow model with LoRA on 300 curated Thingi10K models, which gave significantly less non-manifold edges on the raw output without loss in detail. This week I continued that fine-tuning with a larger dataset of 900 SLat samples trained for six full passes. Alongside the model work, we held a FabLab workshop on 3D clay printing, where I helped the staff run the clay printer, wrote G-code to produce different geometries, and built a web interface that slices meshes to the clay printer's specific requirements. Converting the model's outputs into clay-printable results is the direction I would like to take next, since clay constrains the shape itself rather than only the correctness of the mesh, and that is a constraint fine-tuning can be aimed at directly."""


def _retitle(it):
    if it[0] == "p" and it[1].startswith("Section 2 lists the sources"):
        return ("p", ROADMAP)
    if it[0] == "p" and it[1].startswith("Over the past few weeks I reviewed"):
        return ("p", CONCLUSION)
    return it


if __name__ == "__main__":
    bwr.CONTENT = make_content()
    bwr.ABSTRACT = ABSTRACT

    pdf = os.path.join(HERE, "weekly_report_ali_week2.pdf")
    n = bwr.build(pdf)
    print(f"wrote {pdf}  ({n} pages)")

    tex = os.path.join(HERE, "weekly_report_ali_week2.tex")
    # The tex emitter picks figure width by figure NUMBER; figure 5 is the new
    # pair of workshop photos, two portrait shots side by side.
    figw = {1: r'\linewidth', 2: r'0.85\linewidth', 3: r'0.23\linewidth',
            4: r'\linewidth', 5: r'0.46\linewidth'}
    emit(bwr.CONTENT, ABSTRACT, tex, figw=figw)
    print(f"wrote {tex}")

    todos = [c[1] for c in bwr.CONTENT if c[0] == "p" and "[[" in c[1]]
    print(f"\n{sum(t.count('[[') for t in todos)} placeholder(s) to fill -- search for '[['")
