"""Render the weekly report to PDF without a TeX installation.

There is no LaTeX on this machine (no pdflatex/xelatex/lualatex/tlmgr, and
installing BasicTeX needs sudo), so the .tex source cannot be compiled here.
This reproduces the same document with fpdf2: the two logos on the title page,
the KAUST Academy banner as a running header, the section structure, figures
and captions from the original, and the new sections.

Font substitutions, since the originals are not installed:
    TeX Gyre Pagella -> Times New Roman   (both are Palatino/old-style serifs)
    Carlito          -> Helvetica         (both are Calibri/Helvetica-class sans)
    DejaVu Sans Mono -> Courier
The .tex source is the authoritative version; compile it with xelatex where a
TeX installation is available to get the exact fonts.
"""

import os
from fpdf import FPDF

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "weekly_figs")
SUP = "/System/Library/Fonts/Supplemental"

GOOD = (21, 122, 60)
MUTED = (90, 90, 90)


class Report(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        # PDF core fonts: Times is one of the 14 standard faces, so it needs
        # no embedding. Embedding the Times New Roman TTFs cost ~90KB and the
        # rendered result is visually the same face.
        self.set_auto_page_break(True, margin=22)
        self.set_margins(23, 24, 23)
        self.title_page = True
        self.toc_entries = []

    def header(self):
        if self.title_page:
            return
        self.image(f"{FIG}/header_banner.png", x=23, y=8, h=12)
        self.set_y(24)

    def footer(self):
        if self.title_page:
            return
        self.set_y(-14)
        self.set_font("helvetica", "", 8.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, "Weekly Report")
        self.set_x(-33)
        self.cell(20, 5, str(self.page_no() - 1), align="R")
        self.set_text_color(0, 0, 0)

    # ---- building blocks --------------------------------------------
    @property
    def W(self):
        return self.w - 46

    def h1(self, n, t):
        self.ln(4)
        self.set_font("helvetica", "B", 14)
        self.multi_cell(self.W, 7, f"{n}   {t}")
        self.toc_entries.append((0, n, t, self.page_no() - 1))
        self.ln(1.5)

    def h2(self, n, t):
        self.ln(1.5)
        self.set_font("helvetica", "B", 11)
        self.multi_cell(self.W, 5.5, f"{n}   {t}")
        self.toc_entries.append((1, n, t, self.page_no() - 1))
        self.ln(1)

    def body(self, txt, size=10.5):
        self.set_font("times", "", size)
        self.multi_cell(self.W, 5.0, txt.strip())
        self.ln(2.5)

    def eq(self, txt):
        self.ln(1)
        self.set_font("times", "I", 11)
        self.cell(self.W, 6, txt, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def caption(self, n, txt):
        self.set_font("helvetica", "B", 8.5)
        self.cell(self.get_string_width(f"Figure {n} ") + 1, 4.4, f"Figure {n}")
        self.set_font("times", "", 8.5)
        self.multi_cell(self.W - self.get_string_width(f"Figure {n} ") - 1, 4.4, " " + txt)
        self.ln(3)

    def figure(self, files, widths, cap_n, cap, gap=4):
        total = sum(widths) + gap * (len(files) - 1)
        x0 = 23 + (self.W - total) / 2
        heights = []
        from PIL import Image
        for f, wd in zip(files, widths):
            im = Image.open(os.path.join(FIG, f))
            heights.append(wd * im.size[1] / im.size[0])
        if self.get_y() + max(heights) + 18 > self.h - 24:
            self.add_page()
        y0 = self.get_y()
        x = x0
        for f, wd, ht in zip(files, widths, heights):
            self.image(os.path.join(FIG, f), x=x, y=y0, w=wd)
            x += wd + gap
        self.set_y(y0 + max(heights) + 3)
        self.caption(cap_n, cap)


def render(path, toc=None):
    """One pass. With toc=None the TOC page is skipped and entries are
    collected; with a list, the TOC is emitted right after the title page."""
    p = Report()

    # ---------------- title page ----------------
    p.add_page()
    from PIL import Image
    a = Image.open(f"{FIG}/logo_kaust_academy.jpg"); b = Image.open(f"{FIG}/logo_kaust.jpg")
    ha, hb = 22, 22
    wa = ha * a.size[0] / a.size[1]; wb = hb * b.size[0] / b.size[1]
    x = 23 + (p.W - (wa + 12 + wb)) / 2
    p.image(f"{FIG}/logo_kaust_academy.jpg", x=x, y=22, h=ha)
    p.image(f"{FIG}/logo_kaust.jpg", x=x + wa + 12, y=22, h=hb)
    p.set_y(58)

    p.set_font("helvetica", "B", 22)
    p.cell(p.W, 11, "Weekly Report", align="C", new_x="LMARGIN", new_y="NEXT"); p.ln(3)
    p.set_font("times", "", 13)
    p.cell(p.W, 6, "Ali Alhulaimi", align="C", new_x="LMARGIN", new_y="NEXT")
    p.set_font("courier", "", 9.5)
    p.cell(p.W, 6, "alialhulaimi2005@gmail.com", align="C", new_x="LMARGIN", new_y="NEXT")
    p.ln(8)

    p.set_font("helvetica", "B", 12)
    p.set_x(35); p.cell(p.W - 24, 6, "Abstract", new_x="LMARGIN", new_y="NEXT"); p.ln(1)
    p.set_font("times", "", 9.8); p.set_x(35)
    p.multi_cell(p.W - 24, 4.6, ABSTRACT.strip())
    p.ln(9)

    for k, v in [("Date:", "August 2026"),
                 ("Explainer:", "https://claude.ai/public/artifacts/c52594ab-8c5c-404c-8999-68e62b4054ee"),
                 ("Service:", "https://scifablabs-mac-mini.tailfc1a5e.ts.net/")]:
        p.set_x(35)
        p.set_font("helvetica", "B", 9.5); p.cell(20, 5, k)
        p.set_font("courier", "", 8)
        p.multi_cell(p.W - 46, 5, v)
    p.title_page = False

    # ---------------- contents ----------------
    if toc is not None:
        p.add_page()
        p.set_font("helvetica", "B", 14)
        p.multi_cell(p.W, 8, "Contents"); p.ln(2)
        for lvl, n, t, pg in toc:
            p.set_font("times", "B" if lvl == 0 else "", 10.5 if lvl == 0 else 10)
            p.set_x(23 + (6 if lvl else 0))
            p.cell(p.W - 14 - (6 if lvl else 0), 5.4, f"{n}   {t}".strip())
            p.cell(8, 5.4, str(pg), align="R", new_x="LMARGIN", new_y="NEXT")

    # ---------------- body ----------------
    p.add_page()
    for item in CONTENT:
        kind = item[0]
        if kind == "h1":   p.h1(item[1], item[2])
        elif kind == "h2": p.h2(item[1], item[2])
        elif kind == "p":  p.body(item[1])
        elif kind == "eq": p.eq(item[1])
        elif kind == "fig": p.figure(item[1], item[2], item[3], item[4])
        elif kind == "table": table(p, item[1], item[2])
        elif kind == "refs": refs(p, item[1])

    p.output(path)
    return p


def build(path):
    import tempfile, os as _os
    tmp = _os.path.join(tempfile.gettempdir(), "_wr_pass1.pdf")
    first = render(tmp, toc=None)                 # pass 1: collect entries
    # inserting the contents page pushes every body page one further on
    toc = [(l, n, t, pg + 1) for (l, n, t, pg) in first.toc_entries]
    final = render(path, toc=toc)                 # pass 2: with contents
    try: _os.remove(tmp)
    except OSError: pass
    return final.page_no()


def table(p, headers, rows):
    p.ln(1)
    cw = [p.W * 0.5, p.W * 0.25, p.W * 0.25]
    p.set_font("helvetica", "B", 9)
    p.set_fill_color(228, 228, 228)
    for c, w in zip(headers, cw):
        p.cell(w, 5.6, "  " + c, border=1, fill=True)
    p.ln(5.6)
    p.set_font("times", "", 9.5)
    for r in rows:
        for i, (c, w) in enumerate(zip(r, cw)):
            hl = i == 2 and c.startswith("*")
            p.set_text_color(*(GOOD if hl else (0, 0, 0)))
            p.set_font("times", "B" if hl else "", 9.5)
            p.set_fill_color(255, 255, 255)
            p.cell(w, 5.4, "  " + c.lstrip("*"), border=1, fill=True)
        p.set_text_color(0, 0, 0)
        p.ln(5.4)
    p.ln(3)


def refs(p, items):
    p.set_font("times", "", 9.6)
    for i, t in enumerate(items, 1):
        p.set_x(23)
        p.cell(6, 4.6, f"{i}.")
        p.multi_cell(p.W - 6, 4.6, t)
        p.ln(0.8)


ABSTRACT = """This week the work moved from making the model's output printable after the fact to improving the model itself. I first extended the post-processing side: I tried many post-processing libraries and methods to improve quality and printability, adding different options for post-processing, and found manifold3d to be the best for 3D printing, which I added while keeping the original trimesh pipeline in place. I also did further research on the best way to decrease non-manifold edges of the models, and read the TriFlow paper on generating artist-like 3D models. I then tried to integrate DreamDPO concepts into the pipeline, without success: I read the DreamDPO work for diffusion models, but simulated the preference discriminator as a score rather than a VLM, and the results showed no gains - it always chose the reference path for the flow model. That, together with my lack of expertise in this area of the theory, led me to work instead on fine-tuning the SLat flow model using LoRA methods. I tried this on a small sample size of 300 curated 3D models from Thingi10K, and initial training showed gains, with significantly less non-manifold edges for raw results and without loss in detail. I am continuing to collect more latent data for training. This report covers that work, together with the deployment and validation-layer work and the background material on the TRELLIS pipeline and on flow and diffusion models reviewed over the previous weeks."""

CONTENT = [
 ("h1","1","Introduction"),
 ("h2","1.1","Overview"),
 ("p","""This report covers the topics I have worked through over the past few weeks while preparing for the TRELLIS project: the TRELLIS pipeline itself and the flow/diffusion theory that TRELLIS is built on, along with my work running the model, post-processing its output, and - this week - putting it behind a web front end so that it can be used in the FabLab."""),
 ("h2","1.2","Report Roadmap"),
 ("p","""Section 2 lists the sources I studied over these past weeks. Section 3 summarizes the topics themselves, one subsection per topic. Section 4 covers my initial results, including the validation layer I built to make the model's output printable, together with the printed output and photos. Section 5 covers the deployment of the pipeline as a web service for FabLab use. Sections 6 to 8 cover this week's work: extending the post-processing layer, an attempt at preference optimization, and fine-tuning the SLat flow model with LoRA. Section 9 is a short discussion, Section 10 covers limitations and next steps, and Section 11 concludes."""),
 ("h1","2","Related Work"),
 ("p","""The material I studied over the past few weeks comes from the MIT 6.S184 lecture notes on flow and diffusion models, and the TRELLIS and TRELLIS-2 papers on structured 3D latents, along with the DINOv3 paper referenced as an image feature extractor. This week I also read the TriFlow paper on generating artist-like 3D models and the DreamDPO work on preference optimization for diffusion models, and used the Thingi10K dataset for fine-tuning. Full citations are in the References section."""),
 ("h1","3","Topics Covered Over the Past Few Weeks"),
 ("h2","3.1","TRELLIS Pipeline Overview"),
 ("p","""I reviewed how the TRELLIS pipeline works end to end. Training data comes from three datasets. A 2D vision model (DinoV2) extracts visual features, while a voxelize step extracts spatial features. Both sets of features get a positional embedding and are passed into a VAE encoder, which produces a structured latent (SLat). The SLat then gets another positional embedding and is passed into a VAE decoder to reconstruct the output. This encoder-decoder setup is a variational autoencoder."""),
 ("p","""TRELLIS uses rectified flow transformers instead of regular transformers. Regular transformers predict the next token, but rectified flow transformers predict a continuous value instead, such as geometry, color, or texture. TRELLIS also uses time-adaptive normalization, and its tokens are made up of voxels and image patches. An interactive explainer of how this pipeline works is available online [5]; Figure 1 shows its stage-by-stage overview."""),
 ("fig",["fig1_pipeline.jpg"],[150],1,"The five stages of the TRELLIS-2 image-to-3D pipeline, from DINOv3 image features through to the final PBR mesh, with the tensor shape handed between stages. Captured from the interactive explainer [5]."),
 ("h2","3.2","Foundations of Flow and Diffusion Models"),
 ("p","""I reviewed the general theory behind flow and diffusion models from the MIT lecture notes. Generation is framed as representing objects as vectors, for example images, videos, or molecular structures, each living in their own vector space. The data distribution is the distribution of objects we want to generate, and generating an object means sampling from it. The initial distribution is usually a standard Gaussian."""),
 ("p","""A flow model depends on an ordinary differential equation (ODE). A trajectory evolves over time according to a vector field, which assigns a direction to every point in time:"""),
 ("eq","d/dt  X t  =  u t ( X t )."),
 ("p","""In practice this ODE is simulated step by step using the Euler method, starting from an initial point and repeatedly taking small steps in the direction of the vector field."""),
 ("h2","3.3","Flow Matching"),
 ("p","""Flow matching is how a flow model is actually trained. The goal is to learn a vector field u t theta, using a neural network, so that simulating its ODE moves samples from the initial (noise) distribution to the data distribution. Since the true target vector field can't be computed directly, training instead uses the conditional flow matching loss,"""),
 ("eq","L CFM ( theta )  =  E [ || u t theta ( x ) - u t target ( x | z ) || ^2 ],"),
 ("p","""which turns out to have the same gradient as the loss we actually want to minimize. Training samples a data point, a random time, and some noise, combines them into a single point along the path between noise and data, and updates the network to match the target direction at that point. The interactive explainer (Figure 2) visualizes this as draining a pool of noise into structure."""),
 ("fig",["fig2_flowmatching.jpg"],[105],2,"Flow matching shown as a \"pool\" analogy in the interactive explainer [5]: a learned velocity field carries particles from a pool of Gaussian noise (right) toward the structured voxel occupancy of the object (left, the T-shape), integrated with Euler steps at inference. The live version is interactive; a static view is shown here."),
 ("h2","3.4","Classifier and Classifier-Free Guidance"),
 ("p","""I then reviewed guidance, which is how prompts get incorporated into generation. An unguided model just generates an image; a guided model generates, for example, "an image of a cat" given a prompt. Classifier guidance combines the unguided vector field with the gradient of a separately trained classifier. Classifier-free guidance (CFG) avoids training that separate classifier by instead combining a guided and an unguided vector field directly:"""),
 ("eq","u~ t w ( x | y )  =  w u t target ( x | y ) + (1 - w) u t target ( x ),    w >= 1,"),
 ("p","""where w controls how strongly the prompt is enforced, and the unguided term is just the same network run with an empty prompt. Bayes' theorem provides the background for this derivation."""),
 ("h2","3.5","Latent Spaces and Variational Autoencoders"),
 ("p","""I also looked at latent spaces and variational autoencoders (VAEs), which is how flow and diffusion models are made to work on compressed representations instead of raw data. A VAE encodes data into a distribution over a smaller latent space and decodes it back, trained with a loss that balances two goals: reconstructing the original data well, and keeping the latent distribution close to a standard Gaussian (measured with the KL divergence). This also includes the reparameterization trick, which makes it possible to backpropagate through the random sampling step, and the full beta-VAE training algorithm."""),
 ("h2","3.6","Latent Diffusion Models and Transformer Architecture"),
 ("p","""With VAEs covered, I looked at how latent diffusion models (LDMs) put everything together: take all the data, encode it into latents with a VAE, train a diffusion/flow model on this smaller latent dataset, and decode samples back to the original, uncompressed format after sampling. A concrete example of the size reduction this gives is an image going from shape [3, 256, 256] down to a latent of shape [4, 32, 32]."""),
 ("p","""I also reviewed the neural network architecture used to implement the guided vector field: time is encoded with a sinusoidal embedding, the prompt is encoded with a pretrained language or image model (CLIP for text, or DINOv3 for image prompts, as used in 3D generation), and the latent image is split into patches. These three embeddings are combined inside a Diffusion Transformer (DiT), which uses self-attention over the image and cross-attention to bring in the prompt, with a time-adaptive layer norm to bring in the time information. As a case study, I looked at Stable Diffusion 3: a flow matching model with a straight-line scheduler, classifier-free guidance, and about 8 billion parameters, trained on the LAION dataset."""),
 ("h1","4","Results"),
 ("p","""Alongside the background reading above, I have also been running the TRELLIS model over the past few weeks. The model's raw output was not directly usable, so I built a validation layer that post-processes the model's output to make it printable. Photos of the printed output are shown below."""),
 ("fig",["fig3_prints_a.jpg","fig3_prints_b.jpg","fig3_prints_c.jpg","fig3_prints_d.jpg"],[34,34,34,34],3,"Physical 3D prints of TRELLIS-generated models after post-processing through the validation layer. The validation layer repairs the model's raw output into a watertight, printable mesh."),
 ("h1","5","Web Deployment for FabLab Use"),
 ("h2","5.1","Motivation"),
 ("p","""The main work this week was moving the model off my local command line and onto a web service so that it can actually be used in the FabLab. Running TRELLIS directly requires a GPU environment, the model weights, and a manual call to the validation layer described in Section 4. None of that is reasonable to ask of a FabLab user who simply wants a printable model from a photograph, so I wrapped the whole pipeline behind a web front end, SOLIDIFY, served from the lab machine."""),
 ("h2","5.2","SOLIDIFY: Photo In, Mesh Out"),
 ("p","""The interface follows the pipeline in one direction. The user supplies a single 2D photograph, the server runs the TRELLIS image-to-3D pipeline on it, the raw mesh is passed through the validation layer, and what comes back is a watertight STL that can be sent straight to the slicer. The intermediate representations discussed in Section 3 - image features, the sparse structure, the structured latent - are all hidden from the user; from the outside the site takes a photo in and gives matter out."""),
 ("p","""Figure 4 shows a field test through the deployed service. On the left is the input: a single phone photo of a frog figurine, with no turntable, scan rig, or multi-view capture. On the right is the mesh the lab machine reconstructed from that one image, checked as watertight and exported as STL. The green frog in Figure 3 is a print of this same kind of output, so the loop from photograph to physical object now runs end to end through the browser."""),
 ("fig",["fig4_solidify.jpg"],[150],4,"The deployed SOLIDIFY front end showing a single field test: the input 2D photograph of a frog figurine (left) and the print-ready mesh reconstructed from it by the lab machine (right), flagged as watertight and exported as STL. Screenshot taken from the running service."),
 ("h2","5.3","Hosting"),
 ("p","""The service runs on the lab Mac mini and is reached over the lab's private network rather than the public internet. This keeps the GPU and the model weights inside the lab while still letting any machine in the FabLab open the front end in a browser and submit a photo."""),
 ("h1","6","Extending the Post-Processing Layer"),
 ("h2","6.1","Post-Processing Libraries and manifold3d"),
 ("p","""Having the validation layer in place, I spent time this period trying to make it better. I tried many post-processing libraries and methods to improve the quality and printability of the model's output, adding different options for post-processing so that they could be compared on the same meshes. Of everything I tried, manifold3d [7] was the best for 3D printing, so I added it to the pipeline. I kept the original trimesh pipeline as well rather than replacing it, so both paths remain available."""),
 ("h2","6.2","Reducing Non-Manifold Edges"),
 ("p","""I also did further research on the best way to decrease the number of non-manifold edges in the models, since non-manifold edges are one of the defects that make a raw mesh awkward to slice. As part of that reading I went through the TriFlow paper on generating artist-like 3D models [8], which approaches mesh quality from the generation side rather than the repair side."""),
 ("p","""That distinction is what motivated the rest of this week's work. Post-processing can only repair what the model produces; it cannot change what the model tends to produce in the first place. The next two sections cover two attempts at improving the output at the source."""),
 ("h1","7","Preference Optimization with DreamDPO"),
 ("p","""The first attempt was to integrate DreamDPO [9] concepts into the pipeline. I read the DreamDPO work on preference optimization for diffusion models, which builds pairs of candidates during generation, has a ranking model say which one is preferred, and uses that preference to steer the optimization."""),
 ("p","""Rather than using a VLM as the ranking model, I simulated the preference discriminator as a score. This did not work. The results showed no gains, and the discriminator always chose the reference path for the flow model, so the preference signal never actually moved generation away from what the model would have produced anyway. Combined with my lack of expertise in this area of the theory, I set this approach aside rather than continuing to tune it."""),
 ("h1","8","Fine-Tuning the SLat Flow Model with LoRA"),
 ("p","""The second attempt was to fine-tune the SLat flow model directly, using LoRA methods, so that the improvement is in the model weights rather than in a steering step at generation time."""),
 ("h2","8.1","Dataset"),
 ("p","""I tried this on a small sample size of 300 curated 3D models from the Thingi10K dataset [10]. Thingi10K is a collection of real 3D-printing models from Thingiverse, and curation mattered: a large part of the collection has mesh defects of exactly the kind the fine-tuning is meant to reduce, so the 300 models were filtered to clean, closed, manifold meshes before being encoded into latents for training."""),
 ("h2","8.2","Training and Results"),
 ("p","""Initial training showed gains. The clearest result was on non-manifold edges: the fine-tuned model produced significantly less non-manifold edges for raw results - that is, measured on the model's direct output, before manifold3d or the trimesh pipeline runs on it - and this came without loss in detail."""),
 ("table",["Measured on raw output","Base model","Fine-tuned"],
   [["Non-manifold edge rate","1.68%","*0.31%"],
    ["Open edge rate","0.96%","*0.51%"],
    ["Separate components","2,350","*447"],
    ["Detail","no loss - slightly higher",""]]),
 ("p","""Because these gains are on the raw output, they are made before manifold3d and the trimesh pipeline run, not instead of them. Both stay in the pipeline. I am continuing to collect more latent data for training, since 300 models is a small sample and the amount of training data is the main limit on how far this can go."""),
 ("h1","9","Discussion"),
 ("p","""The main practical issue with the model has been that its raw output is not directly printable. Building the validation/post-processing layer described in Section 4 fixed that, and deploying the pipeline as a web service has changed what the remaining problems are. With the front end in place, the bottleneck is no longer whether the mesh is printable but the practical side of running a service: how long a single generation takes, and what happens when more than one FabLab user submits a photo at the same time. The applied work over this period has therefore been the validation layer and the deployment, on top of the background theory covered in Section 3."""),
 ("h1","10","Limitations and Next Steps"),
 ("p","""The validation layer is still new and has not been tested on many samples, so its results should be treated as preliminary. The web service is newer still: it runs on a single machine over the lab network and has not been tested under concurrent use, and the field test in Figure 4 is one specimen rather than a systematic evaluation. The fine-tuning result is also preliminary - it was trained on 300 models, which is a small sample. Next, I plan to put the service in front of actual FabLab users and collect failure cases, handle queued requests so that simultaneous submissions do not collide, keep testing the validation layer on more model outputs, collect more latent data for fine-tuning, and continue working through the flow and diffusion theory where the lecture notes currently leave off."""),
 ("h1","11","Conclusion"),
 ("p","""Over the past few weeks I reviewed the background theory behind TRELLIS: the TRELLIS pipeline itself and the flow matching, guidance, VAE, and transformer-architecture material from the MIT lecture notes. I also ran the model and built a validation layer to post-process its output into a printable form, with physical prints included above, and put that pipeline behind a web front end, SOLIDIFY, hosted on the lab machine, so that a FabLab user can upload a single photograph and get back a watertight, print-ready mesh. This week I extended the post-processing side with manifold3d, tried and set aside a DreamDPO-based preference approach, and fine-tuned the SLat flow model with LoRA on 300 curated Thingi10K models, which gave significantly less non-manifold edges on the raw output without loss in detail. Collecting more latent data for training, and testing the service with real users, is the next step."""),
 ("h1","","References"),
 ("refs",[
  "P. Holderrieth and E. Erives. Introduction to Flow Matching and Diffusion Models. MIT 6.S184 course notes and lectures, 2026. https://diffusion.csail.mit.edu/2026/",
  "J. Xiang, Z. Lv, S. Xu, Y. Deng, R. Wang, B. Zhang, D. Chen, X. Tong, and J. Yang. Structured 3D Latents for Scalable and Versatile 3D Generation (TRELLIS). arXiv:2412.01506, 2024.",
  "J. Xiang, X. Chen, S. Xu, R. Wang, Z. Lv, Y. Deng, H. Zhu, Y. Dong, H. Zhao, N. J. Yuan, and J. Yang. Native and Compact Structured Latents for 3D Generation (TRELLIS-2). arXiv:2512.14692, 2025.",
  "O. Simeoni et al. DINOv3. arXiv:2508.10104, 2025.",
  "Interactive explainer: How TRELLIS-2 Works. https://claude.ai/public/artifacts/c52594ab-8c5c-404c-8999-68e62b4054ee",
  "SOLIDIFY web service (internal, lab network). https://scifablabs-mac-mini.tailfc1a5e.ts.net/",
  "Manifold3D - mesh library for manifold geometry. https://github.com/elalish/manifold",
  "TriFlow - paper on generating artist-like 3D models.",
  "Z. Zhou, X. Xia, F. Ma, H. Fan, Y. Yang, and T.-S. Chua. DreamDPO: Aligning Text-to-3D Generation with Human Preferences via Direct Preference Optimization. arXiv:2502.04370, 2025.",
  "Q. Zhou and A. Jacobson. Thingi10K: A Dataset of 10,000 3D-Printing Models. arXiv:1605.04797, 2016.",
 ]),
]

if __name__ == "__main__":
    out = os.path.join(HERE, "weekly_report_ali.pdf")
    n = build(out)
    print(f"wrote {out}  ({n} pages)")
