# TouchSafeBench project page

Source of the project website for

> **Probing Collision Grounding in Vision-Language Models for Safe Human–Robot Collaboration**
> Jun Wang*, Xiaohao Xu*†, Xiaonan Huang — University of Michigan, Ann Arbor — CoRL 2026
> arXiv: https://arxiv.org/abs/2605.31196

Live site: https://jun25336-bit.github.io/TouchSafeBench/

The page is a single static `index.html` built on the
[Nerfies](https://github.com/nerfies/nerfies.github.io) template (Bulma + Font Awesome), in the
same style as the ASIMOV Benchmark pages.

## Layout

```
index.html                      the page
static/css/index.css            page-specific styles (template styles + a few additions at the end)
static/css/bulma.min.css        Bulma framework (unmodified)
static/css/fontawesome.all.min.css, static/js/fontawesome.all.min.js   icons (unmodified)
static/images/                  figures exported from the paper, plus the favicon
static/touchsafebench.bib       the "Cite" button target
.nojekyll                       tells GitHub Pages to serve the files as-is
```

## Editing

* Text, links, and figure captions: edit `index.html` directly.
* To add a video, drop an `.mp4` in `static/` and add a `<video controls>` block (see the ASIMOV v1
  page for an example) between the hero and the teaser sections.
* Author homepages: wrap a name in `<a href="..." target="_blank">...</a>` in the authors block.

## Deploying

The site is served from the `gh-pages` branch of
[`jun25336-bit/TouchSafeBench`](https://github.com/jun25336-bit/TouchSafeBench). This folder is a
checkout of that branch, so after editing:

```bash
git add -A
git commit -m "Update project page"
git push
```

GitHub Pages rebuilds within about a minute.
