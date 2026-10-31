"""mkdocs hook: make the section pages' repository-relative links work on the site.

The pages under docs/paper are written to be read on GitHub, where `../../results/figures/x.png`
and `../../src/...` resolve against the repository. The site has no repository around it: the docs
workflow copies `results/figures` into the build root, and everything else is reached on GitHub. So
on the site, figures are re-pointed one level up, the contents link goes to the site's home page,
and any other repository path becomes a link to the file on GitHub.
"""

from __future__ import annotations

import re

REPOSITORY = "https://github.com/PedroM2626/smw-sciml/blob/master/"
_REPO_LINK = re.compile(r"\]\(\.\./\.\./([^)\s]+)\)")


def _rewrite(match: "re.Match[str]") -> str:
    target = match.group(1)
    if target.startswith("results/figures/"):
        return f"](../{target})"
    if target.startswith("README.md"):
        return "](../index.md)"
    return f"]({REPOSITORY}{target})"


def on_page_markdown(markdown: str, page, **kwargs) -> str:  # noqa: ANN001 - mkdocs hook API
    if not page.file.src_uri.startswith("paper/"):
        return markdown
    return _REPO_LINK.sub(_rewrite, markdown)
