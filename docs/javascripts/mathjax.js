/* MathJax configuration for the mkdocs-material site (docs/paper/*.md).
 *
 * pymdownx.arithmatex runs in generic mode, so the README's $...$ and $$...$$ spans
 * are emitted as \(...\) and \[...\] and typeset client-side. The ignore pattern
 * skips everything outside .arithmatex spans, which keeps code blocks and prose
 * untouched - the same discipline tests/test_typography.py enforces on the source.
 */
window.MathJax = {
  tex: {
    inlineMath: [["\(", "\)"]],
    displayMath: [["\[", "\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
