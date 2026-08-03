# Editorial web direction

The web UI follows the repository-level `DESIGN.md`. When a local note differs,
these rules are authoritative for this Next.js app.

## Visual contract

- Warm paper canvas `#f6f4ee`, paper surface `#fffdf8`, ink `#181715`, muted ink
  `#6d6a63`, rules `#d9d5cc`, and ink blue `#0b5ea8`.
- Chinese-first copy, serif display text, system sans for controls, and monospace
  for compact metadata such as page numbers and session dates.
- Use hairline rules and whitespace for grouping. Default corners are square;
  controls may use at most `4px` rounding.
- Do not use gradients, glass surfaces, generic dashboard cards, pill controls,
  or `transition: all`.

## Chat shell

Chat is an independent `100dvh` reading shell. At `1280px` and wider it has a
248px session rail, a flexible message column, and a 352px answer-level evidence
rail. From 768px to 1279px the evidence rail becomes an overlay; below 768px the
top bar opens a session drawer and each answer keeps its evidence inline.

The composer stays attached to the bottom of the message column. User messages
are right-aligned on a light paper-gray field. Assistant messages remain open on
the canvas without cards. Evidence is only rendered from the API payload: no
sentence-level citations are invented. Source links open the paper route in a
new tab and preserve the existing `paper_id` plus `page` query rules.

## Accessibility and motion

- Keep the global skip link and visible keyboard focus rings.
- Every source link has an accessible label containing the paper and page when
  available; status is also communicated as text, not color alone.
- Use `prefers-reduced-motion` and avoid continuous layout animation.

## Responsive editorial pages

Library, search, papers, and the knowledge-base redirect live under the
`(editorial)` route-group layout. The parentheses do not change their public
URLs. That layout owns the masthead and footer; the root layout only owns the
document shell, metadata, global CSS, and skip link.
