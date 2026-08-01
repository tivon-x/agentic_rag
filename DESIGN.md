# Evidence-first editorial interface

## Product stance

This is a Chinese-first research workspace for reading a local paper library and
answering questions with inspectable evidence. It is useful every day, but it
must also make the system's judgment legible in a technical interview.

## Visual direction

The interface takes editorial cues from a contemporary science magazine: warm
paper canvas, ink-black type, disciplined rules, square geometry, and a single
ink-blue information color. This direction is informed by the WIRED-inspired
reference in VoltAgent/awesome-design-md, without reproducing its branding.

The design signature is the **evidence rail**: citations appear as compact
numbered page markers and expandable source strips beside an answer, rather
than as a detached block of Markdown. This makes provenance a first-class part
of reading.

## Design tokens

| Token | Value | Use |
| --- | --- | --- |
| Canvas | `#f6f4ee` | primary page background |
| Paper | `#fffdf8` | raised reading surfaces |
| Ink | `#181715` | primary text and strong rules |
| Muted ink | `#6d6a63` | metadata and supporting copy |
| Rule | `#d9d5cc` | dividers and field borders |
| Ink blue | `#0b5ea8` | links, active states, evidence markers |
| Signal amber | `#a56513` | processing and caution states |
| Signal red | `#b33a2f` | failures and destructive actions |

## Typography

- Display: `Iowan Old Style`, `Baskerville`, `Songti SC`, serif. Use for page
  titles and research statements, never dense controls.
- Interface: system sans, `PingFang SC`, `Noto Sans SC`, sans-serif. Use for
  Chinese body copy and controls with 1.7–1.8 line height.
- Metadata: system monospace. Use sparingly for counts, run IDs, page numbers,
  and technical labels.

## Layout and hierarchy

- Desktop-first content grid: a narrow navigation column or masthead, a main
  reading column, and an optional evidence or metadata rail.
- Use hairline rules and generous whitespace to group content. Avoid floating
  rounded-card grids and decorative gradients.
- Keep primary actions ink-black. Ink blue communicates navigation and evidence,
  not generic emphasis.
- Use square corners by default; `4px` is the maximum rounding for controls.

## Components

- **Masthead**: a compact publication name, current section, and thin rule.
- **Section label**: uppercase Latin or spaced Chinese metadata above a display
  heading.
- **Evidence rail**: source label, page marker, quote preview, direct paper link,
  and an accessible disclosure for the full quote.
- **Result row**: title, source metadata, excerpt, score or status; separated by
  rules rather than boxed as a card.
- **Status marker**: text plus a small square or dot. Never rely on color alone.
- **Action**: a high-contrast rectangular button with an explicit label.

## Interaction and motion

- Focus states are visible at all times and meet keyboard navigation needs.
- Evidence disclosures and lightweight entry transitions may animate for
  140–220ms. Do not animate layout continuously or use a global `transition`.
- Honour `prefers-reduced-motion` by disabling non-essential animation.
- Loading states explain the current step in plain Chinese; errors say what the
  user can do next.

## Content rules

- Chinese is the primary language. Keep English to proper names and technical
  identifiers.
- Cite source title, section, and page whenever the data is available.
- Never present experimental retrieval variants as a selectable production mode.
  The fixed baseline is `v1_flat_rerank`; experiment outcomes belong to the
  evaluation workspace.
