# Research library interface

## 1. Visual theme and atmosphere

Warm editorial utility: cream paper, deep ink, and restrained teal marks. The UI should feel like a working paper index, not a generic analytics dashboard.

## 2. Color palette and roles

- Canvas: `oklch(0.955 0.018 85)`, page background.
- Surface: `oklch(0.985 0.01 85)`, primary work surface.
- Ink: `oklch(0.255 0.025 235)`, primary text.
- Muted ink: `oklch(0.49 0.025 235)`, supporting text.
- Teal: `oklch(0.52 0.105 174)`, links, focus, verified state.
- Amber: `oklch(0.72 0.13 75)`, review and degraded state.
- Rose: `oklch(0.58 0.16 25)`, failed state.

## 3. Typography rules

- Display: Source Han Serif SC or Songti SC, 32 to 48px, 650 weight, `-0.022em`.
- UI: Source Han Sans SC or system Chinese sans, 14 to 16px.
- Evidence: serif body at 15px and 1.75 line height.
- Scores and page numbers use tabular numerals.

## 4. Component stylings

- Buttons use the pill radius and a 40px minimum hit area. Press state is `scale(0.95)`.
- Paper rows are cardless index entries separated by hairlines.
- Evidence uses an inset paper surface with a short teal rule, never a thick side border.
- Inputs use a 12px radius, visible focus ring, and inline error text.

## 5. Layout principles

Spacing scale: 4, 8, 12, 16, 24, 32, 48. Desktop pages use a 12-column content grid; mobile collapses to one column without hiding evidence or status.

## 6. Depth and elevation

Depth comes from background steps. Only floating upload and edit panels use `0 12px 32px oklch(0.25 0.02 235 / 0.08)`.

## 7. Do and do not

- Do keep page numbers beside every quote.
- Do expose fallback and OCR reasons in plain language.
- Do keep metadata confidence beside the field it describes.
- Do not use decorative gradients or glass surfaces.
- Do not hide failed parsing behind a generic job status.
- Do not show `retrieval_text` as a quote.

## 8. Responsive behavior

The navigation wraps below 640px. Two-column reading layouts collapse below 1024px. Every action remains at least 40 by 40px, with pointer-only hover effects.

## 9. Agent prompt guide

- "Create a paper index row on `oklch(0.985 0.01 85)`, title 22px serif weight 650 at `-0.012em`, metadata 13px sans, page/status controls with pill radius."
- "Create an evidence block on `oklch(0.97 0.015 85)`, quote 15px serif at 1.75 line-height, teal `oklch(0.52 0.105 174)` page link, 12px radius."
- "Create a metadata form with 12px radius inputs, 40px controls, teal focus ring, and confidence/source text directly below each field."
