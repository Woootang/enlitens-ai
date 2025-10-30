# Monitoring UI Redesign Brief

## Goals
- Transform the monitoring dashboard into a vibrant, neurodivergent-friendly interface aligned with Enlitens brand.
- Reduce cognitive overload by using clear hierarchy, modular panels, and animated feedback.
- Integrate assets, colors, and typography already defined in `style.css`.
- Inspire implementation using best-in-class references (Flowfest, Relats, surf analytics, DSFD dashboard).

## Brand & Token References
- **Primary palette** (from `style.css`):
  - `--sunset-orange: #FF8A4D`
  - `--lemon-glow: #FCE76E`
  - `--cotton-candy: #FFB6E1`
  - `--deep-indigo: #2B2D83`
  - `--forest-teal: #0E6F6F`
- **Neutrals & glassmorphism**:
  - `--warm-sand: #F6E9DA`
  - `--ink: #1F2033`
  - Shadows: `var(--shadow-xl)` / blurred overlays defined near `:root`.
- **Typography**:
  - Heading font: `"ClashDisplay", "Poppins", sans-serif`
  - Body font: `"Inter", "Nunito", sans-serif`
  - Utility classes: `.heading-xl`, `.label-sm`, `.badge`, `.pill`
- **Components already available**:
  - Gradient buttons `.btn-gradient`
  - Card shells `.panel`, `.panel-glass`
  - Tab styles `.tab-pill`
  - Badge chips `.chip`, `.chip-outline`

## Layout Inspiration Interpretation

### Flowfest (retro cards with rounded frames)
- Apply layered rounded rectangles with drop shadows for the hero status bar.
- Use accent arcs / pill-shaped buttons for “Pause”, “Export”, “Refresh JSON”.
- Introduce subtle animation (scale-up on hover) to replicate playful festival feel.

### Relats industry grid (modular mosaic)
- Split main viewport into a responsive CSS grid:
  - **Row 1**: Status hero (spans full width).
  - **Row 2**: Three equal cards — “Views”, “Quick Stats”, “Controls”.
  - **Row 3**: Two-thirds width timeline chart + one-third width “Foreman AI” Q&A.
  - **Row 4**: Full-width log stream using masonry-style segments.
- Each card uses `.panel-glass` with border radius `32px`, shadow `var(--shadow-xl)`.

### Surf analytics (data tiles)
- Create micro-tiles for key metrics (documents processed, errors, warnings, quality score) with horizontal gradient backgrounds.
- Include iconography (use `phosphor-icons` or `lucide`) to make tiles scannable.
- Add mood gauge (semi-circle progress) for quality score with `conic-gradient` background.

### DSFD dashboard (dark/light contrast)
- Introduce toggle for “Focus Mode” (dark) vs “Daylight Mode” (light) leveraging CSS variables.
- Align chart styles (line charts with glowy gradients, clean axis labels).

## Proposed Screen Structure

```
<main class="dashboard">
  <section class="status-bar">
    <div class="status-card">
      <h1 class="heading-xl">Current Document</h1>
      <div class="status-meta">
        <span class="chip chip-glow">Initializing…</span>
        <span class="label-sm">Progress: 0%</span>
        <span class="label-sm">Last update: --</span>
      </div>
      <div class="progress-gradient">
        <div class="progress-fill" style="width: var(--progress)"></div>
      </div>
    </div>
  </section>

  <section class="grid-panels">
    <article class="panel panel-glass">Views (tabbed)</article>
    <article class="panel panel-glass">Quick Stats + micro tiles</article>
    <article class="panel panel-glass">Controls with glossy buttons</article>
  </section>

  <section class="grid-row">
    <article class="panel panel-glass span-2">Live Timeline / charts</article>
    <article class="panel panel-glass">Foreman AI conversational view</article>
  </section>

  <section class="panel panel-glass logs-masonry">Live Processing Logs</section>
</main>
```

## Detailed Component Guidance

### 1. Status Bar
- Use gradient background `linear-gradient(135deg, var(--sunset-orange), var(--cotton-candy))`.
- Add animated background stripes (CSS `background-size: 200%` + `animation: shimmer 8s infinite`).
- Progress bar: `border-radius: 999px`, `box-shadow: inset 0 0 0 2px rgba(255,255,255,0.2)`.
- Include icon buttons for “open latest_output.json” and “view logs” with tooltip.

### 2. Quick Stats Tiles
- Four tiles in a flex row, gap 16px.
- Each tile color-coded: e.g., documents `--forest-teal`, errors `--sunset-orange`, warnings `--lemon-glow`, quality `--deep-indigo`.
- Include sparkline mini-chart using SVG `<path>` or `canvas` overlay.

### 3. Controls Bar
- Convert buttons to `.btn-gradient` with neon hover outlines.
- Display runtime meta (PID, uptime) as pill badges.
- Add switch component (`.toggle-pill`) for “Auto-scroll logs”.

### 4. Views Section
- Use tab pills across top (Live Logs, Agent Pipeline, Quality, JSON Viewer, Foreman, Statistics).
- Active tab underlined with sliding indicator.
- Content area uses `overflow-y: auto` with subtle `scrollbar-color` defined in `style.css` (`.scrollbar-soft`).

### 5. Foreman AI Panel
- Chat bubble layout, alternating gradient backgrounds for system vs user.
- Add floating action button “Ask Foreman” bottom-right (circular, glowing ring animation).

### 6. Logs Masonry
- Wrap log lines in collapsible groups (INFO/WARN/ERROR) with gradient header bars.
- Sticky filter bar with checkboxes (Info/Warning/Error), search input styled via `.input-chip`.
- Apply `monospace` font but blend with background (rgba ink panel with 60% opacity).

## Accessibility & Motion
- Ensure 4.5:1 contrast for text; use `--ink` on light surfaces, `--warm-sand` on dark.
- Provide `prefers-reduced-motion` media query disabling shimmer + bounce.
- Add focus outlines: `outline: 2px solid var(--cotton-candy); outline-offset: 4px;`.

## Implementation Notes
- Centralize new CSS in `monitoring_ui/src/styles/dashboard.css` and import existing tokens from `style.css` via `@import '../style.css';` or copy token section into dedicated `_tokens.css`.
- Component mapping:
  - `StatusHeader.tsx` or `.vue` -> status bar.
  - `StatsGrid.tsx` -> micro tiles.
  - `ControlsPanel.tsx` -> buttons & toggles.
  - `ViewsTabs.tsx` -> tabbed panel.
  - `LogsMasonry.tsx` -> virtualization for logs (`react-virtualized` or manual `IntersectionObserver`).
- Use CSS variables for dynamic data: e.g., `<main style="--progress: ${progress}%">` so progress bar updates via CSS only.

## Interaction Ideas
- On log hover, show tooltip with timestamp, agent, and copy button.
- Real-time glow pulse when new errors arrive (applied to Errors tile and Logs header).
- Provide screenshot/export button that captures the dashboard using `html2canvas` styled as `.btn-icon`.

## Deliverables for Devs
1. Update monitoring UI layout per structure above.
2. Link `style.css` tokens; remove redundant inline styles.
3. Implement responsive breakpoints at 1440px, 1024px, 768px.
4. Add dark mode toggle using `data-theme="dark"` switching CSS variables (predefined in `style.css` section `:root[data-theme='dark']`).

## Assets to Reuse
- `style.css` gradients (`.gradient-arc`, `.blob-background`, `.glass-panel`).
- Icon sprites under `monitoring_ui/public/icons` (if not present, reference `phosphoricons.com`).
- Motion keyframes: `@keyframes floaty`, `@keyframes shimmer`, `@keyframes pulse-ring`.

---
This brief should equip Claude Code / GPT Codex to rework the UI while staying faithful to Enlitens branding and the inspiration references.
