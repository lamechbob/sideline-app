# Changelog
All notable changes to this project will be documented in this file.

Format: Keep a Changelog.  
Versioning: Semantic Versioning (MAJOR.MINOR.PATCH).  
Release cadence: weekly on Wednesdays.

---

## [Unreleased] — next scheduled release: 2025-09-10

### In Progress

### Planned
- **Email Triggers**
  - Send scheduled emails to coaches with a PDF summary of the current week’s statistics  
  - Delivery: Sundays at noon (all high school games should be completed by Saturday night)
- **Player awards:**
  - Admin can assign awards to a player.
  - Generate an S3 “award card” image on save.
  - Show the public URL after save for quick Instagram sharing.
- **Social graphics:**
  - Add exportable stat graphics sized for Instagram feed (1080×1350) and stories (1080×1920).
  - Simple preset layouts for player highlights and weekly leaders.

---

## [0.4.0] — 2025-09-03
### Added
- **Weekly View**
  - Display **FG Attempts** and **PAT Attempts**.

### Changed
- **Weekly View**
  - Kicking columns now list **Attempts** before **Made** (FG Attempts → FG Made; PAT Attempts → PAT Made) for readability.

### Fixed

### Notes
- No schema changes required beyond having `fg_attempts` and `pat_attempts` present (normalizers already map common variants).

---

## [0.3.0] — 2025-08-26

### Added
- **Spreadsheet**
  - `Class` (graduation year) column.
  - `Position 1`, `Position 2`, `Position 3` columns (up to three positions per player).
- **Database**
  - `Players.GraduationYear` (INT).
  - `TeamRoster.PositionID1`, `PositionID2`, `PositionID3`.
  - New `Position` codes: `FB`, `DE`, `SS`, `FS`, `MLB`, `OLB`, `DB`, `ATH`.

### Changed
- **Lambda (Roster Import)**
  - Imports three positions per player (`PositionID1/2/3`).
  - Expects height already in inches (no conversion in Lambda).
  - Header normalization updated: `No` → `JerseyNumber`, `First Name`, `Last Name`, `Class` → `GraduationYear`, `Position 1/2/3`.
  - PlayerID resolution no longer uses `DOB`.
- **Database Schema**
  - `Players`: dropped `DOB` and `MiddleInitial`.
  - `TeamRoster`: removed single `Position` column.
- **View**
  - `public.player_week_stats` now concatenates up to three positions (`PositionID1/2/3`) into one string (e.g., `RB, SS, ATH`), skipping blanks.

### Fixed
- **Streamlit App**
  - Jersey labeling now accepts `#0` as a valid number.

### Notes


---

## [0.2.0] — 2025-08-20
### Added

### Changed
- **Player Details Page**
  - Add missing stats: Deflections, Rushing Average, Receiving Average, Assisted Tackles  
  - Remove “Season Totals” from stat headers (e.g., “Passing – Season Totals” → “Passing”)  
  - Add “Season Totals” as a separate header  

- **Weekly View**
  - Update header: “Players – Week 0 Totals” → “Week 0 Totals”  
  - 
### Fixed

### Notes


---

## [0.1.0] — 2025-08-15
### Added
- Initial public release of the South Broward Football app.
- Two primary views: Season Leaders and Weekly View.
- Column normalizer for common stat header variants.
- “Last updated” banner sourced from `game_date`.
- Public S3 mode for summary CSV and optional private mode via secrets.

### Changed
- Week picker defaults to the latest available week.
- Jersey number formatting and player labels improved.
- Season Leaders rendered as clean tables without index.
- Weekly View layout updates: Pass Deflections displayed next to Interceptions; Rush Attempts derived from “Rush” events when needed; rushing and receiving averages added; Targets limited to “Pass Target” plus “Catch”.

### Fixed
- Targets and averages now compute correctly across normalized columns.

### Notes
- End-to-end ETL tested during a preseason game to validate S3 → Lambda → RDS → Streamlit.

---

## Maintenance

### Release checklist (run each Wednesday)
1. Merge PRs labeled `next-release`.
2. Update version in app and deployment artifacts.
3. Update this CHANGELOG under the new version section with date and items shipped.
4. Tag the release in Git with `vX.Y.Z`.
5. Deploy to Streamlit Cloud and verify:
   - App loads
   - Secrets resolve
   - S3 access works in selected mode
6. Post-release: move any incomplete items back to **Unreleased** and re-label issues.

### Conventions
- Use labels: `feature`, `fix`, `perf`, `docs`, `infra`, `breaking-change`, `next-release`.
- Keep entries user-facing and brief. Technical details live in PRs and issues.

