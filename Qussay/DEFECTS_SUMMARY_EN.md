# Known Limitations — Tactical Camera and TV Broadcast

Last updated: 2026-10-18. Tactical-camera issues are **measured** on two full matches (Juventus–Milan, Inter–Napoli)
and compared with Sofascore statistics. Broadcast issues are **expected** — the system has not been run on broadcast footage yet.

---

## 1. Tactical camera (measured)

### Ball
- Visible in only **65%** of frames (up from 46%).
- Fast and airborne balls are lost: the detector sees them at 0.14–0.55 confidence, below the threshold.
- **Ball height is unknown**: a lofted ball is projected to the wrong spot on the ground.
- **Static spare balls** near the touchline are taken for the match ball (34% of top candidates in Inter–Napoli; now rejected outside the pitch).
- A fixed object behind one goal was detected as a ball in Juventus–Milan (fixed).

### Player identity
- **Fragmentation:** 1,738 track IDs for 32 players in one match.
- **ID switches:** an ID jumps between players in contact (88 cases in 10 minutes), sometimes across teams.
- **Jersey numbers cannot be read** (~8 px at 720p).
- **Naming is manual:** ~150 chains per match (30–45 minutes).
- Automatic linking degrades with time: 87% correct at a 5 s gap, 60% at 20 s.
- A name is attached to a whole track ID, so a chain containing a switch is partly mislabelled.
- The new kit-colour-aware tracker is not adopted yet: mixed results, and its metric is biased in its favour.

### Teams and goalkeepers
- Team = shirt colour; tested only on clearly different kits.
- **Goalkeepers** are misassigned by colour (~50% of keeper frames in Juventus–Milan); corrected by position.
- **Attack direction** from goalkeepers failed in Inter–Napoli; measured manually from the kick-off formation.

### Events (Inter–Napoli vs Sofascore)
| Event | System | Truth | Issue |
|---|---|---|---|
| Passes | 813 | 884 | 92%; about half are "inferred" (ball not seen all the way); one-touch passes are missed |
| Goals | 5 | 5 | ✓ (from kick-offs only) |
| Corners | 19 candidates | 15 | ~4 extra, need review |
| **Shots** | **8 automatic** | **45** | **Weakest point**; current fix is human review of ~127 clips |
| Tackles | 56–65 | 42 | **Inflated**, and the team split is wrong |
| Fouls, cards, offside | — | — | Not possible |
| Throw-ins, free kicks | partial | — | Not reliable |
- **Possession** is time-based (Sofascore is pass-based) — not directly comparable.
- **Blocked shots** where play continues are not detected automatically.

### Camera and calibration
- Position jump every 10 frames (median 0.37 m, sometimes 12 m) — partly fixed by smoothing.
- Part of the pitch is sometimes out of view (~20 of 22 players visible).
- **720p** limits everything: ball ~5 px, player ~37 px.

### Operations and review
- **Human review ~1–1.5 hours per match** (shots + names).
- Needs a rented GPU (~$1.5, ~2–3 hours per match); local laptop is far too slow (3.8 s/frame).
- Some steps (heatmaps, name suggestions) are still tied to a single match.
- No official event log for event-by-event comparison; only totals are compared.
- All thresholds are tuned on two matches only — untested on other leagues, stadiums and kits.

---

## 2. TV broadcast (expected — not yet tested)

### Camera cuts
- **Close-ups, replays, crowd and coach shots** — every cut **breaks tracking** and IDs restart.
- **Replays** duplicate events unless removed.
- Needs a shot-type layer (wide / close-up / replay) — not built yet.

### Calibration
- **Fails on close-ups** (not enough pitch lines).
- Constant panning and zooming ⇒ larger position errors.

### Coverage
- **Only ~8–14 players visible** — the camera follows the ball.
- **Team shape, heatmaps and distances** are incomplete.
- **Kick-offs and corners** (which rely on both teams' formation) are harder to detect.
- Anything that happens during a replay is lost.

### Identity
- Re-linking players after every cut is **the hardest part**; needs the role/formation layer (not built).
- Roles separate lines, not individuals (two centre-backs, two holding midfielders).

### Graphics and noise
- Scoreboard, ads and graphic overlays may be detected as players or balls.
- Slow-motion effects in replays.

### Expected advantages of broadcast
- The ball is larger and clearer.
- **Jersey numbers and faces are readable in close-ups** ⇒ easier player naming.
- Usually higher resolution (1080p).

---

## 3. Shared limitations
- No event-by-event ground truth (only Sofascore totals and a few human verdicts).
- No consistent confidence score shown to the user for each number.
- Reliance on a single human reviewer.
- No ball height, so no detection of aerial balls or headers.
- Licences: some useful models are non-commercial or GPL/AGPL (acceptable for internal use only).
