# models/

| File | What | Provenance | Licence |
|---|---|---|---|
| `xt_model.json` | Expected Threat grid, 12 rows × 16 columns, socceraction JSON format | Fitted with `socceraction.xthreat.ExpectedThreat(l=16, w=12)` on all 1,890 legacy SPADL games (StatsBomb open data: Bundesliga, La Liga, Ligue 1, Premier League, Serie A, World Cup 2018), every game played left to right. Copied unchanged from the sibling project `VAEP_Football/scripts/train_xt.py`. | Derived from [StatsBomb open data](https://github.com/statsbomb/open-data) — attribution required by its user agreement. |
| `yolo11m.pt` | Ultralytics YOLO11m COCO detector | Auto-downloaded by Ultralytics on first use | AGPL-3.0 — **not committed** (`*.pt` is git-ignored) |

The xT grid describes elite men's European football. Applying it to an ISL match
assumes similar scoring geography; it is used here only for the exploratory
ball-zone values, never as a validated rating.
