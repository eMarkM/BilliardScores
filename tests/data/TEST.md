Test Sheets

11 v 2
11 Cyrus, Francis, Brad
2 Will, Anthony, Bryan

14 v 9
14 Angel, Erwin, Randolph
9 John, Jake, Gil

4 v 13
4 Jason, Biren, Steve
13 Mike, Bobby, Duke

5 v 10
5 Sue, Gary, Glenn
10 Kelley, Matt, Pablo

7 v 12
7 Tony, Gina, Dansop
12 Pavel, TS, Mooses

Remy Notes

## Iteration 1 (setup + baseline)
- Added integration test harness for Team11vTeam2 + Team14vTeam9 expected JSONs.
- Installed python deps locally (pytest, Pillow, openai, opencv-headless) so tests can run.
- Baseline behavior on Team11vTeam2 (record mode) shows major row-detection instability:
  - Border-line heuristic rejects many candidate crops.
  - Eventually accepts wrong handwriting for the slot (e.g. row_index=1 but player name looks like Francis/Brad).
  - Can produce duplicate player names across rows.
- Found a bug: `extract_rows_by_cropping()` can return `None` when row band detection fails, which then crashes normalization.

## Iteration 2 (tighten crop + retries)
- Problem observed: row-slot swapping / duplicate names (row1+row2 both become "Francis").
- Change: tightened crop padding and reduced scan range to avoid bleeding into adjacent rows.
- Change: added retries in `OpenAIVisionClient` to reduce flaky "Model returned empty response" failures.
- Result: worse missed-row behavior (home player 3 often fails entirely) — the border precheck rejects too many candidates.

## Iteration 3 (relax border precheck)
- Change: only enforce the border-line heuristic for the first few attempts; later attempts proceed to the model.
- Result: still failing on home player 3 for Team11vTeam2. The model frequently returns row_index=-1 / blank player, suggesting the crop isn’t consistently capturing the tiny printed row index cue.

## Iteration 4 (anchor on header + grid triplets, remove row_index gating) 
- Goal: stop depending on tiny printed row numbers; anchor row bands off the printed header + grid lines.
- Change (WIP, not yet committed at time of this note):
  - Removed row_index requirement from row prompt/schema and acceptance logic.
  - Updated gridline band selection to assume repeating triplets: SCORE, MARK, OPPONENTS; take the 1st band of each triplet.
  - Tried shrinking the crop to the top fraction of each band to avoid opponents row.
  - Removed border-line precheck entirely (was causing missed rows under glare/shadows).
- Current result: still failing on Team11vTeam2 because crops are still contaminated by opponents/total-team area (e.g. player becomes "3v6" or "158"). Indicates our banding/cropping still includes printed opponents row or the handwritten "158" below the tables.

## Iteration 5 (fix header anchor detection)
- Problem: we were overshooting from the start because `detect_row_bands_by_grid()` guessed the header line at ~16% of image height; on this sheet it picked a much lower gridline, so every subsequent row crop was shifted down into opponents.
- Change: pick the header separator line by gridline spacing pattern (short header band followed by taller score band), with a fallback heuristic.
- Result: extraction now finds 6 rows consistently, but the *content* is still wrong (player names like "chris" and scores badly off expected), indicating the crop geometry is still incorrect and/or the model is reading wrong cells.

