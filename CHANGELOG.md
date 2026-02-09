# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.1.0-pre.1] - 2026-02-09

First cut of a working OCR bot demo.

### Added
- Telegram bot that accepts scoresheet photos and produces a CSV.
- `/confirm` workflow for accepting a pending upload.
- `/fixname` and `/fixscore` commands for correcting OCR mistakes.
- Human-friendly “processed” reply caption including the processed filename.

### Changed
- Row extraction/cropping heuristics to reduce grabbing "opponents" / "mark" sub-rows.
- Extraction failures now report which team/player could not be processed.
- `/fixscore` now recalculates totals when editing game1..game6.

### Known issues
- Some handwritten `10`s may be misread as `0`.
- Slanted / low-light / glare-heavy photos reduce accuracy.

[Unreleased]: https://example.invalid/compare/v0.1.0-pre.1...HEAD
[v0.1.0-pre.1]: https://example.invalid/releases/tag/v0.1.0-pre.1
