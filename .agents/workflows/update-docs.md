---
description: Update project documentation (README, ADRs, roadmap) after implementation changes
---

# Documentation Update Workflow

After completing any implementation work on the poker AI project, follow these steps:

## 1. Update README.md Roadmap
- Check `README.md` roadmap table
- Update phase statuses: 🔲 (not started), 🔨 (in progress), ✅ (complete)
- If project structure changed (new folders/files), update the directory tree

## 2. Check for New Architecture Decisions
- If a design decision was made during this session (new approach, rejected alternative, trade-off chosen):
  - Create a new ADR in `docs/adr/NNN-short-title.md`
  - Update `docs/adr/README.md` index table with the new entry
- ADR format: Context → Decision → Consequences → Alternatives Considered

## 3. Update requirements.txt
- If new Python dependencies were installed, add them to `requirements.txt`
- Keep Phase 2+ dependencies commented until actually needed

## 4. Quick Sanity
// turbo
- Run `source venv/bin/activate && python -m pytest tests/ -v` to confirm nothing is broken
