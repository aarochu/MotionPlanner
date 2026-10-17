# CLAUDE.md

Read `AGENTS.md` first: it has the project goal, the hardware and Viam setup, how the code is organized, the
decisions made so far, what is verified, and what is still open.

Rules that matter most here:

- Never move a robot arm without the user's explicit go-ahead. Read-only checks are fine.
- Never commit secrets: credentials live only in `env.sh` (gitignored).
- No AI co-author or attribution lines in commits or PRs.
- Machine config changes are live with no undo: ask first and record how to revert.
