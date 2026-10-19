import type { Plugin } from "@opencode-ai/plugin"

// Titanium Ataxx session guard: makes compaction preserve project memory
// instead of lobotomizing the next session. Fires before the LLM writes the
// continuation summary; injected context tells it what MUST survive.
export const TitaniumSessionGuard: Plugin = async (_ctx) => {
  return {
    "experimental.session.compacting": async (_input, output) => {
      output.context.push(`## Titanium Ataxx — must survive compaction
This project keeps its memory ON DISK (AGENTS.md auto-loads every session).
The continuation summary MUST preserve verbatim:
1. Current branch (from \`git branch --show-current\`) + HEAD commit hash.
2. Dirty files (from \`git status --short\`) — uncommitted work is state.
3. Strongest-build pointer: main @73a859d + tupS55tree.tup is STRONGEST; exp/* branches default to classical hand-eval (~-500 Elo).
4. The TOP HANDOFFS.md entry's key facts (what shipped, proof, caveats, held decisions a/b/c).
5. Any iron-law change the user stated this session (BEHAVIOR.md wording).
6. What is owed next (gates to extend, SPRT legs, unbuilt fronts).
Rules reminder for the resumed session: MAIN IS UNTOUCHABLE (exp/* only);
TIME CROWNS (5k = diagnosis, 100ms+200ms = crown); no gate stops below 100
games; gates are by seat 50/50; never reject on a dip; FOMO-parked nets are
never rejected. First action after resume: run /rules.`);
    },
  };
};
