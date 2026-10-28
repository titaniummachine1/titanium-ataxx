path = 'C:/gitProjects/main-ataxx/crates/titanium/src/search.rs'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update deadline_ms to include 10ms safety buffer
t1 = """        self.deadline_ms = limits
            .time
            .map(|t| self.start_ms + t.as_secs_f64() * 1000.0);"""

r1 = """        let safety_margin_ms = 10.0;
        self.deadline_ms = limits
            .time
            .map(|t| self.start_ms + (t.as_secs_f64() * 1000.0 - safety_margin_ms).max(1.0));"""

# 2. Update iteration prediction check (remove !BZ so Botzone also predicts and doesn't overrun)
t2 = """            // Botzone variant: no early exit. The clock is a hard limit there, so use all of it;
            // a deeper iteration cut by the deadline still contributes its completed root moves
            // (partial-iteration adoption above). Classic search keeps the prediction rule unless
            // defending a forced loss, where all available time is used to maximize delay.
            if !BZ && result.score > -MATE_BOUND && self.elapsed_ms() + proj_ms > budget_ms {
                self.stop_reason = StopReason::Time;
                break;
            }"""

r2 = """            // Iteration cost prediction: project if the next iteration would overrun the budget.
            // On Botzone, overrunning by even 1 ms deducts time from future moves and risks time forfeit.
            // We stop cleanly at the completed depth. Only when defending a forced loss do we use all
            // remaining time to find delay paths.
            if result.score > -MATE_BOUND && self.elapsed_ms() + proj_ms > budget_ms {
                self.stop_reason = StopReason::Time;
                break;
            }"""

for target, rep in [(t1, r1), (t2, r2)]:
    tc = target.replace('\r\n', '\n').replace('\n', '\r\n')
    rc = rep.replace('\r\n', '\n').replace('\n', '\r\n')
    if tc in content:
        content = content.replace(tc, rc)
        print("Replaced CRLF")
    elif target in content:
        content = content.replace(target, rep)
        print("Replaced LF")
    else:
        print("Target not found:", target[:40])

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated search.rs!")
