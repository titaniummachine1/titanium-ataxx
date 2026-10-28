path = 'C:/gitProjects/main-ataxx/crates/titanium/src/search.rs'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

target = """        result.best = Some(root_moves[0]);
        if let Some(entry) = root_tt {
            if entry.score.abs() < MATE_BOUND {
                result.score = entry.score;
            }
        }

        let budget_ms = limits
            .time
            .map(|t| t.as_secs_f64() * 1000.0)
            .unwrap_or(f64::INFINITY);
        // Iteration cost prediction: project the next iteration as the max
        // of the last two completed iterations.
        let mut it_ms: [f64; 2] = [0.0, 0.0];
        let mut has_completed_iter = false;

        // Dynamic iterative-deepening startup (Quoridor / persistent session):
        // If the prior search (or opponent move in persistent TT) left a deep entry
        // from the recent turn, skip the shallow iterations we already know the answer to
        // and resume from near that depth.
        let start_depth = if let Some(entry) = root_tt {
            let age_diff = self.tt.age.wrapping_sub(entry.age);
            if age_diff <= 2
                && entry.depth >= 3
                && (entry.flag == FLAG_EXACT || entry.flag == FLAG_LOWER)
                && entry.score.abs() < MATE_BOUND
            {
                (entry.depth.saturating_sub(1) as u8).min(limits.max_depth).max(1)
            } else {
                1
            }
        } else {
            1
        };

        for depth in start_depth..=limits.max_depth.max(1) {"""

replacement = """        // Initial fallback move: prioritize the proven TT move if legal, else first root move.
        let tt_legal_mv = if tt_mv != 0 {
            (0..n_root)
                .find(|&i| root_moves[i].to_u32() == tt_mv)
                .map(|i| root_moves[i])
        } else {
            None
        };
        result.best = tt_legal_mv.or(Some(root_moves[0]));
        if let Some(entry) = root_tt {
            if entry.score.abs() < MATE_BOUND {
                result.score = entry.score;
            }
        }

        let budget_ms = limits
            .time
            .map(|t| t.as_secs_f64() * 1000.0)
            .unwrap_or(f64::INFINITY);
        // Iteration cost prediction: project the next iteration as the max
        // of the last two completed iterations.
        let mut it_ms: [f64; 2] = [0.0, 0.0];
        let mut has_completed_iter = false;

        // Iterative deepening MUST always start at depth 1.
        // Skipping shallow iterations (even if TT has a deep entry) is fatal under hard time
        // limits (e.g. Botzone): if the high-depth iteration is cut short by the deadline before
        // completing, no iteration ever finishes, leaving depth at 0 and blundering with an
        // unverified move. Shallow iterations take <0.1% of search time and guarantee that a
        // verified completed iteration always exists.
        for depth in 1..=limits.max_depth.max(1) {"""

# Normalize newlines for matching
target_crlf = target.replace('\r\n', '\n').replace('\n', '\r\n')
replacement_crlf = replacement.replace('\r\n', '\n').replace('\n', '\r\n')

if target_crlf in content:
    content = content.replace(target_crlf, replacement_crlf)
    print("Replaced successfully with CRLF")
elif target in content:
    content = content.replace(target, replacement)
    print("Replaced successfully with LF")
else:
    print("Target not found!")

# Also update the test warm_iterative_deepening_skips_shallow_iterations
test_target = """    #[test]
    fn warm_iterative_deepening_skips_shallow_iterations() {
        let b = Board::start();
        let mut se = Searcher::with_tt_bits(16);
        // Initial search to depth 6
        let r1 = se.search(&b, &no_time(6));
        assert_eq!(r1.depth, 6);
        let nodes1 = r1.nodes;

        // In a persistent session, searching the same root position again
        // should start from depth 5 (tt_depth - 1), skipping depths 1..4.
        let r2 = se.search(&b, &no_time(6));
        assert_eq!(r2.depth, 6);
        assert!(r2.nodes < nodes1, "warm start nodes ({}) must be fewer than cold start ({})", r2.nodes, nodes1);
    }"""

test_replacement = """    #[test]
    fn warm_tt_reduces_search_nodes() {
        let b = Board::start();
        let mut se = Searcher::with_tt_bits(16);
        // Initial search to depth 6
        let r1 = se.search(&b, &no_time(6));
        assert_eq!(r1.depth, 6);
        let nodes1 = r1.nodes;

        // In a persistent session, searching the same root position again
        // starts from depth 1 but TT cutoffs massively prune the tree.
        let r2 = se.search(&b, &no_time(6));
        assert_eq!(r2.depth, 6);
        assert!(r2.nodes < nodes1, "warm start nodes ({}) must be fewer than cold start ({})", r2.nodes, nodes1);
    }"""

test_target_crlf = test_target.replace('\r\n', '\n').replace('\n', '\r\n')
test_replacement_crlf = test_replacement.replace('\r\n', '\n').replace('\n', '\r\n')

if test_target_crlf in content:
    content = content.replace(test_target_crlf, test_replacement_crlf)
    print("Replaced test successfully with CRLF")
elif test_target in content:
    content = content.replace(test_target, test_replacement)
    print("Replaced test successfully with LF")
else:
    print("Test target not found!")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated search.rs!")
