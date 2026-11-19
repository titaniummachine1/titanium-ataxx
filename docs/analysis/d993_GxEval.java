import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import spamalot.gataxx.board.Position;
import spamalot.gataxx.eval.Evaluator;
import spamalot.gataxx.eval.FeatureEvaluator;
import spamalot.gataxx.eval.PairEvaluator;
import spamalot.gataxx.uai.UaiEngine;

/** gataxx 2.0.0's default UAI static evaluation on a FEN file: columns full eval (base + pairs, blend 1.0), base only; cp. */
public class GxEval {
    public static void main(String[] a) throws Exception {
        FeatureEvaluator base = new FeatureEvaluator(new FeatureEvaluator.Weights(FeatureEvaluator.SCALE,
                UaiEngine.DEFAULT_EVAL_SAFE, 0, 0, UaiEngine.DEFAULT_EVAL_REACH, 0, UaiEngine.DEFAULT_EVAL_EDGE,
                UaiEngine.DEFAULT_TEMPO, 0, 0, 0, UaiEngine.DEFAULT_EVAL_COHESION, 0));
        Evaluator full = PairEvaluator.builtIn(base, 1.0);
        try (BufferedReader in = Files.newBufferedReader(Path.of(a[0]));
             BufferedWriter out = Files.newBufferedWriter(Path.of(a[1]))) {
            String line;
            while ((line = in.readLine()) != null) {
                Position p = Position.fromFen(line);
                out.write(UaiEngineCp.cp(full.evaluate(p)) + " " + UaiEngineCp.cp(base.evaluate(p)) + "\n");
            }
        }
    }
}

class UaiEngineCp {
    /** UaiEngine.toCentipieces without the mate branch: score * 100 / SCALE, rounded */
    static int cp(int s) {
        return Math.floorDiv(s * 100 + FeatureEvaluator.SCALE / 2, FeatureEvaluator.SCALE);
    }
}
