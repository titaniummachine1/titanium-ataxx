import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import spamalot.gataxx.board.Position;
import spamalot.gataxx.eval.RustHceEvaluator;

/** D1029: RustHceEvaluator (gataxx-ab) on a FEN file -> one line per FEN: HCE cp (Titanium scale) and gataxx score. */
public class d1029_RustHceParity {
    public static void main(String[] a) throws Exception {
        RustHceEvaluator e = RustHceEvaluator.load(a[0], a[1]);
        List<String> fens = Files.readAllLines(Path.of(a[2]));
        StringBuilder sb = new StringBuilder();
        for (String f : fens) {
            Position p = Position.fromFen(f);
            sb.append(String.format("%.2f %d%n", e.cp(p), e.evaluate(p)));
        }
        Files.writeString(Path.of(a[3]), sb.toString());
    }
}
