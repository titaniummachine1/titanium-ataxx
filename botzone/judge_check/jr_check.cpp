// Reads queries "n x0 y0 x1 y1 ... (n moves)" from stdin, replays them through the reference Botzone
// judge (judge_ref.h from EmptyBlueBox/Ataxx, not copied into this repo) and prints
// "finish winner err errorturn" for the LAST move of each query. err: 0 ok, 1 illegal move, 2 repetition.
// Build: g++ -O2 -std=c++17 -I <ataxx_cpp>/src -o jr_check.exe jr_check.cpp
#include <array>
#include <cstdio>
#include <vector>

#include "judge_ref.h"

int main() {
  int n;
  while (scanf("%d", &n) == 1) {
    std::vector<std::array<int, 4>> mv(n);
    for (auto& m : mv) scanf("%d %d %d %d", &m[0], &m[1], &m[2], &m[3]);
    auto* j = new ataxx::JudgeRef();
    ataxx::Verdict v = j->run(mv);
    printf("%d %d %d %d\n", v.finish ? 1 : 0, v.winner, v.err, v.errorturn);
    fflush(stdout);
    delete j;
  }
  return 0;
}
