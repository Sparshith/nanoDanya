# DeepMind Searchless Chess comparison

Checked primary sources on 2026-09-19:

- [Paper, October 2024 version](https://arxiv.org/html/2402.04494v2)
- [Official code](https://github.com/google-deepmind/searchless_chess)
- [DeepMind's AlphaZero explanation](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/)

Searchless Chess trains supervised predictors from Stockfish labels. AlphaZero
is the self-play RL comparison and incorporates MCTS. The former gives the model
explicit board state and an action, supervising its value; our deployed model
predicts the next SAN token from history, then undergoes exact-reward fine-tuning.
These are materially different learning tasks and inference procedures.

The paper's sections 4 and B.8 report tactical mistakes and problems finishing
winning positions. It describes a Stockfish-assisted winning-position fallback
in its live-play setup. Its headline human blitz rating is not a demonstration
that such failures disappear, nor is it comparable directly to our rapid bot
rating. The paper does not run our mate-threat oversampling experiment, so it
cannot establish the cause or remedy for our observed transfer gap.

Inference for nanoDanya: do not read this work as evidence that more RL alone
will solve our current failure. It motivates separately testing richer value
supervision and explicit-state inputs; it does not justify changing the currently
authorized coverage experiment or adding an inference-time engine veto.

A separate, recent [August 2026 preprint](https://arxiv.org/abs/2608.27757)
from Lodz University of Technology studies self-play RL on an Lc0 Chessformer.
It reports a small tactical-accuracy gain with roughly preserved playing
strength, and a puzzle-only control that improves tactical metrics while losing
about 260 Elo. It is not DeepMind's original Searchless Chess study. Its
algorithm, strong initial policy and evaluation differ from nanoDanya's, so it
supports the need for separate game-strength measurement rather than prescribing
a drop-in fix. It explicitly compares both KL directions; use formulas when
comparing, because terminology can be ambiguous across implementations.
