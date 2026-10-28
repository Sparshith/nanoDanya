import chess
import pytest

from benchmark.run import GameState, apply_model_choice, apply_prompt


@pytest.mark.parametrize('predicted', ['Bb5', 'Bb5+', 'Bb5#'])
def test_both_players_get_canonical_check_marker(predicted):
    board = chess.Board()
    board.push_san('e4')
    board.push_san('d6')
    tokens = {'<bos>': 0, 'e4': 1, 'd6': 2, 'Bb5': 3, 'Bb5+': 4, 'Bb5#': 5}
    state = GameState('test', 0, board, chess.WHITE,
                      {'model': [0, 1, 2], 'opponent': [0, 1, 2]}, ['e4', 'd6'])
    apply_model_choice(state, 'model', tokens[predicted], predicted,
                       {'model': tokens, 'opponent': tokens}, False)
    assert not state.finished
    assert state.prefixes['model'] == state.prefixes['opponent'] == [0, 1, 2, 4]
    assert state.moves == ['e4', 'd6', 'Bb5']
    assert state.board.is_check()


def test_prompt_reconstructs_check_marker_and_supports_stripped_vocab():
    board = chess.Board()
    tokenizers = {'canonical': {'<bos>': 0, 'e4': 1, 'd6': 2, 'Bb5+': 3},
                  'stripped': {'<bos>': 0, 'e4': 1, 'd6': 2, 'Bb5': 4}}
    prefixes = {name: [0] for name in tokenizers}
    apply_prompt(board, ['e4', 'd6', 'Bb5'], prefixes, tokenizers)
    assert prefixes == {'canonical': [0, 1, 2, 3], 'stripped': [0, 1, 2, 4]}


def test_stockfish_check_is_preserved_in_model_input(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import benchmark.run as bench

    tokens = {'<bos>': 0, '<eos>': 1, 'e4': 2, 'd6': 3, 'Bb5': 4, 'Bb5+': 5}
    engine = SimpleNamespace(play=lambda board, limit: SimpleNamespace(
        move=board.parse_san('e4' if board.fullmove_number == 1 else 'Bb5+')), quit=lambda: None)
    monkeypatch.setattr(bench.chess.engine.SimpleEngine, 'popen_uci', lambda *a, **kw: engine)
    monkeypatch.setattr(bench, 'configure_stockfish_strength', lambda *a: None)
    monkeypatch.setattr(bench, 'load_model', lambda *a: (None, None, tokens, {v: k for k, v in tokens.items()}))
    seen = []

    def choose(**kwargs):
        result = []
        for state in kwargs['states']:
            seen.append(state.prefixes['model'].copy())
            result.append((state, 3, 'd6') if len(state.moves) == 1 else (state, None, None))
        return result

    monkeypatch.setattr(bench, 'choose_for_states', choose)
    args = bench.build_parser().parse_args(['games', '--model', 'test', '--game-mode', 'stockfish',
        '--games', '1', '--game-offset', '1', '--max-plies', '4', '--device', 'cpu',
        '--output', str(tmp_path / 'game.jsonl')])
    bench.command_games(args)
    assert seen == [[0, 2], [0, 2, 3, 5]]
