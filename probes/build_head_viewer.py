"""Build a local artifactbin viewer from identity-head game predictions."""
import argparse
import json
from pathlib import Path

import chess


def build(source, destination):
    reports = json.loads(source.read_text())
    models = {"head-warmup-joint": "Jointly trained model", "head-warmup": "Frozen champion"}
    pieces, states = [], []
    for arm, label in models.items():
        for row in reports[arm]["timeline"]:
            ply = row["ply"]
            for piece in row["pieces"]:
                symbol = chess.Piece.from_symbol(piece["piece"])
                pieces.append({**piece, "model": label, "ply": ply,
                               "glyph": symbol.unicode_symbol(),
                               "name": ("White " if symbol.color else "Black ") + chess.piece_name(symbol.piece_type),
                               "status": "Captured" if not piece["alive"] else "Correct" if piece["correct"] else "Misplaced"})
            if arm == "head-warmup-joint":
                after = (f"{(ply + 1) // 2}{'.' if ply % 2 else '...'}{row['last_move']}" if ply else "Starting position")
                next_move = (f"{ply // 2 + 1}{'.' if ply % 2 == 0 else '...'}{row['next_move']}" if row["next_move"] else "Game over: White wins by checkmate")
                states.append({"ply": ply, "label": after if not ply else f"After {after}",
                               "next_move": next_move, "side": row["side_to_move"], "fen": row["fen"]})
    final_ply = states[-1]["ply"]
    assert [row["ply"] for row in states] == list(range(final_ply + 1))
    squares = [{"square": chess.square_name(chess.square(file, rank)), "ordinal": rank * 8 + 7 - file}
               for rank in range(8) for file in range(7, -1, -1)]

    def value(name, rows):
        return f'<Value name="{name}" type="table" value={{{json.dumps(rows, separators=(",", ":"))}}} />'

    css = """
.head-board{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));width:100%;max-width:530px;margin:0 auto;border:1px solid var(--border)}
.head-square{aspect-ratio:1;position:relative;display:flex;align-items:center;justify-content:center;min-width:0;background:var(--background);padding:6px}
.head-square:nth-child(16n+2),.head-square:nth-child(16n+4),.head-square:nth-child(16n+6),.head-square:nth-child(16n+8),.head-square:nth-child(16n+9),.head-square:nth-child(16n+11),.head-square:nth-child(16n+13),.head-square:nth-child(16n+15){background:color-mix(in srgb,var(--muted-foreground) 30%,var(--background))}
.head-coordinate{position:absolute;left:3px;bottom:1px;font-family:var(--font-mono);font-size:9px;opacity:.7}
.head-glyph{font-family:Georgia,"DejaVu Sans",serif;font-size:clamp(23px,4vw,46px);line-height:1;color:var(--foreground);text-align:center;overflow-wrap:anywhere}
.head-crowded{font-size:clamp(14px,2.3vw,27px)}
.head-ghost{opacity:.35;font-size:clamp(13px,2vw,22px)}
.head-error{position:absolute;inset:0;box-shadow:inset 0 0 0 2px var(--destructive);pointer-events:none}
.head-conflict{position:absolute;right:3px;top:1px;font:10px var(--font-mono);color:var(--destructive)}
.head-readout{font-family:var(--font-mono);font-size:27px;font-variant-numeric:tabular-nums}
"""
    parts = ["---\ntitle: nanoDanya | New piece head, move by move\ntemplate: dashboard\ntheme: industry\ncolorMode: light\nvisibility: unlisted\n---",
             '<Helmet><title>nanoDanya | New piece head, move by move</title>', '<style>{`' + css + '`}</style>',
             '<Value name="ply" type="number" default={71} />',
             '<Value name="model" type="string" default="Jointly trained model" />',
             '<Value name="show_captured" type="boolean" default={false} />',
             '<Value name="white_bottom" type="boolean" default={false} />',
             value("pieces_data", pieces), value("states_data", states), value("squares_data", squares)]
    queries = {
        "move_options": "select ply, label from states_data order by ply",
        "state": "select * from states_data where ply=$ply",
        "current_pieces": "select * from pieces_data where ply=$ply and model=$model",
        "visible_pieces": "select * from current_pieces where alive or $show_captured",
        "actual_cells": """select s.square, s.ordinal,
          coalesce(string_agg(p.glyph, ' '), '') glyphs,
          coalesce(bool_or(not p.correct), false) misplaced
          from squares_data s left join current_pieces p on p.actual=s.square and p.alive
          group by s.square,s.ordinal order by case when $white_bottom then -s.ordinal else s.ordinal end""",
        "predicted_cells": """select s.square, s.ordinal,
          coalesce(string_agg(p.glyph, ' ' order by p.identity) filter(where p.alive), '') glyphs,
          coalesce(string_agg(p.glyph, ' ' order by p.identity) filter(where not p.alive), '') ghosts,
          count(p.identity)>1 collision, count(p.identity) piece_count,
          coalesce(bool_or(p.alive and not p.correct),false) misplaced
          from squares_data s left join visible_pieces p on p.predicted=s.square
          group by s.square,s.ordinal order by case when $white_bottom then -s.ordinal else s.ordinal end""",
        "metrics": """select count(*) filter(where alive and correct) correct,
          count(*) filter(where alive) living,
          count(*) filter(where alive and not correct) misplaced,
          count(*) filter(where involved and correct) involved_correct,
          count(*) filter(where involved) involved_count from current_pieces""",
        "collisions": "select count(*) filter(where collision) collision_squares from predicted_cells",
        "details": """select name, identity, coalesce(actual,'Captured') actual, predicted,
          predicted_probability, actual_probability, status,
          case when involved then 'Yes' else '' end next_move_target
          from visible_pieces order by alive desc,correct,identity""",
        "game_scores": """select s.ply,s.label,
          count(*) filter(where p.alive and p.correct) recovered,
          count(*) filter(where p.alive) living,
          count(*) filter(where p.involved and p.correct) involved_correct,
          count(*) filter(where p.involved) involved_count
          from states_data s join pieces_data p on p.ply=s.ply and p.model=$model
          group by s.ply,s.label order by s.ply""",
    }
    parts += [f'<Query name="{name}">{{`{sql}`}}</Query>' for name, sql in queries.items()]
    parts += ['</Helmet>', '<div data-design="tw" className="@container bg-background text-foreground px-4 py-5 @2xl:px-8">',
              '<header className="border-b border-border pb-3"><p className="font-mono text-xs uppercase tracking-widest text-primary">nanoDanya · GZrvqz3j · Identity-location head</p><h1 className="mt-2 text-lg font-medium">Replay the game through the new piece head.</h1><p className="mt-1 text-sm text-muted-foreground">Every White and Black move, from the initial board through 37.Rh2#. Each prediction uses only moves already played.</p></header>',
              '<div className="flex flex-wrap items-end gap-4 py-3"><Select label="Position" value="$ply" options="$move_options" /><Segmented label="Model" value="$model" options={["Jointly trained model","Frozen champion"]} /></div>',
              '<p className="font-mono text-xs text-muted-foreground">Drag the move slider, or focus it and use the left/right arrow keys to step through the game.</p>',
              f'<div className="py-3"><label htmlFor="move-slider" className="font-mono text-xs">Half-moves played: {{$ply}}</label><input id="move-slider" aria-label="Half-moves played" type="range" value="$ply" min={{0}} max={{{final_ply}}} step={{1}} className="w-full" /></div>',
              '<For each={$state} keyBy="ply"><div className="flex flex-wrap gap-4 font-mono text-sm pb-3"><span>{$_row.label}</span><span className="text-muted-foreground">Next: {$_row.next_move}</span></div></For>',
              '<div className="flex flex-wrap items-center gap-5 pb-3"><Switch label="Show captured identities (raw 32 outputs)" checked="$show_captured" /><Switch label="White at bottom" checked="$white_bottom" /></div>',
              '{!$show_captured && (<p className="text-sm text-muted-foreground pb-3">Captured pieces are hidden using the game record. The head supplies locations only; it does not decide which pieces are alive.</p>)}',
              '{$show_captured && (<p className="text-sm text-muted-foreground pb-3">All 32 location outputs are shown. Faded pieces were captured according to the game record. The head has no captured-state output.</p>)}',
              '<Grid mode="flow"><GridItem w={4}><div className="border border-border p-3"><p className="font-mono text-xs text-muted-foreground">LIVING PIECES LOCATED CORRECTLY</p><p className="head-readout"><Number data="$metrics" col="correct" format=".0f" /> / <Number data="$metrics" col="living" format=".0f" /></p></div></GridItem><GridItem w={4}><div className="border border-border p-3"><p className="font-mono text-xs text-muted-foreground">MISPLACED LIVING PIECES</p><p className="head-readout"><Number data="$metrics" col="misplaced" format=".0f" /></p></div></GridItem><GridItem w={4}><div className="border border-border p-3"><p className="font-mono text-xs text-muted-foreground">SQUARES WITH MULTIPLE PREDICTIONS</p><p className="head-readout"><Number data="$collisions" col="collision_squares" format=".0f" /></p></div></GridItem></Grid>',
              '<section className="mt-4"><Grid mode="flow">']
    for name, title in [("actual_cells", "Actual board"), ("predicted_cells", "New head's predicted locations")]:
        parts.append(f'<GridItem w={{6}}><div className="border border-border p-3"><h2 className="font-mono text-sm mb-3">{title}</h2><For each={{${name}}} keyBy="square" className="head-board"><div className="head-square"><span className="head-coordinate">{{$_row.square}}</span>')
        if name == "predicted_cells":
            parts += ['{$_row.collision ? (<span className="head-glyph head-crowded">{$_row.glyphs}<span className="head-ghost">{$_row.ghosts}</span></span>) : (<span className="head-glyph">{$_row.glyphs}<span className="head-ghost">{$_row.ghosts}</span></span>)}',
                      '{$_row.collision && (<span className="head-conflict">{$_row.piece_count} pieces</span>)}']
        else:
            parts.append('<span className="head-glyph">{$_row.glyphs}</span>')
        parts += ['{$_row.misplaced && (<span className="head-error" />)}', '</div></For></div></GridItem>']
    parts += ['</Grid></section>', '<p className="mt-3 text-sm text-muted-foreground">Outlined squares mark misplaced living pieces, at their actual and predicted squares. Multiple predictions on one square stay visible; no legal-board correction is applied. Empty squares earn no accuracy credit.</p>',
              '<section className="mt-5"><h2 className="font-mono text-sm mb-2">Piece by piece · errors first</h2><DataTable data="$details" columns={[{"col":"name","title":"Piece"},{"col":"identity","title":"Starting square"},{"col":"actual","title":"Actual location"},{"col":"predicted","title":"Head location"},{"col":"status","title":"Result"},{"col":"predicted_probability","title":"P(head location)","fmt":".1%"},{"col":"actual_probability","title":"P(actual location)","fmt":".1%"},{"col":"next_move_target","title":"Involved next move"}]} height="300px" /></section>',
              '<section className="mt-5"><h2 className="font-mono text-sm mb-2">Whole game · known living pieces</h2><DataTable data="$game_scores" columns={[{"col":"label","title":"Position"},{"col":"recovered","title":"Correct locations"},{"col":"living","title":"Living pieces"},{"col":"involved_correct","title":"Involved correct"},{"col":"involved_count","title":"Involved targets"}]} height="260px" /></section>',
              '<footer className="mt-5 border-t border-border pt-3 text-sm text-muted-foreground"><p>Frozen champion: original workspace2 15M checkpoint plus the head fitted for 1,000 updates. Jointly trained model: a further 1,000 updates to both model and head. Both heads use LR 1e-3; this game was excluded from training.</p><p className="mt-2">A head readout is not a literal model belief. Piece identities, types, capture status, and the actual board come from the game record; predicted squares come from the head. “Involved next move” is an evaluation label and never enters the prediction. Probabilities are uncalibrated head scores. These checkpoints differ from the own-game model used in the original viewer.</p><p className="mt-2"><a href="https://lichess.org/GZrvqz3j/black" className="text-primary underline">Original game</a> · dala-1300 vs nanoDanya · 19 September 2026</p></footer></div>']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts) + "\n")
    print(f"{destination}: {len(states)} positions, {len(models)} models, {len(pieces)} identity predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/probes/involved-pieces-20260921/head-game.json"))
    parser.add_argument("--out", type=Path, default=Path("output/artifactbin/piece-head-game.jsx"))
    args = parser.parse_args()
    build(args.source, args.out)
