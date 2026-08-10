"""Run one input through the whole system and record every stage.

The order of the stages is the order of the argument the project makes, not a
convenience. Whether the text states a checkable claim is settled before any
verdict is computed, because a verdict on a joke is noise dressed as a result.
Content and structure are then read separately, so that the integration stage
can show what the second adds to the first rather than asserting it. Influence
comes last and deliberately carries its own refutation, since the ranking is
structurally meaningful and diagnostically useless.

Nothing here renders. Every stage appends a ``Step`` carrying its numbers and
the data its figures need, and the interface decides how to draw them.
"""

from __future__ import annotations

from typing import Any

from factnet.nlp.checkworthy import explain as checkworthy_rules
from factnet.nlp.checkworthy import score as checkworthy_score
from factnet.serve import content as content_module
from factnet.serve import structure as structure_module
from factnet.serve.registry import by_key, primary
from factnet.serve.trace import Figure, Step, Trace, confidence_from

CHECKWORTHY_FLOOR = 0.3


def _parse_step(trace: Trace, text: str, data, handles, origin: str) -> bool:
    """Report what actually arrived, and whether the features can be trusted."""
    detail: dict[str, Any] = {"origin": origin, "has_text": bool(text.strip())}
    has_features = True
    if data is not None:
        measured = structure_module.shape(data)
        detail |= measured
        # six of the ten profile slots were unavailable from the endpoints the
        # collector used; a cascade whose features are all constant must not be
        # scored by a model that expects them to vary
        varying = int((data.x.std(dim=0) > 1e-9).sum())
        detail["varying_features"] = varying
        detail["total_features"] = int(data.x.size(1))
        has_features = varying >= 2
        if not has_features:
            trace.warn("Account features are constant on this cascade, so the "
                       "structure-only detector is used.")
    trace.add(Step(key="parse", title="What came in",
                   summary=(f"{detail.get('accounts', 0)} accounts, "
                            f"{detail.get('direct', 0)} direct shares"
                            if data is not None else "a text, with no cascade"),
                   detail=detail,
                   status="ok" if data is not None or text.strip() else "warning"))
    return has_features


def _checkworthy_step(trace: Trace, text: str) -> bool:
    value = checkworthy_score(text)
    rules = checkworthy_rules(text)
    passes = value >= CHECKWORTHY_FLOOR
    trace.add(Step(
        key="checkworthy", title="Is this a claim worth checking?",
        summary=(f"check-worthiness {value:.2f}, "
                 f"{'above' if passes else 'below'} the {CHECKWORTHY_FLOOR:.1f} floor"),
        detail={"score": round(value, 3), "floor": CHECKWORTHY_FLOOR,
                "rules": {k: bool(v) for k, v in rules.items()}},
        status="ok" if passes else "skipped",
        note="" if passes else
             "Opinions, questions and jokes state nothing that can be verified. "
             "A verdict is not produced for them, which is a decision rather "
             "than a failure."))
    return passes


def _content_step(trace: Trace, text: str, model_key: str,
                  advisory: bool = False) -> float | None:
    """Read the text. ``advisory`` means the reading is shown but not used.

    A post below the check-worthiness floor still gets read when a cascade is
    present, because refusing to display the reading would hide a judgement the
    system made rather than explain it. The number is reported and excluded, and
    the interface says which.
    """
    card = by_key(model_key)
    probability, extra = content_module.score(text, model_key)
    figures: list[Figure] = []

    if "attention" in extra:
        figures.append(Figure(
            kind="tokens", title="Where the model looked (attention)",
            data={"tokens": extra["attention"], "signed": False},
            caption="Last layer, averaged over heads. Attention describes the "
                    "computation; on its own it is not evidence that a token "
                    "mattered, which is why the next figure measures that directly."))
        occlusion = content_module.occlusion(text, card, probability)
        figures.append(Figure(
            kind="tokens", title="What actually moved the verdict (occlusion)",
            data={"tokens": occlusion, "signed": True},
            caption="Each token removed in turn. Positive means the token pushed "
                    "the verdict towards reliable."))
    if "contributions" in extra:
        figures.append(Figure(
            kind="bars", title="Exact contributions",
            data={"rows": [{"metric": t, "this": v} for t, v in extra["contributions"]],
                  "keys": ["this"]},
            caption="Weight times TF-IDF value: for a linear model the attribution "
                    "is exact and needs no approximation."))

    comparison, missing = content_module.compare(text)
    figures.append(comparison)
    if missing:
        trace.warn(f"Content models not available: {', '.join(missing)}.")

    trace.add(Step(
        key="content",
        title="What the text says" + (" (not counted)" if advisory else ""),
        summary=f"{card.name} puts p(reliable) at {probability:.3f}"
                + (", shown but excluded from the verdict" if advisory else ""),
        status="warning" if advisory else "ok",
        detail={"model": card.name, "model_key": card.key, "p_reliable": round(probability, 4),
                "counted": not advisory,
                "macro_f1": card.macro_f1, "trained_on": card.trained_on,
                "spread": comparison.data["spread"]},
        figures=figures,
        note=("This post did not pass the check-worthiness stage, so the reading "
              "below is reported for inspection and takes no part in the verdict. "
              "The rule set is a prototype and misses claims a reader would call "
              "checkable, which is why the reading is shown rather than withheld. "
              if advisory else "")
             + ("The models disagree by "
              f"{comparison.data['spread']:.2f} on this input, which is what a "
              "content ceiling near 0.63 looks like from the inside."
              if comparison.data["spread"] > 0.2 else "")))
    return probability


def _structure_step(trace: Trace, data, handles, origin: str, has_features: bool,
                    corpus: str, forced: str | None = None,
                    kinds: list[str] | None = None,
                    followers: list[int] | None = None) -> tuple[float, str]:
    checkpoint, key, why = structure_module.pick_checkpoint(origin, has_features, forced)
    probability = structure_module.verdict(data, checkpoint)
    measured = structure_module.shape(data)
    figures = [structure_module.graph_figure(data, handles, kinds, followers),
               structure_module.shape_figure(measured, corpus),
               structure_module.early_curve(data, checkpoint)]
    trace.add(Step(
        key="structure", title="How it travelled",
        summary=f"{by_key(key).name} puts p(reliable) at {probability:.3f}",
        detail={"model": by_key(key).name, "model_key": key,
                "p_reliable": round(probability, 4), "why_this_model": why} | measured,
        figures=figures))
    return probability, checkpoint


# each platform gets the pair trained on it: showing the ablation with a
# benchmark model on a Bluesky cascade would contradict the transfer result the
# same interface reports two steps earlier
ABLATION_PAIR = {
    "bluesky": ("bigcn-collected.pt", "bigcn-collected-score.pt"),
    "benchmark": ("bigcn-upfd-profile.pt", "bigcn-upfd-profile-score.pt"),
}


def _integration_step(trace: Trace, data, content_p: float,
                      origin: str) -> float | None:
    """The ablation, run on this cascade: the verdict with and without the text."""
    from factnet.graph.integration import attach_scores

    plain, scored_checkpoint = ABLATION_PAIR.get(
        "bluesky" if origin in ("bluesky", "url", "cascade") else "benchmark")
    scored = attach_scores([data], [content_p])[0]
    try:
        with_score = structure_module.verdict(scored, scored_checkpoint)
        without = structure_module.verdict(data, plain)
    except Exception:
        trace.warn("The integrated model does not accept this cascade's feature "
                   "layout, so the ablation is not shown.")
        return None
    trace.add(Step(
        key="integration", title="Does the text change the structural verdict?",
        summary=f"p(reliable) moves from {without:.3f} to {with_score:.3f} "
                f"when the content score is attached",
        detail={"without_score": round(without, 4), "with_score": round(with_score, 4),
                "delta": round(with_score - without, 4), "content_score": round(content_p, 4)},
        figures=[Figure(
            kind="bars", title="Ablation on this cascade",
            data={"rows": [{"metric": "structure alone", "this": round(without, 4)},
                           {"metric": "structure + content", "this": round(with_score, 4)}],
                  "keys": ["this"]},
            caption="The article measures this gain at +0.056 macro-F1 on PolitiFact "
                    "with light features, and nothing where the cascade already "
                    "carries text.")],
        note="This is a demonstration configuration: the published ablation is a "
             "macro-F1 over a whole test split, not a probability on one cascade."))
    return with_score


def _influence_step(trace: Trace, data, handles) -> None:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(data.edge_index.t().tolist())
    if graph.number_of_edges() == 0:
        return
    from factnet.graph.influence import influence_ranking
    top, reach, pagerank, coreness = influence_ranking(graph, top_k=10)
    rows = [{"account": (handles[n] if handles and n < len(handles) else f"A{n}"),
             "influence": round(s, 4), "reach": round(reach[n], 4),
             "pagerank": round(pagerank[n], 4), "k_core": round(coreness[n], 4)}
            for n, s in top]
    trace.add(Step(
        key="influence", title="Who carried it",
        summary=f"top {len(rows)} accounts by composite influence",
        detail={"formula": "0.5 reach + 0.3 PageRank + 0.2 k-core"},
        figures=[Figure(kind="table", title="Influence ranking",
                        data={"rows": rows,
                              "columns": ["account", "influence", "reach",
                                          "pagerank", "k_core"]},
                        caption="Structural centrality only.")],
        status="warning",
        note="Tested against the labelled sample, this ranking does not identify "
             "who spreads misinformation: the hundred most influential accounts "
             "carry 13.4 percent misleading content against a 14.5 percent "
             "chance baseline. High rank means central to diffusion, nothing more."))


# which reference averages a cascade should be read against, by where it came
# from: comparing a Bluesky cascade to PolitiFact averages would invite exactly
# the cross-platform confusion the project spent a phase demonstrating
CORPUS_FOR = {"bluesky": "bluesky", "cascade": "bluesky", "url": "bluesky",
              "benchmark": "politifact", "text": "politifact"}


def run(text: str = "", cascade: dict | None = None, data=None,
        content_model: str | None = None, origin: str = "text",
        corpus: str | None = None, graph_model: str | None = None,
        on_step=None) -> Trace:
    """The whole system on one input, with every stage recorded."""
    trace = Trace(input_kind=origin)
    if on_step is not None:
        trace.listen(on_step)
    corpus = corpus or CORPUS_FOR.get(origin, "politifact")
    handles = kinds = followers = None
    if cascade is not None:
        from factnet.ingestion.to_graph import cascade_to_pyg
        nodes = cascade["nodes"]
        order = sorted(range(len(nodes)), key=lambda i: nodes[i].get("kind") != "source")
        handles = [nodes[i].get("handle", "") for i in order]
        # the picture needs these too: without them every edge reads as a repost
        # and the largest audiences cannot be named
        kinds = structure_module.edge_kinds(cascade)
        followers = structure_module.node_followers(cascade, order)
        data = data if data is not None else cascade_to_pyg(cascade)
        text = text or cascade.get("text", "")

    content_model = content_model or primary("content").key
    has_features = _parse_step(trace, text, data, handles, origin)

    has_cascade = data is not None and data.num_nodes > 1
    content_p = None
    if text.strip():
        if _checkworthy_step(trace, text):
            content_p = _content_step(trace, text, content_model)
        elif not has_cascade:
            # nothing else to read: the input was a sentence stating no claim
            trace.label = "not a claim"
            trace.confidence = "high"
            trace.provenance = {"stopped_at": "check-worthiness"}
            return trace
        else:
            # the wording carries no checkable claim, but the cascade is still a
            # propagation object and the structural reading does not depend on it
            trace.warn("The text did not pass the check-worthiness stage, so its "
                       "content reading is shown for inspection only and takes no "
                       "part in the verdict.")
            _content_step(trace, text, content_model, advisory=True)

    structure_p = None
    if has_cascade:
        structure_p, _ = _structure_step(trace, data, handles, origin,
                                         has_features, corpus, graph_model,
                                         kinds, followers)
        if content_p is not None and has_features:
            combined = _integration_step(trace, data, content_p, origin)
            if combined is not None:
                structure_p = combined
        _influence_step(trace, data, handles)

    final = structure_p if structure_p is not None else content_p
    trace.verdict = final
    if final is not None:
        trace.label = "reliable" if final >= 0.5 else "misleading"
        trace.confidence = confidence_from(
            final, accounts=(data.num_nodes if data is not None else 0),
            has_features=has_features)
    trace.provenance = {"content": content_p, "structure": structure_p,
                        "content_model": content_model, "graph_model": graph_model,
                        "read_structure": data is not None}
    trace.add(Step(
        key="verdict", title="Verdict",
        summary=(f"{trace.label}, {trace.confidence} confidence"
                 if final is not None else "no verdict could be formed"),
        detail=trace.provenance | {"p_reliable": final},
        note="A research prototype trained on 400 collected cascades and two "
             "benchmarks. It is not a deployable classifier and its verdict "
             "should be read as an argument, not a ruling."))
    return trace
