"""The content-side transfer experiment, on the parts that can silently rot.

The headline of that experiment is a confound: a classifier reaches 0.889 on
the collected posts by reading the link rather than the claim. That number is
only meaningful if two things hold, and both are easy to break by accident
later, so they are pinned here.
"""

from factnet.nlp.ood_eval import _domain_split, strip_links


def test_strip_links_removes_what_gives_the_label_away():
    """A label read off the outlet must not survive in the text as a token."""
    cases = [
        ("Protesters target hotel https://rt.com/news/12345 alleged to be", "rt.com"),
        ("Report says otherwise rt.com/news/x", "rt.com"),
        ("See www.gatewaypundit.com for more", "gatewaypundit"),
        ("Source: apnews.com/article/trump-ethics", "apnews"),
    ]
    for text, giveaway in cases:
        assert giveaway not in strip_links(text).lower(), text


def test_strip_links_keeps_the_sentence():
    text = "The Ohio Supreme Court nullified the judgement, apnews.com/article/x"
    kept = strip_links(text)
    assert "Ohio Supreme Court" in kept
    assert "nullified" in kept


def test_domain_split_holds_outlets_out_entirely():
    """No outlet may appear on both sides, or the split measures memorisation."""
    rows = [{"source_domain": f"d{i % 9}.com", "label": i % 2, "text": "x"}
            for i in range(90)]
    for seed in (0, 1, 2):
        train, test = _domain_split(rows, seed)
        assert train and test
        assert not ({r["source_domain"] for r in train}
                    & {r["source_domain"] for r in test})


def test_export_does_not_publish_the_label_beside_the_question():
    """The shared sample must not carry the outlet the label was read from."""
    from factnet.ingestion.anonymise import scrub_mentions
    from factnet.nlp.ood_eval import URLISH

    posts = [
        'Protesters shattered windows. www.dailymail.co.uk/news/article...',
        'Report: https://rt.com/news/12345 claims otherwise',
        'See ria.ru/20250426/kursk for the statement',
        'Source apnews.com/article/trump-ethics-rules',
        'WHO says so www.who.int/publications/i/item/x',
    ]
    for post in posts:
        assert not URLISH.search(scrub_mentions(post)), post


def test_influence_ranking_does_not_depend_on_its_weights():
    """Both papers claim the ordering survives reweighting. It has to stay true.

    The claim is about the merged network of collected cascades, which is dense
    because accounts recur across cascades. A tree would not test it: k-core is
    zero everywhere on a tree, so two of the three terms vanish and the score
    reduces to a weighted sum of two measures that genuinely do trade off.
    """
    import networkx as nx
    from scipy.stats import spearmanr

    from factnet.graph.influence import _normalise

    # several overlapping cascades, so accounts recur and a core exists
    g = nx.DiGraph()
    for root in range(3):
        for child in range(10, 26):
            g.add_edge(root, child)
    for a in range(10, 22):
        g.add_edge(a, a + 4 if a + 4 < 26 else 10)
    assert max(nx.core_number(nx.Graph(g)).values()) > 1, "graph has no core"

    pr = _normalise(nx.pagerank(g))
    core = _normalise(nx.core_number(nx.Graph(g)))
    reach = _normalise({n: len(nx.descendants(g, n)) for n in g})
    nodes = list(g)

    def score(a, b, c):
        return {n: a * reach[n] + b * pr[n] + c * core[n] for n in g}

    reference = score(0.5, 0.3, 0.2)
    top = set(sorted(reference, key=reference.get, reverse=True)[:8])
    for weights in ((0.4, 0.4, 0.2), (0.6, 0.2, 0.2), (0.34, 0.33, 0.33),
                    (0.7, 0.2, 0.1)):
        other = score(*weights)
        rho = spearmanr([reference[n] for n in nodes],
                        [other[n] for n in nodes]).statistic
        overlap = top & set(sorted(other, key=other.get, reverse=True)[:8])
        assert rho > 0.98, (weights, rho)
        assert len(overlap) >= 7, (weights, len(overlap))
