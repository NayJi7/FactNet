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
