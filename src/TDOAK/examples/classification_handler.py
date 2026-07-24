# classification_handler.py — Handler DAT for the 'classes' stream of
# classification_pipeline.py.
#
# fill-only: the built-in Classifications decode (ClassificationEntry dicts) stays,
# this just writes the top results into a Table DAT. Bind the Streams block's Scriptop
# to a Table DAT — DATs are pushed automatically each new frame, no cook callback needed.
# Rows: label | score, sorted by score, one header row. Render text from the table
# however you like (the data stays data).

TOP_N = 5


def fill(dat, classes) -> None:
    """MAIN THREAD — classes: list of {'label': str, 'score': float}, sorted by score."""
    if not isinstance(classes, list):
        return  # block bound to a non-classification stream — ignore rather than error
    dat.clear()
    dat.appendRow(['label', 'score'])
    for entry in classes[:TOP_N]:
        dat.appendRow([entry['label'], f"{entry['score']:.3f}"])
