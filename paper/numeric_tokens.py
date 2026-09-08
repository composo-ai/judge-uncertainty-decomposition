"""Numeric-token lister for results cross-checks (reviewer round 6 ask).

Prints every numeric token in main.tex's body (preamble and comments stripped)
plus a total, so "N tokens verified" claims are reproducible from the repo
instead of living in a session transcript.

The 27 Aug morning cross-check ("the cut moved no digit", commit 965cab5)
counted 138 tokens by a by-hand convention on the merged-main snapshot; that
convention was not committed, which is exactly the failure mode this file
ends. This script is the canonical convention from here on: future
cross-check claims should cite its count. To run it against a historical
snapshot:

    git show <ref>:paper/main.tex \
        > /tmp/snapshot.tex
    python3 numeric_tokens.py --file /tmp/snapshot.tex
"""

from __future__ import annotations

import argparse
import re


def tokens(path: str) -> list[str]:
    src = open(path).read()
    body = src.split(r"\begin{document}", 1)[-1]
    body = re.sub(r"(?<!\\)%.*", "", body)  # strip comments, keep \%
    return re.findall(r"\d+(?:[.,]\d+)*", body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="main.tex")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the total")
    args = parser.parse_args()
    toks = tokens(args.file)
    if not args.quiet:
        for tok in toks:
            print(tok)
    print(f"total numeric tokens: {len(toks)}")


if __name__ == "__main__":
    main()
