#!/usr/bin/env python3
"""
Calculating similarity metric between songs using chromaprint

Install chromaprint at https://github.com/acoustid/chromaprint
According to https://oxygene.sk/2011/01/how-does-chromaprint-work/
"""
import argparse
import json
import subprocess
from functools import lru_cache, partial
from operator import sub
from typing import Dict, List, Tuple

from Bio.pairwise2 import align


def get_grey_code(n: int) -> List[str]:
    """
    Get n-bit grey code
    """
    if n <= 0:
        raise ValueError("must be positive integer")

    if n == 1:
        return ['0', '1']

    l = get_grey_code(n - 1)
    l1 = ['0' + e for e in l]
    l2 = ['1' + e for e in l[::-1]]

    return l1 + l2


GREY_CODE: Dict[int, int] = {int(c, 2): i for i, c in enumerate(get_grey_code(2))}


def get_fingerprint(p):
    """
    Use chromaprint to calculate fingerprint from file path
    """
    out = subprocess.run(['fpcalc', '-json', '-raw', p], capture_output=True)
    return json.loads(out.stdout)


def get_chunk(a: int, i: int) -> int:
    return GREY_CODE[(a >> (2 * i)) & 0b11]


@lru_cache(maxsize=2048)
def get_chunks(a: int) -> Tuple[int]:
    """
    Splitting integer into 16 2-bit chunks
    """
    return tuple(map(partial(get_chunk, a), range(16)))


def match_score(a: int, b: int) -> int:
    """
    alignment match score calculator

    Get absolute difference of grey code between chunks. <8 is match and >40 is mismatch.
    No penalty otherwise. Theoretical minimum in grey difference is 0, maximum is 48.
    """
    score = sum(map(abs, map(sub, get_chunks(a), get_chunks(b))))

    if score < 16:
        return 1
    if score > 32:
        return -1
    return 0


def similarity_score(fp1: List[int], fp2: List[int]) -> float:
    """
    Get similarity score between fingerprints

    Use global alignment to align fingerprints, with custom match score calculator.
    """
    aln = align.globalcs(fp1, fp2, match_score, -2, 0, gap_char=[627964279])[0]
    return aln[2] / aln[4]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs=2, help="files to compair")

    args = parser.parse_args()

    data1, data2 = map(get_fingerprint, args.file)

    score = similarity_score(data1["fingerprint"], data2["fingerprint"])
    print(f"Similarity score: {score}")
