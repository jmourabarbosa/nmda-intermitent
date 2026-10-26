"""Locations of the recordings and of the computed results."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the 25 sessions of Panichello et al. (2024), Dryad doi:10.5061/dryad.kkwh70sct (yymmdd.mat)
DATA = os.environ.get('NMDAINT_DATA', os.path.join(os.path.dirname(ROOT), 'panichello'))
RESULTS = os.environ.get('NMDAINT_RESULTS', os.path.join(ROOT, 'results'))
CACHE = os.environ.get('NMDAINT_CACHE', os.path.join(ROOT, 'cache'))     # large intermediate files, not committed


def results(*p):
    path = os.path.join(RESULTS, *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def cache(*p):
    path = os.path.join(CACHE, *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
