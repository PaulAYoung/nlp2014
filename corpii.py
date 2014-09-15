#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module is here to make it easy load various text corpuses into other
scripts.
"""

from os import path

from nltk.corpus.reader import PlaintextCorpusReader

pwd = path.curdir


def load_pres_debates():
    """
    Returns the corpus for the presidential debates.
    """
    debates = PlaintextCorpusReader(path.join(pwd, "pres_debates"), ".*.txt")
    return debates
