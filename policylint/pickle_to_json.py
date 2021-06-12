#!/usr/bin/env python3

import json
import pickle
import sys

with open(sys.argv[1], "rb") as fin:
    data = pickle.load(fin)
    print(json.dumps(data, indent=2))
