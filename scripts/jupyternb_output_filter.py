#!/usr/bin/env python3
"""
jq 'del(.cells[] ["id", "outputs", "execution_count", "output_type"])'

"""

import json
import sys

if __name__ == "__main__":
    nb = json.load(sys.stdin)
    for cell in nb["cells"]:
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
        if "output_type" in cell:
            del cell["output_type"]
        if "id" in cell:
            del cell["id"]
        if ("metadata" in cell) and ("tags" in cell["metadata"]):
            cell["metadata"] = {"tags": cell["metadata"]["tags"]}
        else:
            cell["metadata"] = {}

    json.dump(nb, sys.stdout, sort_keys=True, indent=1)
