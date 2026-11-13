# Phase 7 — Search and Pygame replay

All Phase 6 commands remain available. Use Python 3.13 and install requirements.txt.

```powershell
python -m unittest discover -s tests -v
python app.py search --start python --goal scikit_learn --algorithm all --output output/search_results.json
python app.py search --start python --goal scikit_learn --algorithm astar --visualize
```

Space pauses/resumes; Right advances a step; R restarts; Esc closes.
With `--algorithm all --visualize`, close each window to view the next algorithm.
JSON works without a display. Pygame needs a desktop session.

Algorithms: BFS minimizes edge count; DFS finds a deterministic route without an optimality guarantee; uniform-cost search minimizes entered-skill hours; A* uses reverse Dijkstra distances as an exact admissible heuristic. A* preprocessing visits reverse-reachable nodes and is reported separately. Its forward expansion count alone is not an overall speed comparison.

Costs exclude the start skill, assumed known. Edges point from prerequisite to target. A single route does not satisfy every prerequisite of a skill with several parents. Use the existing `growth` command for complete prerequisite plans. Search never changes candidate scores or decisions. Hour estimates retain Phase 5's heuristic status.

The window shows ancestors of the target plus the start skill, directed arrows, visited nodes in orange and the final route in green. JSON includes expansion order, route, hours and unreachable status. Unknown nodes and invalid costs produce errors.

Phase 8 (Tkinter GUI and final validation) remains. The ML dataset remains synthetic; real dataset evaluation is still outstanding.
