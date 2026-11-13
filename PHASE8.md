# CareerFit desktop

Open this folder in VS Code. In PowerShell:

```powershell
& "C:\Users\swast\anaconda3\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --only-binary=:all: -r requirements.txt
python -m unittest discover -s tests -v
python app.py gui
```

Tkinter ships with standard Windows Python/Anaconda distributions; do not pip install tkinter. Linux may require the system Tk package.

Choose resumes with the file picker. The sample job, model and dataset paths are filled automatically. Choose Parse, Match, Train, Score, Growth or Fairness Audit, select an output JSON path and click Run workflow. Training and Search do not require resume selection. Parse does not require a job or model. Match uses requirements without ML. Training uses the selected model's directory and produces careerfit_role_model.joblib; select that filename afterwards if a different model name was previously chosen.

Ranking results appear in a table; complete reports appear in the Full report tab and are saved to JSON. Growth and fairness reports use that full report tab. Each job runs in a separate process so the interface stays responsive. Cancel terminates the active process; interrupted output files may need regeneration. Existing reports and model training replacements require confirmation.

For search, enter canonical skill names such as python and scikit_learn. Choose BFS, DFS, UCS, A* or all; use Run workflow for JSON, or Open search visualization for Pygame. Close each Pygame window to finish; all opens four sequential windows. See PHASE7.md for controls and search limitations.

The GUI calls the same tested CLI commands as earlier phases; report phase numbers describe the producing engine, so a score report remains phase 4 and a fairness report remains phase 6.

All eight implementation phases now exist. ML remains trained on synthetic demonstration data. Real dataset training/evaluation, subject-matter review of skill edges and representative fairness evaluation remain outstanding before operational hiring use. The GUI does not add evidence of real-world predictive performance.
