"""Tkinter desktop workspace backed by CareerFit CLI workflows."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def build_command(
    action, resumes, job, model, dataset, output, start, goal, algorithm
):
    args = [sys.executable, str(ROOT / "app.py")]

    if action == "search":
        return args + [
            "search",
            "--start", start,
            "--goal", goal,
            "--algorithm", algorithm,
            "--output", output,
        ]

    if action == "train":
        if not Path(dataset).is_file():
            raise ValueError("Select a dataset CSV.")

        return args + [
            "train",
            "--dataset", dataset,
            "--artifacts", str(Path(model).parent),
            "--output", output,
            "--seed", "42",
        ]

    if not resumes:
        raise ValueError("Select at least one resume.")

    if action == "parse":
        return args + ["parse", *resumes, "--output", output]

    if action not in ("match", "score", "growth", "fairness-audit"):
        raise ValueError("Unknown workflow.")

    if not Path(job).is_file():
        raise ValueError("Select a job description.")

    args += [
        action,
        "--job", job,
        "--resumes", *resumes,
        "--output", output,
    ]

    if action != "match":
        if not Path(model).is_file():
            raise ValueError("Select or train a model first.")
        args += ["--model", model]

    return args


def main():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText

    root = tk.Tk()
    root.title("CareerFit | Candidate insights")
    root.geometry("1120x820")
    root.minsize(850, 650)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TButton", padding=7)
    style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"))

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame, text="CareerFit", style="Title.TLabel"
    ).pack(anchor="w")

    ttk.Label(
        frame,
        text=(
            "Resume analysis · Explainable matching · "
            "Learning plans · Fairness review"
        ),
    ).pack(anchor="w", pady=(0, 12))

    ttk.Label(
        frame,
        text=(
            "Scores depend on the selected model and extracted resume "
            "evidence. Load only trusted model files."
        ),
    ).pack(anchor="w")

    model_info = tk.StringVar(
        value=(
            "Model details appear after a scoring run. "
            "Fit scores are not hiring probabilities."
        )
    )
    ttk.Label(
        frame, textvariable=model_info, wraplength=950
    ).pack(anchor="w", pady=4)

    fields = {}
    resumes = []
    resume_label = tk.StringVar(value="No resumes selected")

    def choose_resumes():
        selected = filedialog.askopenfilenames(
            filetypes=[("Resumes", "*.pdf *.docx *.txt")]
        )
        if selected:
            resumes[:] = selected
            resume_label.set(
                f"{len(resumes)} resume(s): "
                + ", ".join(Path(p).name for p in resumes)
            )

    row = ttk.Frame(frame)
    row.pack(fill="x", pady=6)

    ttk.Button(
        row, text="Select resumes", command=choose_resumes
    ).pack(side="left")

    ttk.Label(
        row, textvariable=resume_label, wraplength=800
    ).pack(side="left", padx=10)

    defaults = {
        "Job description": ROOT / "samples/sample_job_description.txt",
        "Model": ROOT / "output/demo_model/careerfit_role_model.joblib",
        "Dataset": ROOT / "samples/demo_resume_dataset.csv",
        "Output": ROOT / "output/desktop_result.json",
    }

    for label, default in defaults.items():
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=3)

        ttk.Label(row, text=label, width=17).pack(side="left")

        value = tk.StringVar(value=str(default))
        fields[label] = value

        ttk.Entry(
            row, textvariable=value
        ).pack(side="left", fill="x", expand=True)

        def browse(v=value, name=label):
            if name == "Output":
                path = filedialog.asksaveasfilename(
                    defaultextension=".json"
                )
            else:
                path = filedialog.askopenfilename()

            if path:
                v.set(path)

        ttk.Button(
            row, text="Browse", command=browse
        ).pack(side="left", padx=(6, 0))

    options = ttk.Frame(frame)
    options.pack(fill="x", pady=8)

    action = tk.StringVar(value="score")

    ttk.Label(options, text="Workflow").pack(side="left")

    ttk.Combobox(
        options,
        textvariable=action,
        state="readonly",
        width=17,
        values=[
            "parse", "match", "train", "score",
            "growth", "fairness-audit", "search",
        ],
    ).pack(side="left", padx=8)

    start = tk.StringVar(value="python")
    goal = tk.StringVar(value="scikit_learn")
    algorithm = tk.StringVar(value="astar")

    for label, variable in [
        ("Start skill", start),
        ("Goal skill", goal),
    ]:
        ttk.Label(options, text=label).pack(side="left", padx=4)
        ttk.Entry(
            options, textvariable=variable, width=15
        ).pack(side="left")

    ttk.Combobox(
        options,
        textvariable=algorithm,
        state="readonly",
        width=8,
        values=["bfs", "dfs", "ucs", "astar", "all"],
    ).pack(side="left", padx=5)

    status = tk.StringVar(
        value="Ready. Select resumes, choose a workflow and run."
    )
    ttk.Label(
        frame, textvariable=status
    ).pack(anchor="w", pady=5)

    notebook = ttk.Notebook(frame)
    notebook.pack(fill="both", expand=True)

    table = ttk.Treeview(
        notebook,
        columns=("name", "rank", "score", "eligible", "reasons"),
        show="headings",
    )

    headings = [
        ("name", "Candidate"),
        ("rank", "Rank"),
        ("score", "Fit score"),
        ("eligible", "Meets parsed requirements"),
        ("reasons", "Requirement failures (select row for detail)"),
    ]

    for key, title in headings:
        table.heading(key, text=title)
        table.column(key, width=170)

    table.column("reasons", width=360)
    notebook.add(table, text="Ranking")

    details = ScrolledText(
        notebook, wrap="word", font=("Consolas", 10)
    )
    notebook.add(details, text="Full report / log")

    candidate_details = ScrolledText(
        notebook, wrap="word", font=("Segoe UI", 11)
    )
    notebook.add(candidate_details, text="Candidate details")

    candidates = {}

    def inspect_candidate(event=None):
        selection = table.selection()
        if not selection:
            return

        item = candidates.get(selection[0], {})
        lines = [
            item.get("candidate_name", "Candidate"),
            "Fit score: " + str(item.get("fit_score", "")),
            "Requirement failures based on extracted evidence:",
        ]

        failures = item.get("failed_requirements", [])
        lines.extend("- " + reason for reason in failures)

        if not failures:
            lines.append("None detected.")

        lines += [
            "",
            "Matched skills: "
            + ", ".join(item.get("matched_skills", [])),
            "Missing or undetected skills: "
            + ", ".join(item.get("missing_skills", [])),
            "",
            "Warnings:",
        ]
        lines += item.get("warnings", [])
        lines += [
            "",
            "Review extraction against the original resume. "
            "Undetected evidence is not proof that a qualification is absent.",
        ]

        candidate_details.delete("1.0", "end")
        candidate_details.insert("end", "\n".join(lines))

    table.bind("<<TreeviewSelect>>", inspect_candidate)
    table.bind(
        "<Double-1>",
        lambda event: notebook.select(candidate_details),
    )

    state = {
        "process": None,
        "log": None,
        "output": None,
    }

    def show(text):
        details.delete("1.0", "end")
        details.insert("end", text)

    def poll():
        proc = state["process"]
        if proc is None:
            return

        code = proc.poll()
        if code is None:
            root.after(150, poll)
            return

        state["log"].seek(0)
        log = state["log"].read()
        state["log"].close()
        state["process"] = None

        run.configure(state="normal")
        visual.configure(state="normal")

        if code:
            show(log)
            notebook.select(details)
            status.set("Failed. See log for details.")
            return

        try:
            report = json.loads(
                Path(state["output"]).read_text(encoding="utf-8")
            )
            show(json.dumps(report, indent=2, ensure_ascii=False))

            for item_id in table.get_children():
                table.delete(item_id)

            candidates.clear()
            candidate_details.delete("1.0", "end")

            ranking = report.get("ranking", [])

            for item in ranking:
                key = table.insert(
                    "",
                    "end",
                    values=(
                        item.get("candidate_name", ""),
                        item.get("rank", ""),
                        f"{item.get('fit_score', 0):.2%}",
                        item.get("eligible", ""),
                        "; ".join(item.get("failed_requirements", [])),
                    ),
                )
                candidates[key] = item

            context = report.get("scoring_context", {})
            selected_model = (
                ranking[0].get("model_name", "See full report")
                if ranking else "See full report"
            )
            confidence = context.get("job_role_confidence")

            model_info.set(
                f"Model: {selected_model} | "
                f"File: {Path(fields['Model'].get()).name} | "
                f"Target: {context.get('target_role', 'N/A')} | "
                f"Job-role confidence: "
                f"{confidence if confidence is not None else 'N/A'}\n"
                + " ".join(context.get("warnings", []))
            )

            notebook.select(table if ranking else details)
            status.set(
                f"Completed. Report saved: {state['output']}"
            )

        except (OSError, ValueError) as exc:
            show(log + "\n" + str(exc))
            status.set(
                "Process finished; report could not be displayed."
            )

    def launch(is_visual=False):
        if state["process"] is not None:
            return

        try:
            selected = "search" if is_visual else action.get()
            output = str(
                Path(fields["Output"].get()).expanduser().resolve()
            )

            if Path(output).exists():
                if not messagebox.askyesno(
                    "Replace report?",
                    f"Replace existing report?\n{output}",
                ):
                    return

            cmd = build_command(
                selected,
                resumes,
                fields["Job description"].get(),
                fields["Model"].get(),
                fields["Dataset"].get(),
                output,
                start.get(),
                goal.get(),
                algorithm.get(),
            )

            if selected == "train" and Path(fields["Model"].get()).exists():
                if not messagebox.askyesno(
                    "Replace trained model?",
                    "Training writes model and evaluation files in "
                    "the selected model folder. Continue?",
                ):
                    return

            if is_visual:
                cmd.append("--visualize")

            log = tempfile.TemporaryFile(
                mode="w+", encoding="utf-8"
            )

            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception:
                log.close()
                raise

            state.update(
                process=process,
                log=log,
                output=output,
            )

            run.configure(state="disabled")
            visual.configure(state="disabled")
            status.set(
                "Running… Close the Pygame window to finish "
                "a visual replay."
            )
            root.after(150, poll)

        except (ValueError, OSError) as exc:
            messagebox.showerror("Cannot run", str(exc))

    # FIX: allocate space for the buttons BEFORE the expanding notebook.
    # This keeps the results panel from compressing the button labels.
    buttons = ttk.Frame(frame)
    buttons.pack(
        side="bottom",
        fill="x",
        pady=(10, 0),
        before=notebook,
    )

    run = ttk.Button(
        buttons,
        text="Run workflow",
        command=launch,
    )
    run.pack(side="left")

    visual = ttk.Button(
        buttons,
        text="Open search visualization",
        command=lambda: launch(True),
    )
    visual.pack(side="left", padx=8)

    def cancel():
        if state["process"] is not None:
            state["process"].terminate()

    ttk.Button(
        buttons,
        text="Cancel running task",
        command=cancel,
    ).pack(side="left")

    def close():
        if state["process"] is not None:
            if not messagebox.askyesno(
                "Task running",
                "Stop the running task and close?",
            ):
                return

            state["process"].terminate()
            state["process"].wait()
            state["log"].close()

        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())