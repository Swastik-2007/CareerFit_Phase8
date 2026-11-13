"""Convenient entry point for the CareerFit command-line application."""

from careerfit.cli import main
import sys


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'gui':
        from careerfit.gui import main as gui_main
        raise SystemExit(gui_main())
    if len(sys.argv) > 1 and sys.argv[1] == 'search':
        from careerfit.search import main as search_main
        raise SystemExit(search_main(sys.argv[2:]))
    raise SystemExit(main())

