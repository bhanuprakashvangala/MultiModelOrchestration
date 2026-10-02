# Golden reproduction outputs

`reproduce_hashes.txt` holds the SHA-256 of the 13 files the flat-script reproduction (tag `v1.0.0`,
`python scripts/reproduce.py`) writes into `results/`: the eight CSVs and the five figures. `reproduce_stdout.txt`
is its console output (88/88 checks and the 100% keyword re-run).

They were recorded on Windows with CPython 3.12.10, numpy 2.0.2, matplotlib 3.10.8 and Pillow 11.3.0. The CSV digests
hold on any platform. The PNG bytes depend on the plotting stack, so the figure comparison is strict only with
`MMORCH_STRICT_FIGURES=1` on that stack (`constraints/reproduce.txt`).

The CSVs themselves are the committed `results/*.csv`; they are not copied here. Never regenerate these files from the
code under test.
