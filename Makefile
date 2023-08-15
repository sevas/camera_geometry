


totxt:
	jupytext --set-formats ipynb,py:percent cg.ipynb

fromtxt:
	jupytext --set-formats ipynb,py:percent --sync cg.ipynb


rustpy: rust/src/$(wildcard *.rs)
	cd rust && maturin develop --release


bench:
	python -m pytest tests/ -v --benchmark-histogram

report-macos: bench
	qlmanage -t -s 2028 *.svg -o .

report-win: bench
	inkscape -w 1024 *.svg -o benchmark_windows.png  # requires inkscape

