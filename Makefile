


totxt:
	jupytext --set-formats ipynb,py:percent cg.ipynb

fromtxt:
	jupytext --set-formats ipynb,py:percent --sync cg.ipynb


rustpy: rust/src/$(wildcard *.rs)
	cd rust && maturin develop --release


bench:
	python -m pytest tests/ -v --benchmark-histogram

report: bench
	qlmanage -t -s 2028 *.svg -o .
