all: figures/fig_timeseries.pdf figures/fig_phase.pdf

figures/fig_timeseries.pdf: code/fig_timeseries.py
	python code/fig_timeseries.py figures/fig_timeseries.pdf

figures/fig_phase.pdf: code/fig_phase.py
	python code/fig_phase.py figures/fig_phase.pdf
