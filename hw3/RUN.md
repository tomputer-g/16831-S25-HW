# How to reproduce results

Run the shell scripts starting with q1, q2, or q3. They should run the experiments and produce the data files needed for analysis.

# How to produce the figures

For q2 and q3, use tensorboard on the resulting logs:

```bash
python -m tensorboard.main --logdir <log dir of run>
```

For q1, replace the `log_dir` and `log_dir_root` inside the python script to match the actual runs. Then run:

```bash
python q1_viz.py
```