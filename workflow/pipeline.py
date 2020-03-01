import mlflow

from logparser import Drain

input_dir = '../logs/raw/'  # The input directory of log file
output_dir = 'result/'  # The output directory of parsing results
log_file = 'log4j-2020-02-28-22.log'  # The input log file name
log_format = '<Date> <Time> <Level> <Component>: <Content>'  # HDFS log format
# Regular expression list for optional preprocessing (default: [])
regex = []
st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
parser.parse(log_file)

mlflow.log_artifacts(output_dir, "parsed_log")
