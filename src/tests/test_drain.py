import os.path as path
import pathlib

from ..data import drain

pwd = pathlib.Path(__file__).parent.absolute()
parser = drain.LogParser(log_format="<Date> <Time> <Level> <Component>: <Content>",
                         indir=path.join(pwd, "data"),
                         outdir=path.join(pwd, "data"))


def test_has_numbers():
    assert parser.has_numbers("xxx9xxx")
    assert parser.has_numbers("xxx9x1x")
    assert not parser.has_numbers("xxxaxxx")


def test_load_data():
    parser.log_name = "test.log"
    parser.load_data()
    df = parser.df_log
    assert df.columns.size == 6, "should contains all components + ID"


def test_preprocess():
    line = "20/02/28 22:47:37 INFO DatabricksILoop$: Finished creating throwaway interpreter"
    processed = parser.preprocess(line)
    assert processed == line, "preprocessing should not change this line"
