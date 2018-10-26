import time
import numpy as np
import pandas as pd
import aemvl_survey_noise as asn


def read_line(f):
    df = pd.read_csv(f, sep=",")
    return df.as_matrix()

# Use py.test -s to see ouputs
def test_simple_call():

    m = read_line("data/200101.csv")

    # Transform.
    m[:, 1:] = np.arcsinh(m[:, 1:])
    start = time.time()
    noise_array = asn.detect_noise_sections(m, 0.003, 0.01, 20, 10, 2, 4)
    end = time.time()
    print("Detection completed in: " + str(end - start))
    print(noise_array)

    assert noise_array
