# Small utility to iterate over the parquet files generated by the preprocessing
# step.

import logging
import os
import sys
import pyarrow.parquet as parquet

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # If first argument is a directory, we will iterate over all the files in
    # that directory
    if os.path.isdir(sys.argv[-1]):
        files = os.listdir(sys.argv[-1])
        files = [os.path.join(sys.argv[-1], f) for f in files if f.endswith('.parquet')]
        files.sort()
        num_samples = 0
        for f in files:
            table = parquet.read_table(f)
            num_samples += table.num_rows
            logging.info("Found %d samples in %s", table.num_rows, f)
        logging.info("Found %d samples in total", num_samples)
    # Otherwise, we will just read the parquet file and print the content
    # of each sample one by one
    else:
        table = parquet.read_table(sys.argv[-1])
        df = table.to_pandas()
        for index, row in df.iterrows():
            content = row['content']
            print(content, end='')
            print('---')
            print("> Press Enter to continue: ", end='')
            input()
            print("\033c", end="")