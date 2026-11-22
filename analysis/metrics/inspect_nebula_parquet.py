"""Print the schema and one row of the released Nebula parquet files."""
import glob

import pyarrow.parquet as pq

files = sorted(glob.glob('datasets_nebula/data/*.parquet'))
print(len(files), 'files')
total = 0
for path in files:
    total += pq.ParquetFile(path).metadata.num_rows
print('rows', total)
first = pq.ParquetFile(files[0])
print(first.schema_arrow)
frame = first.read_row_group(0).to_pandas().head(2)
for column in frame.columns:
    value = frame[column].iloc[0]
    print(column, type(value), str(value)[:200])
