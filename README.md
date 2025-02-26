# Observer

## Requirements
- Python >= 3.10
- NumPy >= 1.26.4
- pandas >= 2.2.3
- SciPy >= 1.15.1
- pickle
- tqdm

## Usage
The script requires an input file or directory containing `.rep` files and an output directory for processed results.

### Input File Structure Requirement
The input directory must follow this structure:
```
data/
  ├── <replay_name>/
      ├── state
      ├── meta
      ├── event
      ├── terrain
      ├── vision
```
Each `.rep` file should have a corresponding directory under `data/` containing the required sub-files.

### Example
```bash
python script.py --input /path/to/replays --output /path/to/output
```

### Outputs
```
results/
  ├── <replay_name>/
      ├── meta.txt
      ├── 0.npy
      ├── 1.npy
      ├── 2.npy
      ├── ...
```
### Example

(example)

## TODO
- Ensure Linux compatibility (verify multiprocessing functionality)

