# Observer

## Requirements
- Python >= 3.10
- NumPy >= 1.26.4
- pandas >= 2.2.3
- Zarr >= 2.13.3
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

Visualization Example for Data.

Each row represents the unit categorized into Worker, Ground, Air, and Building for Player 1 and Player 2, respectively.

<table border="1">
  <tr>
    <th width="20%">Overview</th>
    <th width="16%">Resource <br>/ Vision</th>
    <th width="16%">Worker</th>
    <th width="16%">Ground</th>
    <th width="16%">Air</th>
    <th width="16%">Building</th>
  </tr>
  <tr>
    <td rowspan="2" align="center">
      <figure>
        <img src="./figure/12.rep_16119_Overview.png" width="400">
      </figure>
    </td>
        <td rowspan="2" align="center">
      <figure>
        <img src="./figure/12.rep_16119_Resource.png" width="300">
      </figure>
      <br><br> <!-- 개행 추가로 셀 높이 확보 -->
      <figure>
        <img src="./figure/12.rep_16119_Vision.png" width="300">
      </figure>
    </td>
    <td>
      <figure>
        <img src="./figure/12.rep_16119_Player_1_Worker.png" width="300">
      </figure>
    </td>
    <td>
      <figure>
        <img src="./figure/12.rep_16119_Player_1_Ground.png" width="300">
      </figure>
    </td>
    <td>
      <figure>
        <img src="./figure/12.rep_16119_Player_1_Air.png" width="300">
      </figure>
    </td>
    <td>
      <figure>
        <img src="./figure/12.rep_16119_Player_1_Building.png" width="300">
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure>
        <img src="./figure/12.rep_16119_Player_2_Worker.png" width="300">
      </figure>
    </td>
    <td>
      <figure>
        <img src="./figure/12.rep_16119_Player_2_Ground.png" width="300">
      </figure>
    </td>
    <td>
      <figure>
        <img src="./figure/12.rep_16119_Player_2_Air.png" width="300">
      </figure>
    </td>
    <td>
      <figure>
        <img src="./figure/12.rep_16119_Player_2_Building.png" width="300">
      </figure>
    </td>
  </tr>
</table>




## TODO
- Ensure Linux compatibility (verify multiprocessing functionality)

