### Running the analysis

#### Precondition

You need [pixi](https://pixi.sh) installed.

#### Setup
Change to your repository directory:
```
cd W:\scratch\...
```

Define the current working directory:
```
$env:WD="runs/20250113"
```

Run configuration:
```
pixi run config
```

This will ask you for the input folder and other parameters of the analysis.


#### Run the analysis
```
pixi run s01_segmentation
pixi run s02_aggresomes
```

Done!
