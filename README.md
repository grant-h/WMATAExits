# WMATA Metro Station Exit

A script to take the handcrafted Washington, DC Metro (WMATA) PDF diagrams created by [u/eable2](https://www.reddit.com/user/eable2/) on Reddit ([original post](https://www.reddit.com/r/washingtondc/comments/15mbos4/i_mapped_the_layouts_of_all_98_metro_stations_so/)). The presentation source is [available here](https://docs.google.com/presentation/d/17O0lMfjuyOvhWv75Umjb-2kkFi-HYRT8onZUGhwBXMM/edit).

Export a PDF from there to get the latest version.

## Usage

You will need a Python 3 runtime environment. Use pip to install `pypdf` (preferably into a virtual environment).

```
pip3 install pypdf
```

Then run the script on the PDF to produce a JSON file:

```
./wmata_parse.py "WMATA Metro Station Platform Exit Guide.pdf"
```

## TODO

* Include which metro lines are at each station
* Resolve indicator position to a track/platform relative location that is independent of the page size
* Resolve preferred exits (need to understand PDF shapes)
* Resolve exit/escalator/stair direction
