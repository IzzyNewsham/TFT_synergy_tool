# TFT synergy tool

This is a Jupyter Notebook tool to help you craft your TFT (Teamfight Tactics) teams in real time. TFT is a game with characters from League of Legends, where you craft your team of characters with 'synergies' to try to win battles against other teams.


I created this because I found TFT to have a very steep learning curve, having never played league. This will help new players (and when new player sets come in) work out which synergies there are and where the intersection of different synergies are. Unfortunately this doesn't help with items!


If you just want to give it a go, see tft_synergy_tool.ipynb.


Note the team selector will probably not work in Jupyter lab (due to issues with ipywidgets).

tft_synergies_set4.csv (and tft_synergies.csv at an earlier date for set3) was obtained from copying and pasting text from this website: https://tftactics.gg/db/champions. 
tft_synergy_info_set4.csv was obtained by copying and pasting text from this website: https://lolchess.gg/synergies (tip: if you don't want the images along with the text to be pasted, copying into Google Sheets worked for me).



### Steps to get code working 

This is based on what worked for me on my Windows machine.


#### Requirements:
- Python3
- Jupyter Notebook
- pandas
- numpy
- networkx
- seaborn
- matplotlib
(installing the last 5 packages is done something like this: pip3 install --user pandas)


Get the code:
```
git clone https://github.com/IzzyNewsham/TFT_synergy_tool.git
```

Now you should have the code in the folder TFT_synergy_tool/.

I also use the package ipyevents from here: https://github.com/mwcraig/ipyevents.
You can go there to view the installation instructions, here are the basic commands that worked for me:

```
pip3 install ipyevents
jupyter nbextension enable --py --sys-prefix ipyevents
```

Now it should work! Go to tft_synergy_tool.ipynb and run the cells one at a time, hopefully it should be quite self explanatory.