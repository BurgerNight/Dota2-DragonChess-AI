## Dota2 Dragon Chess Match-3 Game AI

This is an AI for the match-3 minigame embedded in Dota2 Crownfall Act 3. It uses image recognition to identify the elements and performs actions based on specific algorithms.

It will help you beat the game.

### Install
```
pip install -r requirements.txt
```
### Usage
1. Start the Dragon chess game in Dota2.
2. Run the following command in your terminal:
```
python run_agent.py
``` 
3. Switch the screen back to Dota2.

### Debug
To check if the game board is identified correctly, run:
```
python run_agent.py -s
``` 
The board image shown should look similar to board_sample.png.