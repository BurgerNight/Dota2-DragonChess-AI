## Dota2 Dragon Chess Match-3 Game AI

This is an AI for the match-3 minigame embedded in Dota2 Crownfall Act 3. It uses image recognition to identify the elements and performs actions based on specific algorithms.

It will help you beat the game.

### Download
[Here](https://github.com/BurgerNight/Dota2-DragonChess-AI/releases)

### Usage

1. Start the DragonChess game in Dota2.
2. Run `run_agent.exe`
3. Switch the screen back to Dota2.
4. When finished, press ```Q``` or ```Esc``` to exit the program.



### Supported Arguments

`--delay` or `-d` (default: 10), The delay in seconds before the program starts running.

`--fast` or `f`, if set, it will not wait for a static board to perform the next action.

`--show` or `-s`, if set, it will display the identified game board image, use this for debugging.

`--board_coordinates` or `-b`,
Specify the coordinates of the game board manually in the format x1,y1,x2,y2 where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the square game board. This is useful if the program has trouble automatically identifying the game board.
```
# Run the following command from a terminal:
path/to/run_agent.exe -b 185 195 1260 1270
```
You can take a screenshot and paste it to Windows paint to find out the coordinates (as shown below). 
![board_example.png](board_example.png)


### Q&A
1. The program is not functioning properly.

* The program keeps showing
`Cannot locate the game board, make sure the game is on the screen.` even after you switch back to the game.
* The mouse is not clicking within the area of the board or not switching the right element.
* The program raises an error and exits.

These problems occur when the game board is not correctly identified. Try changing game resolution or use `--board_coordinates` above to manually set up board coordinates.

2. Will it get you VAC banned?

The program functions based on image recognition. It does not access or modify any game files and should be safe to use.
