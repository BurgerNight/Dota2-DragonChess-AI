import os
import sys
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import cv2
import numpy as np
import pyautogui


def capture_screenshot(x=0, y=0, w=2560, h=1600, show=False):
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    screenshot_array = np.array(screenshot)
    screenshot_array = cv2.cvtColor(screenshot_array, cv2.COLOR_RGB2BGR)
    if show:
        screenshot.show()
    return screenshot_array


def locate_image_on_screen(target_image_path, confidence=0.8):
    screenshot_array = capture_screenshot()
    target_image = cv2.imread(target_image_path)
    result = cv2.matchTemplate(screenshot_array, target_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= confidence:
        top_left = max_loc
        h, w, _ = target_image.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)

        return top_left, bottom_right
    else:
        raise Exception('game board not found on the screen.')


def load_image_templates(image_paths):
    image_templates = {}
    for image_name, path in image_paths.items():
        image_templates[image_name] = cv2.imread(path, cv2.IMREAD_COLOR)
    return image_templates


def load_elem_images(directory):
    elem_images = {}
    for filename in os.listdir(directory):
        if filename[0].isdigit():
            image_path = os.path.join(directory, filename)
            elem_name = filename.split('.')[0]
            elem_images[elem_name] = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return elem_images


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image


def find_game_board(resource_dir, show=False):
    '''
    Set show_board=True to check the board image
    Make sure the image shown is similar to board_sample.pgn
    '''
    top_left_loc = locate_image_on_screen(os.path.join(resource_dir, 'topLeft.png'))
    bottom_right_loc = locate_image_on_screen(os.path.join(resource_dir, 'botRight.png'))

    get_middle = lambda x: (x[0][0] + (x[1][0] - x[0][0]) // 2, x[0][1] + (x[1][1] - x[0][1]) // 2)
    top_left = get_middle(top_left_loc)
    bottom_right = get_middle(bottom_right_loc)

    print('board top left corner:', top_left)
    print('board bottom right corner:', bottom_right)

    x, y, w, h = top_left[0], top_left[1], bottom_right[0], bottom_right[1]
    capture_screenshot(x, y, w - x, h - y, show=show)
    return top_left, bottom_right


class MatchThreeAgent:
    ROW_NUM = 8
    COL_NUM = 8
    ELEM_TYPE = {'{}_{}'.format(i, j): i for i in range(1, 7) for j in range(1, 4)}
    SPECIAL_ELEM = ['{}_{}'.format(i, j) for i in range(1, 7) for j in range(2, 4)]
    MATCH_CONFIG_LIST = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1), (0, -2), (0, 2), (-2, 0),
        (2, 0)
    ]
    MATCH_CONFIG_DICT = {
        10: [{(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)}, {(0, -2), (0, -1), (0, 0), (0, 1), (0, 2)}],
        11: [
            {(-2, 0), (-1, 0), (0, 0), (0, -1), (0, -2)}, {(-2, 0), (-1, 0), (0, 0), (0, 1), (0, 2)},
            {(2, 0), (1, 0), (0, 0), (0, -1), (0, -2)}, {(2, 0), (1, 0), (0, 0), (0, 1), (0, 2)},
            {(-2, 0), (-1, 0), (0, 0), (0, -1), (0, 1)}, {(2, 0), (1, 0), (0, 0), (0, -1), (0, 1)},
            {(-1, 0), (1, 0), (0, 0), (0, -1), (0, -2)}, {(-1, 0), (1, 0), (0, 0), (0, 1), (0, 2)},
        ],
        12: [{(-2, 0), (-1, 0), (0, 0), (1, 0)}, {(-1, 0), (0, 0), (1, 0), (2, 0)}],
        13: [{(0, -2), (0, -1), (0, 0), (0, 1)}, {(0, -1), (0, 0), (0, 1), (0, 2)}],
    }

    def __init__(self, top_left, bottom_right):
        self.prev_elem_array = None
        self.elem_array = np.zeros((8, 8), dtype=np.int64)
        self.special_elem_array = np.zeros((8, 8), dtype=np.int64)
        self._grid_height = None
        self._grid_width = None
        self._grid_location = {}
        self._top_left = top_left
        self._bottom_right = bottom_right
        self._scores = [1] * self.ROW_NUM * self.COL_NUM

    def identify_game_board(self):
        x, y, w, h = self._top_left[0], self._top_left[1], self._bottom_right[0], self._bottom_right[1]
        board_array = capture_screenshot(x, y, w - x, h - y)
        return board_array

    def split_board_into_grids(self, grid_array):
        height, width, _ = grid_array.shape

        # initialize grid location on screen
        if self._grid_height is None:
            self._grid_height = height // self.ROW_NUM
            self._grid_width = width // self.COL_NUM
            for i in range(self.ROW_NUM):
                for j in range(self.COL_NUM):
                    self._grid_location[(i, j)] = (self._top_left[0] + j * self._grid_width + self._grid_width / 2,
                                                   self._top_left[1] + i * self._grid_height + self._grid_height / 2)

        grids = []
        for i in range(self.ROW_NUM):
            for j in range(self.COL_NUM):
                grid = grid_array[i * self._grid_height:(i + 1) * self._grid_height,
                       j * self._grid_width:(j + 1) * self._grid_width]
                grids.append(grid)
        return grids

    def update_elements(self, grids, elem_images):
        def find_best_match_element(grid, elem_images):
            best_match_score = -1
            best_match_elem = None

            for elem_name, elem_image in elem_images.items():
                match_scores = []
                for channel in range(3):  # BGR channel
                    result = cv2.matchTemplate(grid[:, :, channel], elem_image[:, :, channel], cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    match_scores.append(max_val)

                combined_score = np.mean(match_scores)
                if combined_score > best_match_score:
                    best_match_score = combined_score
                    best_match_elem = elem_name

                # early exit
                if best_match_score > 0.9:
                    break

            return best_match_elem, best_match_score

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda cell: find_best_match_element(cell, elem_images), grids))

        elem_array = np.zeros((8, 8), dtype=np.int64)
        special_elem_array = np.zeros((8, 8), dtype=np.int64)

        # update element array
        self._scores = []
        for i, (elem_name, score) in enumerate(results):
            row, col = i // 8, i % 8

            elem_array[row, col] = self.ELEM_TYPE[elem_name]
            if elem_name in self.SPECIAL_ELEM:
                special_elem_array[row, col] = self.ELEM_TYPE[elem_name]
            self._scores.append(score)

        self.prev_elem_array = self.elem_array
        self.elem_array = elem_array
        self.special_elem_array = special_elem_array

    def get_grid_element(self, index):
        if 0 <= index[0] < self.ROW_NUM and 0 <= index[1] < self.COL_NUM:
            return self.elem_array[index]
        return None

    def grid_index_add(self, index1, index2):
        return index1[0] + index2[0], index1[1] + index2[1]

    def grid_index_subtract(self, index1, index2):
        return index1[0] - index2[0], index1[1] - index2[1]

    def get_action(self):
        cur_grid_dict = {(i, j) for i in range(0, 8) for j in range(0, 8)}

        def equal_match_value(index1, index2):
            return index1 in cur_grid_dict and index2 in cur_grid_dict \
                and ar_match[index1] == ar_match[index2]

        def index_could_swap(index):
            return index in cur_grid_dict and ar_swap[index]

        ar_swap = np.zeros((self.ROW_NUM, self.COL_NUM))
        ar_match = np.zeros((self.ROW_NUM, self.COL_NUM))
        for index, grid in cur_grid_dict:
            ar_swap[index] = self.elem_array[index]
            ar_match[index] = self.elem_array[index]

        action_set = set()
        for i, j in np.argwhere(ar_swap // 10 == 1):
            if ar_swap[i, j] != 10:
                action_set.add(((i, j), (i, j)))
            for index in [(i - 1, j), (i + 1, j), (i, j + 1), (i, j - 1)]:
                if index_could_swap(index):
                    action_set.add(((i, j), index))
        for i, j in np.argwhere(np.logical_and(ar_match[:, :-1] == ar_match[:, 1:], ar_match[:, :-1] != 0)):
            if index_could_swap((i, j - 1)):
                for index in [(i - 1, j - 1), (i + 1, j - 1), (i, j - 2)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        action_set.add(((i, j - 1), index))
            if index_could_swap((i, j + 2)):
                for index in [(i - 1, j + 2), (i + 1, j + 2), (i, j + 3)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        action_set.add(((i, j + 2), index))
        for i, j in np.argwhere(np.logical_and(ar_match[:-1, :] == ar_match[1:, :], ar_match[:-1, :] != 0)):
            if index_could_swap((i - 1, j)):
                for index in [(i - 1, j - 1), (i - 1, j + 1), (i - 2, j)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        action_set.add(((i - 1, j), index))
            if index_could_swap((i + 2, j)):
                for index in [(i + 2, j - 1), (i + 2, j + 1), (i + 3, j)]:
                    if index_could_swap(index) and equal_match_value((i, j), index):
                        action_set.add(((i + 2, j), index))
        for i, j in np.argwhere(np.logical_and(ar_match[:, :-2] == ar_match[:, 2:], ar_match[:, :-2] != 0)):
            if index_could_swap((i, j + 1)):
                for index in [(i - 1, j + 1), (i + 1, j + 1)]:
                    if equal_match_value((i, j), index):
                        if index_could_swap(index):
                            action_set.add(((i, index[1]), index))
        for i, j in np.argwhere(np.logical_and(ar_match[:-2, :] == ar_match[2:, :], ar_match[:-2, :] != 0)):
            if index_could_swap((i + 1, j)):
                for index in [(i + 1, j - 1), (i + 1, j + 1)]:
                    if equal_match_value((i, j), index):
                        if index_could_swap(index):
                            action_set.add(((index[0], j), index))

        return list(action_set)

    def get_form_special_action(self, actions):
        indices = [index for indices in actions for index in indices]
        candidate_indices = [k for k, v in Counter(indices).items() if v >= 2]
        candidate_actions = [action for action in actions if
                             action[0] in candidate_indices or
                             action[1] in candidate_indices]

        form_special_actions = []

        for action in candidate_actions:
            for i in range(2):
                valid_indices = {(0, 0)}
                main_index, neighbor_index = action[i], action[1 - i]
                neighbor_grid_element = self.get_grid_element(neighbor_index)
                if neighbor_grid_element is not None:
                    for offset in self.MATCH_CONFIG_LIST:
                        grid_element = self.get_grid_element(self.grid_index_add(main_index, offset))
                        if grid_element is not None:
                            grid_index = self.grid_index_add(main_index, offset)
                            if grid_index != neighbor_index and grid_element == neighbor_grid_element:
                                valid_indices.add(self.grid_index_subtract(grid_index, main_index))
                for number, all_index_list in self.MATCH_CONFIG_DICT.items():
                    for index_set in all_index_list:
                        if valid_indices.intersection(index_set) == index_set:
                            if action not in form_special_actions:
                                form_special_actions.append(action)

        return form_special_actions

    def get_use_special_action(self, actions):
        use_special_actions = []

        for action in actions:
            index1, index2 = action[0], action[1]
            elem1, elem2 = self.elem_array[index1], self.elem_array[index2]
            neighbor_offset = [[-2, 0], [-1, 0], [1, 0], [2, 0], [0, -2], [0, -1], [0, 1], [0, 2]]

            for offset in neighbor_offset:
                neighbor_index1 = self.grid_index_add(index1, offset)
                if neighbor_index1 != index2 and self.get_grid_element(neighbor_index1) and self.special_elem_array[
                    neighbor_index1] == elem2:
                    use_special_actions.append(action)
                    break

                neighbor_index2 = self.grid_index_add(index2, offset)
                if neighbor_index2 != index1 and self.get_grid_element(neighbor_index2) and self.special_elem_array[
                    neighbor_index2] == elem1:
                    use_special_actions.append(action)
                    break

        return use_special_actions

    def take_action(self, actions, form_actions, use_actions):
        def swap_element(index1, index2):
            pyautogui.moveTo(index1[0], index1[1])
            pyautogui.mouseDown()
            pyautogui.moveTo(index2[0], index2[1], duration=0.16)
            pyautogui.mouseUp()

        if len(actions) == 0:
            return []
        if len(form_actions) > 0:
            action = random.choice(form_actions)
        elif len(use_actions) > 0:
            action = random.choice(use_actions)
        else:
            action = random.choice(actions)
            # action = sorted(actions, reverse=True)[0]

        screen_index1, screen_index2 = self._grid_location[action[0]], self._grid_location[action[1]]
        swap_element(screen_index1, screen_index2)

        # print('all_actions', actions)
        # print('form_actions', form_actions)
        # print('use_actions', use_actions)
        return action

    def get_confidence_score(self):
        return np.mean(self._scores)


def test_case():
    top_left = (185, 195)
    bottom_right = (1260, 1270)

    agent = MatchThreeAgent(top_left, bottom_right)
    x = np.array([[5,5,6,3,2,4,2,3],
                  [5,2,5,4,6,2,5,5],
                  [2,4,2,5,1,3,4,2],
                  [1,5,3,1,2,1,2,3],
                  [1,1,4,5,1,5,2,1],
                  [4,2,6,2,1,4,5,3],
                  [3,5,1,6,2,3,1,6],
                  [4,6,3,5,3,2,3,5]])

    y = np.array([[0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0]])

    agent.elem_array = x
    agent.special_elem_array = y
    action = agent.get_action()
    print(x)
    print(action)
    print(agent.get_form_special_action(action))
    print(agent.get_use_special_action(action))


def run(delay=5, show=False):
    time.sleep(delay)

    resource_dir = os.path.join(os.getcwd(), 'resource')

    # This finds the top left and bottom right coordinate of the game board.
    # If this function doesn't work, you could set the board coordinate manually
    top_left, bottom_right = find_game_board(resource_dir, show=show)
    # top_left = (185, 195)
    # bottom_right = (1260, 1270)

    elem_images = load_elem_images(resource_dir)
    agent = MatchThreeAgent(top_left, bottom_right)

    while True:
        t1 = time.time()

        board_array = agent.identify_game_board()
        grids = agent.split_board_into_grids(board_array)
        agent.update_elements(grids, elem_images)
        avg_confidence_score = agent.get_confidence_score()

        print('element matching confidence:', avg_confidence_score)
        if avg_confidence_score < 0.5:
            print('game board is not found on the screen')
            time.sleep(5)
        else:
            if agent.prev_elem_array is None or not np.array_equal(agent.prev_elem_array, agent.elem_array):
                actions = agent.get_action()
                form_actions = agent.get_form_special_action(actions)
                use_actions = agent.get_use_special_action(actions)
                action = agent.take_action(actions, form_actions, use_actions)

                print(agent.elem_array)
                print('action', action)
                print('time cost', time.time() - t1)

        print('=' * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delay', type=int, default=5, help='delay before start in seconds')
    parser.add_argument('-s', '--show', help='whether show game board when start', action="store_true")
    args = parser.parse_args()

    run(delay=args.delay, show=args.show)
