import os
import sys
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import cv2
import numpy as np
import pyautogui
from pynput.keyboard import Listener, Key
from PIL import ImageGrab


start_agent = False
stop_agent = False
pause_agent = False


def on_press(key):
    global start_agent
    global stop_agent
    global pause_agent
    try:
        if key.char == 'b':
            start_agent = True
        elif key.char == 'p':
            pause_agent = not pause_agent
    except AttributeError:
        if start_agent and key == Key.esc:
            stop_agent = True
            # return False


def get_image_array(image):
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_array


def resize_image_by_resolution(image):
    # Get screen resolution
    screen_width, screen_height = ImageGrab.grab().size
    image_width, image_height, _ = image.shape
    original_width, original_height = 2560, 1600

    # Resize target image based on screen resolution
    image_height, image_width, _ = image.shape
    width_ratio = screen_width / original_width
    height_ratio = screen_height / original_height
    resize_ratio = min(width_ratio, height_ratio)

    image_resized = cv2.resize(image, (int(image_width * resize_ratio), int(image_height * resize_ratio)), interpolation=cv2.INTER_AREA)
    return image_resized


def locate_image_on_screen(target_image_path, confidence=0.8):
    screenshot = pyautogui.screenshot()
    image_array = get_image_array(screenshot)
    target_image = cv2.imread(target_image_path)
    target_image = resize_image_by_resolution(target_image)

    result = cv2.matchTemplate(image_array, target_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= confidence:
        top_left = max_loc
        h, w, _ = target_image.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        return top_left, bottom_right
    else:
        return None


def load_elem_images(directory):
    elem_images = {}
    for filename in os.listdir(directory):
        if filename[0].isdigit():
            image_path = os.path.join(directory, filename)
            elem_name = filename.split('.')[0]
            elem_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            elem_image = resize_image_by_resolution(elem_image)
            elem_images[elem_name] = elem_image
    return elem_images


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image


def find_game_board(resource_dir):
    '''
    Set show_board=True to check the board image
    Make sure the image shown is similar to board_sample.pgn
    '''
    top_left_loc = locate_image_on_screen(os.path.join(resource_dir, 'topLeft.jpg'))
    bottom_right_loc = locate_image_on_screen(os.path.join(resource_dir, 'botRight.jpg'))

    # Return None when board corner not found
    if top_left_loc is None or bottom_right_loc is None:
        return None, None

    get_middle = lambda x: (x[0][0] + (x[1][0] - x[0][0]) // 2, x[0][1] + (x[1][1] - x[0][1]) // 2)
    top_left = get_middle(top_left_loc)
    bottom_right = get_middle(bottom_right_loc)

    x, y, w, h = top_left[0], top_left[1], bottom_right[0], bottom_right[1]

    # Return None when board corner location is incorrectly identified
    if w - x < 0 or h - y < 0:
        return None, None

    print('board top left corner:', top_left)
    print('board bottom right corner:', bottom_right)

    return top_left, bottom_right


class ElementMatcher:
    def __init__(self, elem_images):
        self._elem_images = elem_images

    def find_best_match_element(self, grid):
        best_match_score = float('inf')
        best_match_elem = None

        for elem_name, elem in self._elem_images.items():
            hist_score = self.calculate_histogram_similarity(grid, elem)

            # Choose the element with the smallest histogram score (closest match)
            if hist_score < best_match_score:
                best_match_score = hist_score
                best_match_elem = elem_name

        return best_match_elem, best_match_score

    def calculate_histogram_similarity(self, grid, elem):
        # Convert images to HSV space
        hsv_grid = cv2.cvtColor(grid, cv2.COLOR_BGR2HSV)
        hsv_template = cv2.cvtColor(elem, cv2.COLOR_BGR2HSV)

        # Calculate histograms for each channel (Hue, Saturation, Value)
        hist_grid_hue = cv2.calcHist([hsv_grid], [0], None, [256], [0, 256])
        hist_grid_saturation = cv2.calcHist([hsv_grid], [1], None, [256], [0, 256])
        hist_grid_value = cv2.calcHist([hsv_grid], [2], None, [256], [0, 256])

        hist_template_hue = cv2.calcHist([hsv_template], [0], None, [256], [0, 256])
        hist_template_saturation = cv2.calcHist([hsv_template], [1], None, [256], [0, 256])
        hist_template_value = cv2.calcHist([hsv_template], [2], None, [256], [0, 256])

        # Normalize histograms
        cv2.normalize(hist_grid_hue, hist_grid_hue)
        cv2.normalize(hist_grid_saturation, hist_grid_saturation)
        cv2.normalize(hist_grid_value, hist_grid_value)

        cv2.normalize(hist_template_hue, hist_template_hue)
        cv2.normalize(hist_template_saturation, hist_template_saturation)
        cv2.normalize(hist_template_value, hist_template_value)

        # Calculate histogram comparison scores (using Bhattacharyya distance)
        hist_score_hue = cv2.compareHist(hist_grid_hue, hist_template_hue, cv2.HISTCMP_BHATTACHARYYA)
        hist_score_saturation = cv2.compareHist(hist_grid_saturation, hist_template_saturation,
                                                cv2.HISTCMP_BHATTACHARYYA)
        hist_score_value = cv2.compareHist(hist_grid_value, hist_template_value, cv2.HISTCMP_BHATTACHARYYA)

        # Combine scores
        hist_score = (hist_score_hue + hist_score_saturation + hist_score_value) / 3.0

        return hist_score


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

    def __init__(self, top_left, bottom_right, elem_images):
        self.prev_elem_array = None
        self.elem_array = np.zeros((8, 8), dtype=np.int64)
        self.special_elem_array = np.zeros((8, 8), dtype=np.int64)
        self._grid_height = None
        self._grid_width = None
        self._grid_location = {}
        self._top_left = top_left
        self._bottom_right = bottom_right
        self._scores = [1] * self.ROW_NUM * self.COL_NUM
        # self._elem_images = elem_images
        self._element_matcher = ElementMatcher(elem_images)

    def identify_game_board(self):
        x, y, w, h = self._top_left[0], self._top_left[1], self._bottom_right[0], self._bottom_right[1]
        board = pyautogui.screenshot(region=(x, y, w - x, h - y))
        board_array = get_image_array(board)
        return board_array

    def split_board_into_grids(self, grid_array):
        height, width, _ = grid_array.shape

        # Initialize grid location on the screen
        if self._grid_height is None:
            self._grid_height = height // self.ROW_NUM
            self._grid_width = width // self.COL_NUM
            for i in range(self.ROW_NUM):
                for j in range(self.COL_NUM):
                    self._grid_location[(i, j)] = (self._top_left[0] + j * self._grid_width + self._grid_width / 2,
                                                   self._top_left[1] + i * self._grid_height + self._grid_height / 2)
            # for elem_name, image in self._elem_images.items():
            #     self._elem_images[elem_name] = cv2.resize(image, (self._grid_width, self._grid_height))  # unused

        grids = []
        for i in range(self.ROW_NUM):
            for j in range(self.COL_NUM):
                x_margin, y_margin = 0, 0
                start_x, end_x = int(j * self._grid_width + x_margin), int((j + 1) * self._grid_width - x_margin)
                start_y, end_y = int(i * self._grid_height + y_margin), int((i + 1) * self._grid_height - y_margin)
                grid = grid_array[start_y:end_y, start_x:end_x]
                grids.append(grid)

        return grids

    def find_best_match_element(self, grid):
        best_match_elem, best_match_score = self._element_matcher.find_best_match_element(grid)
        return best_match_elem, best_match_score

    def update_elements(self, grids):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(self.find_best_match_element, grids))

        elem_array = np.zeros((8, 8), dtype=np.int64)
        special_elem_array = np.zeros((8, 8), dtype=np.int64)

        # Update element array
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
        def equal_match_value(index1, index2):
            return index1 in cur_grid_lst and index2 in cur_grid_lst and ar_match[index1] == ar_match[index2]

        def index_could_swap(index):
            return index in cur_grid_lst and ar_swap[index]

        cur_grid_lst = [(i, j) for i in range(0, self.ROW_NUM) for j in range(0, self.COL_NUM)]
        ar_swap = np.zeros((self.ROW_NUM, self.COL_NUM))
        ar_match = np.zeros((self.ROW_NUM, self.COL_NUM))
        for index in cur_grid_lst:
            ar_swap[index] = self.elem_array[index]
            ar_match[index] = self.elem_array[index]

        action_set = set()
        for i, j in np.argwhere(ar_swap // 10 == 1):
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
        index_candidates = [k for k, v in Counter(indices).items() if v >= 2]
        action_candidates = [action for action in actions if
                             action[0] in index_candidates or
                             action[1] in index_candidates]

        form_special_actions = []
        form_type = defaultdict(list)

        for action in action_candidates:
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
                for num, all_index_list in self.MATCH_CONFIG_DICT.items():
                    for index_set in all_index_list:
                        if valid_indices.intersection(index_set) == index_set:
                            form_type[num].append(action)

        for num in sorted(form_type):
            form_special_actions += form_type[num]

        return form_special_actions

    def get_use_special_action(self, actions):
        use_special_actions = []

        for action in actions:
            index1, index2 = action[0], action[1]
            elem1, elem2 = self.elem_array[index1], self.elem_array[index2]
            neighbor_offset = [[-2, 0], [-1, 0], [1, 0], [2, 0], [0, -2], [0, -1], [0, 1], [0, 2]]

            index1_ignore = [2 * i for i in self.grid_index_subtract(index2, index1)]
            index1_candidates = [offset for offset in neighbor_offset if offset not in [index2, index1_ignore]]

            index2_ignore = [2 * i for i in self.grid_index_subtract(index1, index2)]
            index2_candidates = [offset for offset in neighbor_offset if offset not in [index1, index2_ignore]]

            for offset in index1_candidates:
                neighbor_index = self.grid_index_add(index1, offset)
                if self.get_grid_element(neighbor_index) and self.special_elem_array[neighbor_index] == elem2:
                    use_special_actions.append(action)
                    break

            for offset in index2_candidates:
                neighbor_index = self.grid_index_add(index2, offset)
                if self.get_grid_element(neighbor_index) and self.special_elem_array[neighbor_index] == elem1:
                    use_special_actions.append(action)
                    break

        return use_special_actions

    def take_action(self, actions, form_actions, use_actions):
        def swap_element(index1, index2):
            pyautogui.moveTo(index1[0], index1[1])
            pyautogui.mouseDown()
            pyautogui.moveTo(index2[0], index2[1], duration=0.16)
            pyautogui.mouseUp()

        if len(form_actions) > 0:
            action = random.choice(form_actions)
        elif len(use_actions) > 0:
            action = random.choice(use_actions)
        elif len(actions) > 0:
            action = random.choice(actions)
        else:
            return []

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

    agent = MatchThreeAgent(top_left, bottom_right, None)
    x = np.array([[5, 5, 6, 3, 2, 4, 2, 3],
                  [5, 2, 5, 4, 6, 2, 5, 5],
                  [2, 4, 2, 5, 1, 3, 4, 2],
                  [1, 5, 3, 1, 2, 1, 2, 3],
                  [1, 1, 4, 5, 1, 5, 2, 1],
                  [4, 2, 6, 2, 1, 4, 5, 3],
                  [3, 5, 1, 6, 2, 3, 1, 6],
                  [4, 6, 3, 5, 3, 2, 3, 5]])

    y = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])

    agent.elem_array = x
    agent.special_elem_array = y
    action = agent.get_action()
    print(x)
    print('all_actions', action)
    print('form_actions', agent.get_form_special_action(action))
    print('use_actions', agent.get_use_special_action(action))


def run(wait_static, show, board_coordinates):
    global start_agent
    global stop_agent

    while not start_agent:
        time.sleep(1)
    pyautogui.click(interval=0.5)
    time.sleep(4)

    cur_path = sys.argv[0]
    cur_dir = os.path.dirname(cur_path)
    resource_dir = os.path.join(cur_dir, 'resource')

    # Check if board coordinates is defined by arguments
    if board_coordinates is not None:
        print('Set board coordinates', board_coordinates)
        assert len(board_coordinates) == 4, 'Need 4 indices to locate the board, (x1, y1, x2, y2)'

        board_corners = [int(s) for s in board_coordinates]
        top_left, bottom_right = board_corners[:2], board_corners[2:]
    else:
        # Identify the game board from the screen
        while True:
            top_left, bottom_right = find_game_board(resource_dir)
            if top_left is None or bottom_right is None:
                print('Cannot locate the game board, make sure the game is on the screen.')
                time.sleep(5)
            else:
                break

    if show:
        x, y = top_left
        w, h = bottom_right
        board = pyautogui.screenshot(region=(x, y, w - x, h - y))
        board.show()

    elem_images = load_elem_images(resource_dir)
    agent = MatchThreeAgent(top_left, bottom_right, elem_images)

    while True:
        if stop_agent:
            print("Program stopped by user.")
            break
        if pause_agent:
            print("Program paused by user. Press 'p again to unpause.'")
            time.sleep(1)
            continue

        t1 = time.time()

        # Update elements on the board
        board_array = agent.identify_game_board()
        grids = agent.split_board_into_grids(board_array)
        agent.update_elements(grids)
        avg_confidence_score = agent.get_confidence_score()

        # Take action when elements start to change
        if not wait_static or agent.prev_elem_array is None or np.array_equal(agent.prev_elem_array, agent.elem_array):
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
    parser.add_argument('-w', '--wait_static', help='slow action, wait for a static board', action="store_true")
    parser.add_argument('-s', '--show_board', help='Show game board during runtime', action="store_true")
    parser.add_argument('-b', '--board_coordinates', nargs='+',
                        help='Define board coordinates (x1, y1, x2, y2) for the top-left and bottom-right corners')
    args = parser.parse_args()

    # test_case()

    print("Move the mouse over 'play' button and press 'b' to start.")

    listener = Listener(on_press=on_press)
    listener.start()

    try:
        run(wait_static=args.wait_static, show=args.show_board, board_coordinates=args.board_coordinates)
    except Exception as e:
        print('An error occurred: {}'.format(e))
        time.sleep(5)
        raise
