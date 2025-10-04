import numpy as np
import time
import os
import json
import pyautogui
import threading
from dotenv import load_dotenv
from Actions import Actions
from inference_sdk import InferenceHTTPClient

# Load environment variables
load_dotenv()

MAX_ENEMIES = 10
MAX_ALLIES = 10
SPELL_CARDS = ["Fireball", "Zap", "Arrows", "Tornado", "Rocket", "Lightning", "Freeze"]

# Tower priority weights
TOWER_PRIORITY = {
    "enemy_king": 3,
    "enemy_left_princess": 2,
    "enemy_right_princess": 2
}

class ClashRoyaleEnv:
    # Counter cards for enemy troops
    COUNTER_CARDS = {
        "giant": "Mini P.E.K.K.A",
        "hog rider": "Cannon",
        "balloon": "Arrows",
        "skeleton army": "Zap",
        "minions": "Arrows",
        "goblin gang": "Zap",
        "prince": "Mini P.E.K.K.A",
        "pekka": "Inferno Tower",
        "baby dragon": "Wizard",
        # add more as needed
    }

    def __init__(self):
        self.actions = Actions()
        self.rf_model = self.setup_roboflow()
        self.card_model = self.setup_card_roboflow()
        self.state_size = 1 + 2 * (MAX_ALLIES + MAX_ENEMIES) + 12  # include tower positions
        self.num_cards = 4
        self.grid_width = 18
        self.grid_height = 28
        self.screenshot_path = os.path.join(os.path.dirname(__file__), 'screenshots', "current.png")
        self.available_actions = self.get_available_actions()
        self.action_size = len(self.available_actions)
        self.current_cards = []

        self.game_over_flag = None
        self._endgame_thread = None
        self._endgame_thread_stop = threading.Event()

        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.prev_enemy_princess_towers = None
        self.match_over_detected = False

        # Load tower regions
        self.towers = self.load_tower_regions()

    # -------------------
    # Setup Methods
    # -------------------
    def setup_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY not set")
        return InferenceHTTPClient(api_url="http://localhost:9001", api_key=api_key)

    def setup_card_roboflow(self):
        api_key = os.getenv('ROBOFLOW_API_KEY')
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY not set")
        return InferenceHTTPClient(api_url="http://localhost:9001", api_key=api_key)

    # -------------------
    # Tower Methods
    # -------------------
    def load_tower_regions(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "towers_regions.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Towers JSON not found at {path}")
        with open(path, "r") as f:
            towers = json.load(f)
        for k, v in towers.items():
            towers[k] = {"x": v[0], "y": v[1], "w": v[2], "h": v[3]}
        return towers

    def in_tower_region(self, troop_x, troop_y, tower_key):
        tower = self.towers[tower_key]
        return tower["x"] <= troop_x <= tower["x"] + tower["w"] and tower["y"] <= troop_y <= tower["y"] + tower["h"]

    # -------------------
    # Environment Methods
    # -------------------
    def reset(self):
        print("Resetting environment for a new match...")
        time.sleep(2)
        self.game_over_flag = None
        self._endgame_thread_stop.clear()
        self._endgame_thread = threading.Thread(target=self._endgame_watcher, daemon=True)
        self._endgame_thread.start()
        self.prev_elixir = None
        self.prev_enemy_presence = None
        self.prev_enemy_princess_towers = self._count_enemy_princess_towers()
        self.match_over_detected = False
        return self._get_state()

    def close(self):
        self._endgame_thread_stop.set()
        if self._endgame_thread:
            self._endgame_thread.join()

    def step(self, action_index):
        if not self.match_over_detected and hasattr(self.actions, "detect_match_over") and self.actions.detect_match_over():
            self.match_over_detected = True

        if self.match_over_detected:
            action_index = len(self.available_actions) - 1  # No-op

        if self.game_over_flag:
            done = True
            reward = self._compute_reward(self._get_state())
            reward += 100 if self.game_over_flag == "victory" else -100
            return self._get_state(), reward, done

        self.current_cards = self.detect_cards_in_hand()
        if all(card == "Unknown" for card in self.current_cards):
            pyautogui.moveTo(1611, 831, duration=0.2)
            pyautogui.click()
            return self._get_state(), 0, False

        card_index, x_frac, y_frac = self.available_actions[action_index]
        spell_penalty = 0

        if card_index != -1 and card_index < len(self.current_cards):
            card_name = self.current_cards[card_index]
            x = int(x_frac * self.actions.WIDTH) + self.actions.TOP_LEFT_X
            y = int(y_frac * self.actions.HEIGHT) + self.actions.TOP_LEFT_Y
            self.actions.card_play(x, y, card_index)
            time.sleep(0.5)
            if card_name in SPELL_CARDS:
                state = self._get_state()
                enemy_positions = []
                for i in range(1 + 2 * MAX_ALLIES, 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES, 2):
                    ex = state[i]
                    ey = state[i + 1]
                    if ex != 0.0 or ey != 0.0:
                        enemy_positions.append((int(ex * self.actions.WIDTH), int(ey * self.actions.HEIGHT)))
                radius = 100
                if not any((abs(ex - x)**2 + abs(ey - y)**2)**0.5 < radius for ex, ey in enemy_positions):
                    spell_penalty = -5

        princess_tower_reward = 0
        current_enemy_princess_towers = self._count_enemy_princess_towers()
        if self.prev_enemy_princess_towers is not None:
            if current_enemy_princess_towers < self.prev_enemy_princess_towers:
                princess_tower_reward = 20
        self.prev_enemy_princess_towers = current_enemy_princess_towers

        reward = self._compute_reward(self._get_state()) + spell_penalty + princess_tower_reward
        next_state = self._get_state()
        done = False
        return next_state, reward, done

    # -------------------
    # State & Reward
    # -------------------
    def _get_state(self):
        self.actions.capture_area(self.screenshot_path)
        elixir = self.actions.count_elixir()
        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        if not workspace_name:
            raise ValueError("WORKSPACE_TROOP_DETECTION not set")

        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="detect-count-and-visualize",
            images={"image": self.screenshot_path}
        )

        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results and isinstance(results[0], dict) and "predictions" in results[0]:
            predictions = results[0]["predictions"]

        dict_preds = [p for p in predictions if isinstance(p, dict)]

        TOWER_CLASSES = {"ally king tower", "ally princess tower", "enemy king tower", "enemy princess tower"}

        def normalize_class(cls):
            return cls.strip().lower() if isinstance(cls, str) else ""

        allies = [(p["x"], p["y"]) for p in dict_preds if normalize_class(p.get("class", "")) not in TOWER_CLASSES and normalize_class(p.get("class", "")).startswith("ally") and "x" in p and "y" in p]
        enemies = [(p["x"], p["y"]) for p in dict_preds if normalize_class(p.get("class", "")) not in TOWER_CLASSES and normalize_class(p.get("class", "")).startswith("enemy") and "x" in p and "y" in p]

        def normalize(units):
            return [(x / self.actions.WIDTH, y / self.actions.HEIGHT) for x, y in units]

        def pad_units(units, max_units):
            units = normalize(units)
            if len(units) < max_units:
                units += [(0.0, 0.0)] * (max_units - len(units))
            return units[:max_units]

        ally_flat = [coord for pos in pad_units(allies, MAX_ALLIES) for coord in pos]
        enemy_flat = [coord for pos in pad_units(enemies, MAX_ENEMIES) for coord in pos]

        # Tower positions
        tower_positions = []
        for tower_key in ["enemy_left_princess", "enemy_right_princess", "enemy_king",
                          "ally_left_princess", "ally_right_princess", "ally_king"]:
            t = self.towers[tower_key]
            tower_positions.append(t["x"] / self.actions.WIDTH)
            tower_positions.append(t["y"] / self.actions.HEIGHT)

        state = np.array([elixir / 10.0] + ally_flat + enemy_flat + tower_positions, dtype=np.float32)
        return state

    def _compute_reward(self, state):
        reward = 0
        elixir = state[0] * 10
        enemy_positions = state[1 + 2 * MAX_ALLIES: 1 + 2 * MAX_ALLIES + 2 * MAX_ENEMIES]

        # Enemy presence penalty
        reward -= sum(enemy_positions[1::2])

        # Tower-aware rewards
        for tower_key in ["enemy_left_princess", "enemy_right_princess", "enemy_king"]:
            t = self.towers[tower_key]
            priority = TOWER_PRIORITY.get(tower_key, 1)
            for ex, ey in zip(enemy_positions[::2], enemy_positions[1::2]):
                if ex == 0 and ey == 0:
                    continue
                ex_px = int(ex * self.actions.WIDTH)
                ey_px = int(ey * self.actions.HEIGHT)
                if self.in_tower_region(ex_px, ey_px, tower_key):
                    reward += 2 * priority

        # Princess tower destroyed bonus
        current_towers = self._count_enemy_princess_towers()
        if self.prev_enemy_princess_towers is not None and current_towers < self.prev_enemy_princess_towers:
            reward += 10
        self.prev_enemy_princess_towers = current_towers

        # Lane pressure reward
        left_lane = sum(ey for ex, ey in zip(enemy_positions[::2], enemy_positions[1::2]) if ex < self.actions.WIDTH / 2)
        right_lane = sum(ey for ex, ey in zip(enemy_positions[::2], enemy_positions[1::2]) if ex >= self.actions.WIDTH / 2)
        reward += 1.0 / (1.0 + abs(left_lane - right_lane))

        # Elixir efficiency reward
        if self.prev_elixir is not None and self.prev_enemy_presence is not None:
            elixir_spent = self.prev_elixir - elixir
            enemy_reduced = self.prev_enemy_presence - sum(enemy_positions[1::2])
            if elixir_spent > 0 and enemy_reduced > 0:
                reward += 2 * min(elixir_spent, enemy_reduced)

        self.prev_elixir = elixir
        self.prev_enemy_presence = sum(enemy_positions[1::2])
        return reward

    # -------------------
    # Cards & Actions
    # -------------------
    def detect_cards_in_hand(self):
        try:
            card_paths = self.actions.capture_individual_cards()
            cards = []
            workspace_name = os.getenv('WORKSPACE_CARD_DETECTION')
            for card_path in card_paths:
                results = self.card_model.run_workflow(
                    workspace_name=workspace_name,
                    workflow_id="custom-workflow",
                    images={"image": card_path}
                )
                predictions = []
                if isinstance(results, list) and results:
                    preds_dict = results[0].get("predictions", {})
                    if isinstance(preds_dict, dict):
                        predictions = preds_dict.get("predictions", [])
                if predictions:
                    cards.append(predictions[0]["class"])
                else:
                    cards.append("Unknown")
            return cards
        except Exception as e:
            print(f"Error in detect_cards_in_hand: {e}")
            return []

    def get_available_actions(self):
        actions = [
            [card, x / (self.grid_width - 1), y / (self.grid_height - 1)]
            for card in range(self.num_cards)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
        ]
        actions.append([-1, 0, 0])  # No-op
        return actions

    # -------------------
    # Endgame & Tower Counting
    # -------------------
    def _endgame_watcher(self):
        while not self._endgame_thread_stop.is_set():
            result = self.actions.detect_game_end()
            if result:
                self.game_over_flag = result
                break
            time.sleep(0.5)

    def _count_enemy_princess_towers(self):
        self.actions.capture_area(self.screenshot_path)
        workspace_name = os.getenv('WORKSPACE_TROOP_DETECTION')
        results = self.rf_model.run_workflow(
            workspace_name=workspace_name,
            workflow_id="detect-count-and-visualize",
            images={"image": self.screenshot_path}
        )
        predictions = []
        if isinstance(results, dict) and "predictions" in results:
            predictions = results["predictions"]
        elif isinstance(results, list) and results and isinstance(results[0], dict) and "predictions" in results[0]:
            predictions = results[0]["predictions"]

        dict_preds = [p for p in predictions if isinstance(p, dict)]
        return sum(1 for p in dict_preds if p.get("class") == "enemy princess tower")
