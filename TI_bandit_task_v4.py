"""
Transitive Inference + Bandit-Exploration Task  (version 3)
===========================================================
PsychoPy Coder script (no Builder .psyexp needed - just run this .py file).

TASK STRUCTURE
--------------
3 BLOCKS (linear_positive, linear_negative, random; order shuffled for every
participant). Each block has its own NEW set of 7 objects, always shown in the
same left-to-right order. Each block has 4 PARTS:

1) BANDIT PHASE - 5 bandit screens in a row.
   On every screen the participant sees the 7 objects in their fixed
   left-to-right order and clicks N_BANDIT_PICKS (=3) of them to see the reward.
   After the 3rd pick the numbers stay on screen for as long as the participant
   wants; a CONTINUE button takes them to the next bandit screen.

2) LEARNING PHASE 1 (sequence recall, BEFORE TI).
   The participant rebuilds the left-to-right order of the objects, as seen
   in the bandit phase, by clicking the objects one by one.

3) TI PHASE - happens ONCE per block.
   All 21 possible pairs of the 7 objects are shown, each pair N_TI_REPEATS
   (=3) times (63 trials, shuffled, left/right randomised). The participant
   clicks the object they think is "bigger". Feedback: green smiley / red sad face.
   "Bigger" is judged by the block's average reward values (REWARD_MEANS,
   averaged over the 5 bandit screens).

4) LEARNING PHASE 2 (sequence recall, AFTER TI).
   Same task as part 2, repeated after the TI phase.

REWARDS (see "YOU DECIDE HERE"): for every condition and every bandit screen
(1-5) write 7 mean rewards in left-to-right screen order. Participants see
these means plus Gaussian noise (REWARD_NOISE_SD).

STIMULI
-------
21 images in an "images" folder next to this script, one subfolder per block:
    images/block1/stim1.png ... stim7.png
    images/block2/stim1.png ... stim7.png
    images/block3/stim1.png ... stim7.png
(.png, .jpg or .jpeg all work.) The hidden order of the objects is randomised
separately for every participant and block.

DATA OUTPUT (all in a "data" folder)
-------------------------------------
    <ID>_<session>_<date>_hierarchy.csv  - true rank of each stimulus (per block)
    <ID>_<session>_<date>_bandit.csv     - one row per bandit click
    <ID>_<session>_<date>_ti.csv         - one row per TI trial
    <ID>_<session>_<date>_recall.csv     - one row per learning-phase click
                                           ('recall_phase' = pre_ti / post_ti)
"""

from psychopy import visual, core, event, gui, data, logging
import os, random, csv, itertools
import numpy as np

# -----------------------------------------------------------------------
# 1. PARTICIPANT INFO / SETUP
# -----------------------------------------------------------------------
exp_info = {'Participant ID': '', 'Session': '001'}
dlg = gui.DlgFromDict(dictionary=exp_info, title='TI Bandit Task')
if not dlg.OK:
    core.quit()

participant_id = exp_info['Participant ID']
session = exp_info['Session']
date_str = data.getDateStr()

this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

recall_fname = os.path.join(data_dir, f'{participant_id}_{session}_{date_str}_recall.csv')
bandit_fname = os.path.join(data_dir, f'{participant_id}_{session}_{date_str}_bandit.csv')
ti_fname = os.path.join(data_dir, f'{participant_id}_{session}_{date_str}_ti.csv')

recall_file = open(recall_fname, 'w', newline='', buffering=1)
recall_writer = csv.writer(recall_file)
recall_writer.writerow(['participant', 'block_num', 'block_condition', 'recall_phase',
                         'slot_position', 'placed_stim',
                         'correct_stim', 'correct', 'rt'])

bandit_file = open(bandit_fname, 'w', newline='', buffering=1)
bandit_writer = csv.writer(bandit_file)
bandit_writer.writerow(['participant', 'block_num', 'block_condition', 'bandit_screen',
                         'click_order', 'stim_id', 'true_rank', 'reward', 'rt'])

ti_file = open(ti_fname, 'w', newline='', buffering=1)
ti_writer = csv.writer(ti_file)
ti_writer.writerow(['participant', 'block_num', 'block_condition',
                     'trial_num', 'pair_repeat', 'left_stim', 'right_stim',
                     'left_reward_level', 'right_reward_level',
                     'correct_side', 'response_stim', 'correct', 'rt'])

# -----------------------------------------------------------------------
# 2. STIMULI + HIERARCHY + FIXED SEQUENCE SETUP
# -----------------------------------------------------------------------
N_STIM = 7   # objects per block

# =======================================================================
# YOU DECIDE HERE
# =======================================================================
N_BLOCKS = 3
N_BANDIT_SCREENS = 5     # bandit screens per block (each ends with a Continue button)
N_BANDIT_PICKS = 3       # objects the participant can sample on each bandit screen
N_TI_REPEATS = 3         # how many times each of the 21 pairs appears in the TI phase

# Gaussian SD of noise added to every reward shown. 0 = exact numbers below.
REWARD_NOISE_SD = 8

# Stimulus sizes (pixels) - change here to resize everything at once
BANDIT_STIM_SIZE = 150    # objects on the Explore screens   (was 110)
BANDIT_SPACING = 170      # distance between object centres  (was 140)
TI_STIM_SIZE = 240        # objects on the Judge screens     (was 160)
RECALL_STIM_SIZE = 140    # objects on the Learn screens     (was 100)
FEEDBACK_FACE_SIZE = 60   # smiley / sad face radius         (was 90)

# Block order: shuffled for EVERY participant, recorded in 'block_condition'.
CONDITIONS = ['linear_positive', 'linear_negative', 'random']
BLOCK_ORDER = CONDITIONS.copy()
random.shuffle(BLOCK_ORDER)

# Mean rewards, in SCREEN POSITION ORDER (first = leftmost ... last = rightmost).
# One list for each of the 5 bandit screens. Numbers inside a list must differ.
LINEAR_UP = [10, 23, 37, 50, 63, 77, 90]
LINEAR_DOWN = LINEAR_UP[::-1]
RANDOM_FIXED = [50, 10, 77, 37, 90, 23, 63]

REWARD_MEANS = {
    'linear_positive': {1: LINEAR_UP, 2: LINEAR_UP, 3: LINEAR_UP, 4: LINEAR_UP, 5: LINEAR_UP},
    'linear_negative': {1: LINEAR_DOWN, 2: LINEAR_DOWN, 3: LINEAR_DOWN, 4: LINEAR_DOWN, 5: LINEAR_DOWN},
    'random': {1: RANDOM_FIXED, 2: RANDOM_FIXED, 3: RANDOM_FIXED, 4: RANDOM_FIXED, 5: RANDOM_FIXED},
}
# =======================================================================

# =======================================================================
# OBJECT SETS: every block uses its OWN 7 new objects
# =======================================================================
N_ALL_STIM = N_BLOCKS * N_STIM
IMAGES_DIR = os.path.join(this_dir, 'images')


def find_image(block_num, k):
    """images/block<block_num>/stim<k>.(png|jpg|jpeg); None if not found."""
    base = os.path.join(IMAGES_DIR, f'block{block_num}', f'stim{k}')
    for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
        if os.path.exists(base + ext):
            return base + ext
    return None


STIM_FILES = []          # index = global stim id
_missing = []
for _b in range(1, N_BLOCKS + 1):
    for _k in range(1, N_STIM + 1):
        _f = find_image(_b, _k)
        if _f is None:
            _missing.append(f'images/block{_b}/stim{_k}.png')
        STIM_FILES.append(_f)
if _missing:
    raise FileNotFoundError(
        f"Missing {len(_missing)} image(s): {_missing}\n"
        f"Each block needs its own folder with 7 images named stim1 ... stim{N_STIM} "
        f"(.png or .jpg), e.g. images/block1/stim1.png")

BLOCK_STIM_IDS = {b: list(range((b - 1) * N_STIM, b * N_STIM)) for b in range(1, N_BLOCKS + 1)}

rank_map = {}               # stim_id -> true rank within its block
BLOCK_SEQUENCE_ORDER = {}   # block -> stim ids in left-to-right screen order
for _b, _ids in BLOCK_STIM_IDS.items():
    _ranks = list(range(1, N_STIM + 1))
    random.shuffle(_ranks)
    rank_map.update(dict(zip(_ids, _ranks)))
    BLOCK_SEQUENCE_ORDER[_b] = sorted(_ids, key=lambda i: rank_map[i])

# Switched to the current block's values at the start of every block.
STIM_IDS = BLOCK_STIM_IDS[1]
SEQUENCE_ORDER = BLOCK_SEQUENCE_ORDER[1]

with open(os.path.join(data_dir, f'{participant_id}_{session}_{date_str}_hierarchy.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['stim_id', 'stim_file', 'true_rank', 'sequence_position', 'block_num'])
    for _b in range(1, N_BLOCKS + 1):
        for i in BLOCK_STIM_IDS[_b]:
            w.writerow([i, os.path.relpath(STIM_FILES[i], IMAGES_DIR), rank_map[i],
                        BLOCK_SEQUENCE_ORDER[_b].index(i), _b])


def _check_reward_config():
    assert len(BLOCK_ORDER) == N_BLOCKS, "BLOCK_ORDER must have N_BLOCKS entries"
    for cond in CONDITIONS:
        for r in range(1, N_BANDIT_SCREENS + 1):
            means = REWARD_MEANS[cond][r]
            assert len(means) == N_STIM, f"{cond} screen {r}: need {N_STIM} numbers"
            assert len(set(means)) == N_STIM, f"{cond} screen {r}: numbers must all be different"


_check_reward_config()


def build_round_reward_map(condition, bandit_screen):
    """stim_id -> mean reward for one bandit screen (list is in left-to-right order)."""
    means = REWARD_MEANS[condition][bandit_screen]
    return {SEQUENCE_ORDER[i]: float(means[i]) for i in range(N_STIM)}


def build_block_reward_map(condition):
    """stim_id -> mean reward averaged over the bandit screens of the block.
    Used to decide which object is 'bigger' in the TI phase."""
    return {SEQUENCE_ORDER[i]: float(np.mean([REWARD_MEANS[condition][r][i]
                                                for r in range(1, N_BANDIT_SCREENS + 1)]))
            for i in range(N_STIM)}


def get_reward(stim_id, reward_map, noise_sd=REWARD_NOISE_SD):
    mean = reward_map[stim_id]
    if noise_sd == 0:
        return float(mean)
    val = np.random.normal(loc=mean, scale=noise_sd)
    return float(np.clip(val, 0, 100))


# -----------------------------------------------------------------------
# 3. WINDOW + GENERIC HELPERS
# -----------------------------------------------------------------------
win = visual.Window(size=(1280, 800), color='white', units='pix', fullscr=False)
mouse = event.Mouse(win=win)
clock = core.Clock()

instr_text = visual.TextStim(win, text='', color='black', height=26, wrapWidth=1000)


def show_instructions(text, advance_key='space'):
    instr_text.text = text
    instr_text.draw()
    quit_button.draw()
    quit_label.draw()
    win.flip()
    keys = event.waitKeys(keyList=[advance_key, 'escape'])
    if 'escape' in keys:
        cleanup_and_quit()


def wait_for_mouse_release():
    while mouse.getPressed()[0] == 1:
        core.wait(0.01)


# Global on-screen Quit button
quit_button = visual.Rect(win, width=90, height=36, pos=(560, 360),
                           fillColor='firebrick', lineColor='black')
quit_label = visual.TextStim(win, text='Quit', pos=(560, 360), color='white', height=18)


def check_quit():
    """Call once per frame inside any loop. Quits on Escape or Quit-button click."""
    if event.getKeys(keyList=['escape']):
        cleanup_and_quit()
    if mouse.getPressed()[0] == 1 and quit_button.contains(mouse):
        cleanup_and_quit()


def draw_feedback_face(win, correct, pos=(0, -40), size=FEEDBACK_FACE_SIZE):
    """Simple geometric smiley (green, correct) / sad face (red, incorrect)."""
    color = 'forestgreen' if correct else 'firebrick'
    line_w = max(2, size / 12)
    face = visual.Circle(win, radius=size, pos=pos, fillColor=None,
                          lineColor=color, lineWidth=line_w, edges=64)
    eye_off_x, eye_off_y = size * 0.35, size * 0.3
    left_eye = visual.Circle(win, radius=size * 0.10, pos=(pos[0] - eye_off_x, pos[1] + eye_off_y),
                              fillColor=color, lineColor=color)
    right_eye = visual.Circle(win, radius=size * 0.10, pos=(pos[0] + eye_off_x, pos[1] + eye_off_y),
                               fillColor=color, lineColor=color)
    n = 24
    xs = np.linspace(-size * 0.5, size * 0.5, n)
    if correct:
        ys = (xs ** 2) / size - size * 0.35
    else:
        ys = -(xs ** 2) / size + size * 0.15
    verts = list(zip(pos[0] + xs, pos[1] + ys))
    mouth = visual.ShapeStim(win, vertices=verts, closeShape=False, lineColor=color, lineWidth=line_w)
    face.draw()
    left_eye.draw()
    right_eye.draw()
    mouth.draw()


# -----------------------------------------------------------------------
# 4. INTRODUCTION
# -----------------------------------------------------------------------
def run_introduction():
    show_instructions(
        "Welcome to the experiment!\n\n"
        "In each block you will see 7 new objects, each giving a reward number. "
        "The numbers are noisy: the same object can show a different number each time. "
        "So don't try to memorize them.\n\n"
        "Each block has four parts: Explore, Learn, Judge and Learn again.\n\n"
        "Press SPACE to begin.")


# -----------------------------------------------------------------------
# 5. BANDIT PHASE (5 screens, each ended with a Continue button)
# -----------------------------------------------------------------------
continue_button = visual.Rect(win, width=160, height=50, pos=(0, -250),
                               fillColor='forestgreen', lineColor='black')
continue_label = visual.TextStim(win, text='Continue', pos=(0, -250), color='white',
                                  height=24, bold=True)


def run_bandit_screen(block_num, condition, screen_num, reward_map):
    spacing = BANDIT_SPACING
    total_width = spacing * (N_STIM - 1)
    xs = [-total_width / 2 + i * spacing for i in range(N_STIM)]
    y = 0

    stims = {}
    for i, stim_id in enumerate(SEQUENCE_ORDER):
        img = visual.ImageStim(win, image=STIM_FILES[stim_id], pos=(xs[i], y),
                               size=(BANDIT_STIM_SIZE, BANDIT_STIM_SIZE))
        stims[stim_id] = {'img': img, 'pos': (xs[i], y)}

    header = visual.TextStim(win, text=f'Explore {screen_num} of {N_BANDIT_SCREENS}',
                              pos=(0, 280), color='black', height=26, bold=True)
    noise_note = visual.TextStim(win, text='Numbers are noisy: use them as a rough guide, not exact values.',
                                  pos=(0, 235), color='dimgray', height=20, italic=True)

    clicked = []
    reward_labels = {}
    label_offset = BANDIT_STIM_SIZE / 2 + 25

    click_clock = core.Clock()
    click_clock.reset()
    mouse.clickReset()

    # --- picking: 3 objects ---
    while len(clicked) < N_BANDIT_PICKS:
        header.draw()
        noise_note.draw()
        for stim_id, s in stims.items():
            s['img'].draw()
            if stim_id in reward_labels:
                reward_labels[stim_id].draw()

        instr = visual.TextStim(win, text=f'Choose {N_BANDIT_PICKS - len(clicked)} more...',
                                 pos=(0, -250), color='black', height=22)
        instr.draw()
        quit_button.draw()
        quit_label.draw()
        win.flip()
        check_quit()

        if mouse.getPressed()[0] == 1:
            for stim_id, s in stims.items():
                if stim_id in clicked:
                    continue
                if s['img'].contains(mouse):
                    rt = click_clock.getTime()
                    # random block: no noise, so every object shows the SAME number on all 5 screens
                    reward = get_reward(stim_id, reward_map,
                                        noise_sd=0 if condition == 'random' else REWARD_NOISE_SD)
                    clicked.append(stim_id)

                    lbl = visual.TextStim(win, text=f'{reward:.0f}',
                                           pos=(s['pos'][0], s['pos'][1] + label_offset),
                                           color='green', height=26, bold=True)
                    reward_labels[stim_id] = lbl

                    bandit_writer.writerow([
                        participant_id, block_num, condition, screen_num,
                        len(clicked), stim_id, rank_map[stim_id],
                        round(reward, 2), round(rt, 3)
                    ])

                    click_clock.reset()
                    wait_for_mouse_release()
                    break

    # --- review: numbers stay visible until the participant clicks Continue ---
    wait_for_mouse_release()
    done = False
    while not done:
        header.draw()
        noise_note.draw()
        for stim_id, s in stims.items():
            s['img'].draw()
            if stim_id in reward_labels:
                reward_labels[stim_id].draw()
        continue_button.draw()
        continue_label.draw()
        quit_button.draw()
        quit_label.draw()
        win.flip()
        check_quit()

        if mouse.getPressed()[0] == 1 and continue_button.contains(mouse):
            wait_for_mouse_release()
            done = True


def run_bandit_phase(block_num, condition):
    show_instructions(
        f"Block {block_num} - Part 1: Explore\n\n"
        f"You will see the objects {N_BANDIT_SCREENS} times.\n"
        f"Each time, choose {N_BANDIT_PICKS} of the objects to see their reward.\n"
        "Take your time to look at the numbers, then press Continue.\n\n"
        "Pay attention to the order of the objects.\n\n"
        "Press SPACE to begin.")
    for screen_num in range(1, N_BANDIT_SCREENS + 1):
        reward_map = build_round_reward_map(condition, screen_num)
        run_bandit_screen(block_num, condition, screen_num, reward_map)


# -----------------------------------------------------------------------
# 6. TI PHASE (once per block, every pair N_TI_REPEATS times)
# -----------------------------------------------------------------------
def run_ti_phase(block_num, condition, part_num=3):
    block_reward_map = build_block_reward_map(condition)

    show_instructions(
        f"Block {block_num} - Part {part_num}: Judge\n\n"
        "Choose the object that tends to\n"
        "give higher rewards overall.\n\n"
        "Press SPACE to begin.")

    question_text = visual.TextStim(win, text='Which one is bigger?', pos=(0, 190),
                                     color='black', height=30, bold=True)

    # every pair N_TI_REPEATS times, all shuffled together
    base_pairs = list(itertools.combinations(STIM_IDS, 2))
    trials = [(a, b, rep) for rep in range(1, N_TI_REPEATS + 1) for (a, b) in base_pairs]
    random.shuffle(trials)

    left_pos = (-230, -40)
    right_pos = (230, -40)

    fixation = visual.TextStim(win, text='+', color='black', height=60, pos=(0, -40))

    for trial_num, (a, b, rep) in enumerate(trials, start=1):
        left_stim, right_stim = (a, b) if random.random() < 0.5 else (b, a)

        left_img = visual.ImageStim(win, image=STIM_FILES[left_stim], pos=left_pos,
                                    size=(TI_STIM_SIZE, TI_STIM_SIZE))
        right_img = visual.ImageStim(win, image=STIM_FILES[right_stim], pos=right_pos,
                                     size=(TI_STIM_SIZE, TI_STIM_SIZE))

        fixation.draw()
        win.flip()
        core.wait(0.5)

        clock.reset()
        mouse.clickReset()
        response_stim = None
        rt = None

        while response_stim is None:
            question_text.draw()
            left_img.draw()
            right_img.draw()
            quit_button.draw()
            quit_label.draw()
            win.flip()

            if mouse.getPressed()[0] == 1:
                if left_img.contains(mouse):
                    response_stim = left_stim
                    rt = clock.getTime()
                elif right_img.contains(mouse):
                    response_stim = right_stim
                    rt = clock.getTime()
                if response_stim is not None:
                    wait_for_mouse_release()

            check_quit()

        correct_side = 'left' if block_reward_map[left_stim] > block_reward_map[right_stim] else 'right'
        response_side = 'left' if response_stim == left_stim else 'right'
        correct = int(response_side == correct_side)

        win.flip()  # clear objects
        feedback_msg = visual.TextStim(win, text=('Correct!' if correct else 'Wrong'),
                                        pos=(0, 40), color='black', height=30, bold=True)
        feedback_msg.draw()
        draw_feedback_face(win, bool(correct), pos=(0, -40), size=FEEDBACK_FACE_SIZE)
        win.flip()
        core.wait(1.0)

        ti_writer.writerow([
            participant_id, block_num, condition, trial_num, rep,
            left_stim, right_stim, round(block_reward_map[left_stim], 1),
            round(block_reward_map[right_stim], 1),
            correct_side, response_stim, correct, round(rt, 3)
        ])

        win.flip()
        core.wait(0.2)


# -----------------------------------------------------------------------
# 7. LEARNING PHASE: recreate the order seen in the bandit phase
#    (run twice per block: before and after the TI phase)
# -----------------------------------------------------------------------
def run_learning_phase(block_num, condition, recall_phase, part_num):
    if recall_phase == 'pre_ti':
        title = f"Block {block_num} - Part {part_num}: Learn"
        intro = ("Now recreate the order of the objects, from left to right,\n"
                 "exactly as you saw them in the exploration part.\n")
    else:
        title = f"Block {block_num} - Part {part_num}: Learn again"
        intro = ("One more time: recreate the order of the objects, from left to right,\n"
                 "exactly as you saw them in the exploration part.\n")

    show_instructions(
        f"{title}\n\n"
        f"{intro}"
        "Click the objects one by one.\n\n"
        "Press SPACE to begin.")

    spacing = BANDIT_SPACING
    total_width = spacing * (N_STIM - 1)
    top_y = 140
    bottom_y = -120
    slot_xs = [-total_width / 2 + i * spacing for i in range(N_STIM)]
    box_size = RECALL_STIM_SIZE + 10

    palette_order = SEQUENCE_ORDER.copy()
    random.shuffle(palette_order)
    palette_stims = {}
    for i, stim_id in enumerate(palette_order):
        palette_stims[stim_id] = visual.ImageStim(win, image=STIM_FILES[stim_id],
                                                   pos=(slot_xs[i], bottom_y),
                                                   size=(RECALL_STIM_SIZE, RECALL_STIM_SIZE))

    slot_boxes = [visual.Rect(win, width=box_size, height=box_size, pos=(slot_xs[i], top_y),
                               lineColor='black', fillColor=None) for i in range(N_STIM)]
    instr = visual.TextStim(win, text='Click the objects, in order, from left to right',
                             pos=(0, 280), color='black', height=24)
    placed = []
    placed_imgs = []

    recall_clock = core.Clock()
    recall_clock.reset()
    mouse.clickReset()

    while len(placed) < N_STIM:
        instr.draw()
        for box in slot_boxes:
            box.draw()
        for pimg in placed_imgs:
            pimg.draw()
        for stim_id, img in palette_stims.items():
            if stim_id not in placed:
                img.draw()
        quit_button.draw()
        quit_label.draw()
        win.flip()
        check_quit()

        if mouse.getPressed()[0] == 1:
            for stim_id, img in palette_stims.items():
                if stim_id in placed:
                    continue
                if img.contains(mouse):
                    rt = recall_clock.getTime()
                    slot_idx = len(placed)
                    correct_stim = SEQUENCE_ORDER[slot_idx]
                    is_correct = int(stim_id == correct_stim)

                    placed.append(stim_id)
                    placed_imgs.append(visual.ImageStim(win, image=STIM_FILES[stim_id],
                                                         pos=(slot_xs[slot_idx], top_y),
                                                         size=(RECALL_STIM_SIZE, RECALL_STIM_SIZE)))

                    recall_writer.writerow([participant_id, block_num, condition, recall_phase,
                                            slot_idx, stim_id,
                                            correct_stim, is_correct, round(rt, 3)])
                    recall_clock.reset()
                    wait_for_mouse_release()
                    break

    n_correct = sum(1 for i, st in enumerate(placed) if st == SEQUENCE_ORDER[i])
    score_text = visual.TextStim(win, text=f'You got {n_correct}/{N_STIM} correct!',
                                  pos=(0, -280), color='black', height=28)
    for box in slot_boxes:
        box.draw()
    for pimg in placed_imgs:
        pimg.draw()
    score_text.draw()
    win.flip()
    core.wait(2.0)


# -----------------------------------------------------------------------
# 8. MAIN EXPERIMENT LOOP
# -----------------------------------------------------------------------
def cleanup_and_quit():
    recall_file.close()
    bandit_file.close()
    ti_file.close()
    win.close()
    core.quit()


run_introduction()

for block_num, condition in enumerate(BLOCK_ORDER, start=1):
    STIM_IDS = BLOCK_STIM_IDS[block_num]
    SEQUENCE_ORDER = BLOCK_SEQUENCE_ORDER[block_num]

    show_instructions(f"Block {block_num} of {N_BLOCKS}\n\n"
                      "You will see a new set of 7 objects in this block.\n\n"
                      "Press SPACE to begin.")

    run_bandit_phase(block_num, condition)                                    # part 1: 5 bandit screens
    run_learning_phase(block_num, condition, recall_phase='pre_ti', part_num=2)   # part 2: recall BEFORE TI
    run_ti_phase(block_num, condition, part_num=3)                            # part 3: all pairs x3
    run_learning_phase(block_num, condition, recall_phase='post_ti', part_num=4)  # part 4: recall AFTER TI

show_instructions("You're all done!\n\nThank you for participating.\n\nPress SPACE to exit.")

cleanup_and_quit()
