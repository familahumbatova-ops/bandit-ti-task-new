[README.md](https://github.com/user-attachments/files/34035501/README.md)
# TI + Bandit Experiment (browser version)

This repository converts the PsychoPy `TI_bandit_task_v4.py` task into a browser-based experiment using JavaScript and jsPsych 8. It is intended as a research prototype; test it thoroughly before collecting participant data.

## Preserved task structure

- Three blocks: `linear_positive`, `linear_negative`, and `random`, with block order shuffled for each participant.
- Seven unique stimuli per block (21 total); hidden ranks are randomized separately for each block and participant.
- Five bandit screens per block; participants select three objects per screen and can inspect the revealed rewards until continuing.
- Sequence recall before TI and after TI.
- All 21 unordered stimulus pairs shown three times per block (63 TI trials), shuffled, with randomized left/right placement and correctness feedback.
- Four CSV outputs (hierarchy, bandit, TI, recall) plus a JSON bundle.

## 1. Add the 21 stimulus images

The HTML expects these files (JPG is the default):

```text
images/
  block1/
    stim1.jpg  stim2.jpg  stim3.jpg stim4.jpg  stim5.jpg  stim6.jpg  stim7.jpg
  block2/
    stim1.jpg  stim2.jpg  stim3.jpg  stim4.jpg stim5.jpg  stim6.jpg  stim7.jpg
  block3/
    stim1.jpg  stim2.jpg stim3.jpg  stim4.jpg stim5.jpg  stim6.jpg  stim7.jpg
```

Use your actual stimulus images, not placeholder pictures. Keep the same image set and naming scheme across participants. If you use JPEG, update `stimPath()` in `index.html` from `.png` to `.jpg` (or add extension fallback logic). Images are served publicly by GitHub Pages, so do not place confidential or copyrighted images there unless you have permission.

## 2. Test locally

You need internet access because jsPsych and its keyboard-response plugin are loaded from jsDelivr CDN. From the repository folder, start a local static server:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>. Test all three blocks using a test participant ID, and verify the five downloads at the end. Do not use a real participant ID during testing.

## 3. Publish on GitHub Pages

1. Sign in to GitHub and create a new repository, e.g. `ti-bandit-experiment`.
2. Choose **Public** only if you are comfortable making the code and stimulus images public. A private repository can be used for code collaboration, but Pages availability/settings depend on your GitHub plan and repository configuration.
3. Upload `index.html`, `README.md`, and the complete `images/` folder with all 21 image files. Optionally upload the original Python file in a separate `source/` folder for reference; the browser does not run it.
4. Open the repository's **Settings → Pages**.
5. Under **Build and deployment**, select **Deploy from a branch**.
6. Select branch `main` and folder `/ (root)`, then save.
7. Wait for the published URL to appear on the Pages settings screen. It will usually look like `https://YOUR-USERNAME.github.io/ti-bandit-experiment/`.
8. Open that URL on a different browser/device and complete a full test run. Confirm every stimulus image loads and CSV files download.

### Git command-line alternative

Run these commands inside the project folder after creating an empty GitHub repository (replace the URL with your own repository URL):

```bash
git init
git add index.html README.md images
git commit -m "Add browser-based TI bandit experiment"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ti-bandit-experiment.git
git push -u origin main
```

Then enable GitHub Pages using the settings above.

## 4. How data are saved

The page records responses in browser memory and periodically backs up a copy to `localStorage` in that participant's browser. At completion it offers these downloads:

- `*_hierarchy.csv`: stimulus IDs, true rank, sequence position, block and condition.
- `*_bandit.csv`: one row per selected object, reward and reaction time.
- `*_ti.csv`: one row per TI choice, pair, side, reward levels, correctness and reaction time.
- `*_recall.csv`: one row per recall click, target slot and correctness.
- `*_all_data.json`: all tables together.

**GitHub Pages does not receive these data.** The participant must download the files and transfer them to the researcher using an approved secure method. Do not ask participants to upload identifiable or sensitive research data to a public GitHub repository. For remote studies, use a secure institution-approved database/server (e.g. a configured REDCap or other approved research platform) and implement authenticated submission; this starter does not include a backend.

## 5. Research and implementation notes

- The current browser implementation follows the reward arrays and position-based reward mapping in the supplied Python script. In the supplied script, all five screens use the same mean arrays for each condition.
- The source Python file describes images as randomizable stimulus sets but its code assigns fixed `stim1`–`stim7` IDs within each block; the browser version retains fixed file identity and randomizes the hidden rank/order.
- The original script uses Gaussian noise for linear conditions and no noise for the random condition. This is preserved.
- The browser version is a conversion, not a validated scientific equivalent. Confirm timing, display size, response handling, reward mapping, randomization, and data columns with the PI before recruitment. Browser reaction times vary by device/browser and should not be assumed equivalent to PsychoPy timing.
- The script's task wording says objects are “bigger,” while the decision rule is based on reward levels. The browser version uses reward-oriented wording to make the response rule clear; change the heading if exact wording is required by the approved protocol.
- Use an ethics-approved participant information/consent page before the task if required. GitHub Pages is public hosting; the code and images are visible to anyone who can access the page.
