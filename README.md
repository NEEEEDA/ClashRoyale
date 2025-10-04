<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Windows (since that's the only OS it works on right now)
* VSCode (unless you're more familiar with other code editors)
* [Docker](https://www.docker.com/)
* [Roboflow Account](https://www.roboflow.com/)
* [BlueStacks](https://www.bluestacks.com/download.html)
* [Python 3.12](https://www.python.org/downloads/windows/)
* inference_sdk
  ```
  pip install inference-sdk
  ```
* PyTorch
  ```
  pip install torch
  ```
* PyAutoGUI
  ```
  pip install PyAutoGUI
  ```
* NumPy
  ```
  pip install numpy
  ```
  
### Installation

1. Create a Roboflow account as well as a workspace, then get your API key
![roboflow-tutorial](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExNG1uNmtiaTAzamVvNnQwc2k3NDQzOXhzcmhxc2prZTBzM3U3YWY5YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/1KLeC2gw8pimdhH61C/giphy.gif)
2. Clone the repo
   ```sh
   git clone https://github.com/krazyness/CRBot-public.git
   ```
3. Set up your environment variables:
   - Edit `.env` and replace `your_roboflow_api_key_here` with your actual Roboflow private API key
   ```bash 
   # Edit .env and add your API key
   ROBOFLOW_API_KEY=your_actual_api_key_here
   ```
4. Fork both workflows:
[Troop Detection](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiTEx3TjlnOEduenBjWmVYSktKYzEiLCJ3b3Jrc3BhY2VJZCI6Ik5vVUlkM3gyYWRSU0tqaURrM0ZMTzlBSmE1bzEiLCJ1c2VySWQiOiJOb1VJZDN4MmFkUlNLamlEazNGTE85QUphNW8xIiwiaWF0IjoxNzUzODgxNTcyfQ.-ZO7pqc3mBX6W49-uThUSBLdUaCRzM9I8exfEu6-lo8)
[Card Detection](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiMEFmeVpSQ3FSS1dhV1J5QTFGNkciLCJ3b3Jrc3BhY2VJZCI6InJtZHNiY2xlU292aEEwNm15UDFWIiwidXNlcklkIjoiTm9VSWQzeDJhZFJTS2ppRGszRkxPOUFKYTVvMSIsImlhdCI6MTc1Mzg4MjE4Mn0.ceYp4JZoNSIrDkrX2vuc9or3qVakNexseYEgacIrfLA)

![Workspace-Fork-Tutorial](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExM2g4NTZ2MDlkM3JpdGl5emgxNHc3ejJudTRiMDFnbXFkNmxnNzgyeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/94yt100mNmhRIPRu3d/giphy.gif)

5. Get the workspace names from your forked workflows and update your `.env` file:
   - For the Troop Detection workspace, update `WORKSPACE_TROOP_DETECTION=your-troop-workspace-name`
   - For the Card Detection workspace, update `WORKSPACE_CARD_DETECTION=your-card-workspace-name`
   
   Your `.env` file should look like:
   ```bash
   ROBOFLOW_API_KEY=your_actual_api_key_here
   WORKSPACE_TROOP_DETECTION=workspace-your-troop-name
   WORKSPACE_CARD_DETECTION=workspace-your-card-name
   ```
6. Open Docker, and open the terminal on the bottom right, and install inference-cli (don't worry the terminal isn't stuck, it takes a long time)
   ```js
   pip install inference-cli
   ```
7. Start the Inference Server
   ```js
   inference server start
   ```
8. Open http://localhost:9001/, and it should take you to the Roboflow Inference page.
9. Open BlueStacks, and open the "multi-instance manager" (should be the third icon above the Discord icon, or its in the 3 dots), and create a fresh Pie 64-bit instance.
10. Start the Pie 64-bit instance, open Google Play Store, and install Clash Royale.
11. Optional: remove the ads on the left by opening settings (gear), > Preferences > Allow BlueStacks to show Ads during gameplay (disabled)
12. Open Clash Royale, resize and position the window like so (stretched and to the right-most of the screen)

![BlueStacks-window-tutorial](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3k2enMwY3E4cHJ0MDhnbmg1NnhsaDI3bGhmazJ4aXlxczFkamFxeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/y8yXKqwN40cdcr4yR5/giphy.gif)

13. Log in (or make a new) account on Clash Royale, click on battle, then run train.py, but immediately after, make sure the BlueStacks emulator is the front-most window.

**NOTE:** The bot is broken right now, with it not handling "play again" correctly, as well as some minor bugs in gameplay. You can ask me any questions at the contacts page, or make contributions at the contributing page!

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

![Demo](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExaXFtZmh1eG10amdidGhuMXBlb3dyaWZ3MjB5a2d6ZXluYXN6MTY0ZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/SFDKIvtoRL1Og4S7fn/giphy.gif)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

