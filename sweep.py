import itertools as it
import os
from time import time, sleep
import yaml

import torch
import vizdoom as vzd
import skimage
import numpy as np
import wandb
from tqdm import trange

from agents import *


def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, (30, 45))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def create_game_environment(config_file_path):
    print("Creating game environment ...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Game environment initialized ...")
    return game


def test_agent(game, agent, actions, frame_repeat, test_episodes_per_epoch=10):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    test_scores = []
    for _ in trange(test_episodes_per_epoch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)
            game.make_action(actions[best_action_index], frame_repeat)
        reward = game.get_total_reward()
        test_scores.append(reward)

    test_scores = np.array(test_scores)
    return test_scores


def train_agent(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch, save_model, model_path):
    """
    Trains the DQN Agent by running num_epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0

        print("\n\nEpoch #" + str(epoch + 1))

        for _ in trange(steps_per_epoch):
            state = preprocess(game.get_state().screen_buffer)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)
        print(
            "\tResults (Train): mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test_scores = test_agent(game, agent, actions, frame_repeat)
        print(
            "\tResults (Test): mean: {:.1f} +/- {:.1f},".format(
                test_scores.mean(), test_scores.std()
            ),
            "min: %.1f" % test_scores.min(),
            "max: %.1f" % test_scores.max(),
        )

        wandb.log({
            "train_score": train_scores.mean(),
            "train_score_std": train_scores.std(),
            "test_score": test_scores.mean(),
            "test_score_std": test_scores.std(),
        })

        if save_model:
            print("Saving the network weights to:", model_path)
            torch.save(agent.q_net, model_path)
        print("Total elapsed time: %.2f minutes" %
              ((time() - start_time) / 60.0))

    game.close()
    return agent, game


if __name__ == "__main__":
    with open("./config/config-random.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # Read wandb config from yaml
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    discount_factor = wandb.config.discount_factor
    memory_size = wandb.config.memory_size
    frame_repeat = wandb.config.frame_repeat
    steps_per_epoch = wandb.config.steps_per_epoch
    epsilon_decay = wandb.config.epsilon_decay

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    print(f"Using device={device} ...")

    # Setup and create the game environment
    config_file_path = os.path.join(vzd.scenarios_path, "rocket_basic.cfg")
    game = create_game_environment(config_file_path)
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = DQNAgent(
        action_size=len(actions),
        lr=lr,
        batch_size=batch_size,
        memory_size=memory_size,
        discount_factor=discount_factor,
        epsilon_decay=epsilon_decay,
        load_model=False,
        device=device,
    )

    # Run the training for the set number of epochs
    skip_learning = False
    if not skip_learning:
        agent, game = train_agent(
            game,
            agent,
            actions,
            num_epochs=20,
            frame_repeat=frame_repeat,
            steps_per_epoch=steps_per_epoch,
            save_model=False,
            model_path="checkpoints/doom.pth"
        )
        print("======================================")
        print("Training finished.")
