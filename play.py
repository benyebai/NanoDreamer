"""Pygame window for the pellet maze."""

import argparse
from pathlib import Path

import numpy as np
import pygame

from tinydreamer.game import LEVEL, PelletMaze
from tinydreamer.actor import Actor
from mininn.optimizers import Adam
from mininn.checkpoint import load_checkpoint

CELL_SIZE = 64
BOTTOM_SPACE = 76
SPRITE_FOLDER = Path(__file__).parent / "assets" / "sprites"
DIRECTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]


def load_sprites():
    sprites = []
    for direction in DIRECTIONS:
        filename = SPRITE_FOLDER / ("pacman_" + direction + ".png")
        image = pygame.image.load(str(filename)).convert_alpha()
        image = pygame.transform.scale(image, (CELL_SIZE - 12, CELL_SIZE - 12))
        sprites.append(image)
    return sprites


def draw_game(screen, game, sprites, direction, font, small_font, message, ghost_sprite=None):
    screen.fill((9, 11, 24))

    for row in range(len(LEVEL)):
        for col in range(len(LEVEL[row])):
            rectangle = pygame.Rect(
                col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE
            )
            if LEVEL[row][col] == "#":
                pygame.draw.rect(screen, (34, 74, 165), rectangle.inflate(-3, -3))
            else:
                pygame.draw.rect(screen, (17, 20, 39), rectangle.inflate(-2, -2))
            if (row, col) in game.pellets:
                pygame.draw.circle(screen, (255, 218, 120), rectangle.center, 6)

    player_row, player_col = game.player
    player_image = sprites[direction]
    image_rectangle = player_image.get_rect(
        center=(
            player_col * CELL_SIZE + CELL_SIZE // 2,
            player_row * CELL_SIZE + CELL_SIZE // 2,
        )
    )
    screen.blit(player_image, image_rectangle)
    if game.with_ghost:
        ghost_row, ghost_col = game.ghost
        ghost_rectangle = ghost_sprite.get_rect(
            center=(ghost_col * CELL_SIZE + CELL_SIZE // 2,
                    ghost_row * CELL_SIZE + CELL_SIZE // 2)
        )
        screen.blit(ghost_sprite, ghost_rectangle)

    bottom = len(LEVEL) * CELL_SIZE
    pygame.draw.rect(
        screen, (12, 17, 36), (0, bottom, screen.get_width(), BOTTOM_SPACE)
    )
    score_text = font.render("Score: %.2f" % game.total_reward, True, (255, 239, 177))
    pellet_text = font.render("Pellets: %d" % len(game.pellets), True, (221, 230, 255))
    screen.blit(score_text, (16, bottom + 8))
    screen.blit(
        pellet_text, (screen.get_width() - pellet_text.get_width() - 16, bottom + 8)
    )
    if message == "":
        message = "Arrows/WASD: move    R: restart    Esc: quit"
    help_text = small_font.render(message, True, (172, 190, 224))
    screen.blit(
        help_text, ((screen.get_width() - help_text.get_width()) // 2, bottom + 45)
    )
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--random", action="store_true")
    choices.add_argument("--actor", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--ghost", action="store_true")
    parser.add_argument("--local-view", action="store_true", help="use the older 3x3 ghost observation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    if args.local_view and not args.ghost:
        parser.error("--local-view requires --ghost")
    game = PelletMaze(args.max_steps, with_ghost=args.ghost, full_view=not args.local_view)
    game.reset()
    random_generator = np.random.default_rng(args.seed)
    actor = None
    if args.actor:
        actor = Actor(args.seed, input_size=int(np.prod(game.observation_shape)))
        optimizer = Adam(actor.parameters(), learning_rate=0.001)
        checkpoint_name = args.checkpoint
        if checkpoint_name is None:
            if args.local_view:
                checkpoint_name = "checkpoints/ghost_local_actor_critic_actor.npz"
            elif args.ghost:
                checkpoint_name = "checkpoints/ghost_actor_critic_actor.npz"
            else:
                checkpoint_name = "checkpoints/actor_critic_actor.npz"
        checkpoint = Path(checkpoint_name)
        if not checkpoint.is_absolute():
            checkpoint = Path(__file__).parent / checkpoint
        load_checkpoint(checkpoint, actor.network, optimizer)

    pygame.init()
    width = len(LEVEL[0]) * CELL_SIZE
    height = len(LEVEL) * CELL_SIZE + BOTTOM_SPACE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("TinyDreamer Pellet Maze")
    sprites = load_sprites()
    ghost_sprite = None
    if args.ghost:
        path = SPRITE_FOLDER / "ghost_red.png"
        image = pygame.image.load(str(path)).convert_alpha()
        ghost_sprite = pygame.transform.scale(image, (CELL_SIZE - 12, CELL_SIZE - 12))
    font = pygame.font.SysFont(None, 29)
    small_font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    keys = {
        pygame.K_UP: 0,
        pygame.K_w: 0,
        pygame.K_RIGHT: 1,
        pygame.K_d: 1,
        pygame.K_DOWN: 2,
        pygame.K_s: 2,
        pygame.K_LEFT: 3,
        pygame.K_a: 3,
    }
    direction = 1
    base_message = "Actor playing - R restart - Esc quit" if args.actor else ""
    message = base_message
    last_random_move = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    message = base_message
                elif not (args.random or args.actor) and event.key in keys:
                    if not game.terminated and not game.truncated:
                        direction = keys[event.key]
                        game.step(direction)

        if (args.random or args.actor) and pygame.time.get_ticks() - last_random_move > 220:
            if not game.terminated and not game.truncated:
                if args.actor:
                    direction = actor.choose_action(game.observe())
                else:
                    direction = int(random_generator.integers(0, 4))
                game.step(direction)
            last_random_move = pygame.time.get_ticks()

        if game.caught:
            message = "The ghost caught you! Press R to restart."
        elif game.won:
            message = "All pellets eaten! Press R to restart."
        elif game.truncated:
            message = "Out of moves. Press R to restart."

        draw_game(screen, game, sprites, direction, font, small_font, message, ghost_sprite)
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
