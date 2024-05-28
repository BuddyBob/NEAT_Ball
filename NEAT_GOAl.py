import os
import pygame
import neat
from pygame.locals import *
import random
import math
import pickle

# Set up the display
screen_width = 800
screen_height = 800

# Game vars
BALL_RADIUS = 10
GOAL_RADIUS = 20
LAVA_RADIUS = 20
generation = 0
max_gens = 800

pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('NEAT Ball to Goal')

# Initialize font
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def draw_game(balls, goal_x, goal_y, lava_x, lava_y, highest_fitness_index=None, fitnesses=None, ball_x=None, ball_y=None, steps=None):
    screen.fill((255, 255, 255))
    
    pygame.draw.circle(screen, (255, 0, 0), (goal_x, goal_y), GOAL_RADIUS)
    pygame.draw.circle(screen, (0, 0, 255), (lava_x, lava_y), LAVA_RADIUS)
    
    if balls is not None:
        for i, (b_x, b_y) in enumerate(balls):
            color = (0, 255, 0) if i == highest_fitness_index else (0, 0, 0)
    
            pygame.draw.circle(screen, color, (b_x, b_y), BALL_RADIUS)
    
    if ball_x is not None and ball_y is not None:
        pygame.draw.circle(screen, (0, 0, 0), (ball_x, ball_y), BALL_RADIUS)
    
    # Render text
    global generation
    generation_text = font.render(f'Generation: {generation}', True, (0, 0, 0))
    screen.blit(generation_text, (10, 10))
    
    if fitnesses is not None and highest_fitness_index is not None:
        fitness_text = font.render(f'Fitness: {fitnesses[highest_fitness_index].fitness}', True, (0, 0, 0))
        screen.blit(fitness_text, (10, 40))

    if steps is not None:
        steps_text = font.render(f'Steps: {steps}', True, (0, 0, 0))
        screen.blit(steps_text, (10, 70))
    
    pygame.display.flip()

def genome_nn(genomes, config):
    global generation
    clock = pygame.time.Clock()
    render_interval = 40  # Render every 40 steps

    nets = []
    balls = []
    scored = []
    fitnesses = []

    goal_x, goal_y = 700, 700
    lava_x, lava_y = 600, 600
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        balls.append([screen_width // 2, screen_height // 2])
        scored.append(0)
        genome.fitness = 0
        fitnesses.append(genome)
        
    for steps in range(200):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        for i, (net, ball) in enumerate(zip(nets, balls)):
            ball_x, ball_y = ball
            inputs = (ball_x / screen_width, ball_y / screen_height, goal_x / screen_width, goal_y / screen_height)
            output = net.activate(inputs)
            move_x, move_y = output[0] * 2 - 1, output[1] * 2 - 1

            ball_x += int(move_x * 5)
            ball_y += int(move_y * 5)

            # Keep the ball within bounds
            if ball_x < BALL_RADIUS:
                ball_x = BALL_RADIUS + random.uniform(0, 1)
            elif ball_x > screen_width - BALL_RADIUS:
                ball_x = screen_width - BALL_RADIUS - random.uniform(0, 1)

            if ball_y < BALL_RADIUS:
                ball_y = BALL_RADIUS + random.uniform(0, 1)
            elif ball_y > screen_height - BALL_RADIUS:
                ball_y = screen_height - BALL_RADIUS - random.uniform(0, 1)

            dist_to_goal = distance(ball_x, ball_y, goal_x, goal_y)
            dist_to_lava = distance(ball_x, ball_y, lava_x, lava_y)

            # Reward for moving closer to the goal
            fitness = 1.0 / (dist_to_goal + 1)
            fitnesses[i].fitness += fitness

            # Dynamic penalty for getting closer to lava
            if dist_to_lava < BALL_RADIUS + LAVA_RADIUS:
                if scored[i] == 0:
                    fitnesses[i].fitness -= 200  # Strong penalty for hitting lava
                scored[i] = 1
                break
            elif dist_to_lava < 2 * LAVA_RADIUS:  # Penalty for being near the lava
                fitnesses[i].fitness -= 50 / (dist_to_lava + 1)  # Dynamic penalty increases as the ball gets closer

            # Check for goal scoring
            if dist_to_goal < BALL_RADIUS + GOAL_RADIUS:
                if scored[i] == 0:
                    fitnesses[i].fitness += 500  # Increase goal reward
                    fitnesses[i].fitness += 200 - steps  # Adjust speed reward
                    print(f"Goal Scored at Step {steps} by Genome {i} with fitness {fitnesses[i].fitness}")
                scored[i] = 1
                break
            
            balls[i] = [ball_x, ball_y]

        if steps % render_interval == 0:
            max_fitness_index = fitnesses.index(max(fitnesses, key=lambda x: x.fitness))
            draw_game(balls, goal_x, goal_y, lava_x, lava_y, max_fitness_index, fitnesses)

            clock.tick(30) 

    generation += 1
    best_fitness = max(fitnesses, key=lambda x: x.fitness).fitness
    print(f"Generation {generation} completed. Best fitness: {best_fitness}")

    # Verify goal scoring consistency
    scored_genomes = sum(scored)
    print(f"Generation {generation} scored genomes: {scored_genomes}")

def simulate_winner(winner_net):
    clock = pygame.time.Clock()
    goal_x, goal_y = 700, 700
    lava_x, lava_y = 600, 600
    ball_x, ball_y = screen_width // 2, screen_height // 2
    
    for steps in range(200):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
        
        inputs = (ball_x / screen_width, ball_y / screen_height, goal_x / screen_width, goal_y / screen_height)
        output = winner_net.activate(inputs)
        move_x, move_y = output[0] * 2 - 1, output[1] * 2 - 1

        ball_x += int(move_x * 5)
        ball_y += int(move_y * 5)

        # Keep the ball within bounds
        if ball_x < BALL_RADIUS:
            ball_x = BALL_RADIUS + random.uniform(0, 1)
        elif ball_x > screen_width - BALL_RADIUS:
            ball_x = screen_width - BALL_RADIUS - random.uniform(0, 1)

        if ball_y < BALL_RADIUS:
            ball_y = BALL_RADIUS + random.uniform(0, 1)
        elif ball_y > screen_height - BALL_RADIUS:
            ball_y = screen_height - BALL_RADIUS - random.uniform(0, 1)
            
        # hits lava
        if distance(ball_x, ball_y, lava_x, lava_y) < BALL_RADIUS + LAVA_RADIUS:
            print(f"Ball hit lava at Step {steps}")
            break

        draw_game(None, goal_x, goal_y, lava_x, lava_y, None, None, ball_x, ball_y, steps)

        if distance(ball_x, ball_y, goal_x, goal_y) < BALL_RADIUS + GOAL_RADIUS:
            print(f"Goal reached in {steps} steps")
            break

        clock.tick(30)

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(genome_nn, max_gens)  # Run for max_gens generations
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Save the winning genome
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    simulate_winner(winner_net)
    
    pygame.quit()
    
def run_saved_genome(config_file, genome_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    with open(genome_path, 'rb') as f:
        winner = pickle.load(f)
    
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    simulate_winner(winner_net)
    
    pygame.quit()

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'config-feedforward.txt')
    # run(config_path)
    
    # Uncomment the following line to run the saved genome
    run_saved_genome(config_path, 'best_genome2.pkl')

