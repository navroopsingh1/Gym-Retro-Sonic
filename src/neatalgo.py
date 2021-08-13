import retro
import numpy as np
import cv2 as cv
import neat
import pickle

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.act1')
xpos_max = 0
imgarray = []
xpos_end = 0

def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
        obs = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False

        while not done:
            env.render()
            frame += 1
            obs = cv.resize(obs, (inx, iny))
            obs = cv.cvtColor(obs, cv.COLOR_BGR2GRAY)
            obs = np.reshape(obs, (inx, iny))

            for x in obs:
                for y in x:
                    imgarray.append(y)

            nnOutput = net.activate(imgarray)
            
            obs, rew, done, info = env.step(nnOutput)
            imgarray.clear()

            xpos = info['x']
            xpos_end = info['screen_x_end']

            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos

            if xpos == xpos_end and xpos > 500:
                fitness_current =+ 100000

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            if done or counter == 250:
                done = True
            
            genome.fitness = fitness_current
    

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'config-feedforward')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
