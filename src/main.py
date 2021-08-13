import retro

def main():
    env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.act1')
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        print(rew)
    env.close()


if __name__ == "__main__":
    main()