import logging
from rlqp_train.qp_env import QPEnv
from rlqp_train.ddpg import DDPG

if '__main__' == __name__:
    logging.basicConfig(level=logging.DEBUG)

    env = QPEnv(
        eps = 1e-6,
        step_reward = -1,
        iterations_per_step=200)

    env.add_benchmark_problem_class("Random QP", 10, 100)
    # env.add_benchmark_problem_class("Eq QP", 10, 2000) # Solves too quickly
    env.add_benchmark_problem_class("Portfolio", 5, 15)
    env.add_benchmark_problem_class("Lasso", 10, 20)
    env.add_benchmark_problem_class("SVM", 10, 20)
    # env.add_benchmark_problem_class("Huber", 10, 200) # Solves too quickly
    env.add_benchmark_problem_class("Control", 10, 10)
    
    ddpg = DDPG(
        save_dir = 'experiments/ddpg_train',
        env = env,
        hparams = dict(
            replay_size = int(1e6), # TODO 1e8
            pi_lr = 1e-3,
            q_lr = 1e-3,
            lr_decay_rate = 0.999,
            steps_per_epoch = 2000,
            hidden_sizes = (128, 128, 128),
            num_test_episodes = 16,
            num_epochs = 50,
            max_ep_len = 100,
            update_every = 1000,
            batch_size = 100,
            seed = 5,
            gamma = 0.99,
            polyak = 0.995,
            act_noise = 2.0,
            update_after = 1000,
            start_steps = 5000))
        # start_act_noise = 1.0)

    #exp_name = "benchmarks"
    
    # ddpg.load_state(exp_name)
    ddpg.train()
    #ddpg.save_state(exp_name)
    
                    
                
