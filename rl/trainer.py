import os
import torch as th
import time
import numpy as np
import sys
from . import models
from .maml import MAML
from gym.spaces import Box
from .procrunner import ProcRunner
from .testrunner import TestRunner
from .maml_testrunner import MAMLTestRunner
from .online_test_runner2 import OnlineTestRunner
from .l2vecenv import l2vecenv

def get_best_valid_num(save_dir):
    maxi = 1
    maxv = -100
    i = 1
    with open(f'{save_dir}/valid/avg.txt', 'r') as valid_data:
        for data in valid_data.readlines():
            v = float(data.split('\t')[6])
            if v >= maxv:
                maxi = i
                maxv = v
            i += 1
    return maxi

def train(device, agent, env_name, train_dates, valid_dates, test_dates, env_args={}, log_interval=0.01, save_interval=0.1):
    save_path = f'logs/{agent.name}/model.pt'
    agent.load_model(save_path)

    env_args.update({
        'data_dates':train_dates, 
        'save_dir':f'logs/{agent.name}',
        'target_noise':True
        })
    valid_env_args = env_args.copy()
    valid_env_args['save_dir'] = f'logs/{agent.name}/valid'
    valid_env_args['data_dates'] = valid_dates
    valid_env_args['target_noise'] = False
    valid_env_args['time_noise'] = 0
    #valid_env_args['start_noise'] = 0
    #valid_env_args['epilen_min'] = valid_env_args['epilen_max']
    valid_env_args['window_mask_noise'] = 0
    
    test_env_args = valid_env_args.copy()
    test_env_args['save_dir'] = f'logs/{agent.name}/test'
    test_env_args['data_dates'] = test_dates

    isfake = env_args['fake']

    vec_env = l2vecenv(agent.nenvs, env_name, env_args)
    #test_vec_env = l2vecenv(env_name, valid_env_args)
    
    if isinstance(agent, MAML):
        valid_runner = MAMLTestRunner(agent, vec_env, env_name, valid_env_args)
        test_runner = MAMLTestRunner(agent, vec_env, env_name, test_env_args)
        test_env_args['data_dates'] = '20210700~20220700'
        test2_runner = MAMLTestRunner(agent, vec_env, env_name, test_env_args)
    else:
        valid_runner = TestRunner(agent, env_name, valid_env_args, deterministic=False, date_divide=1, repeat=2)
        test_runner = TestRunner(agent, env_name, test_env_args, deterministic=False, date_divide=1, repeat=2)
        test_env_args['data_dates'] = '20210700~20220700'
        test2_runner = TestRunner(agent, env_name, test_env_args, deterministic=False, date_divide=1, repeat=2)

    prevtime = time.time()
    saves = 1
    
    next_log = log_interval
    next_save = 0
    while next_log <= agent.progress:
        next_log += log_interval
    
    while next_save <= agent.progress:
        next_save += save_interval

    while os.path.isdir(f'logs/{agent.name}/valid/{saves}'):
        saves += 1

    start_time = prevtime = time.time()
    start_prog = agent.progress
    while agent.progress < 1:
        agent.train_epoch(vec_env)
        
        if agent.progress >= next_log:
            next_log += log_interval
            if isinstance(agent, MAML):
                inavg = np.average(vec_env.recent_scores['inner'])
                outavg = np.average(vec_env.recent_scores['outer'])
                agent.write_log('inner/score', inavg)
                agent.write_log('outer/score', outavg)
                print(f'Inner Average Score:\t{round(inavg,3)}')
                print(f'Outer Average Score:\t{round(outavg,3)}')
            else:
                avg = np.average(vec_env.recent_scores['default'])
                agent.write_log('Average_score', avg)
                print(f'Average Score:\t{round(avg,3)}')

            
            print(f'progress:\t{round(agent.progress * 100, 2)} %')
            currtime = time.time()
            time_passed = currtime - prevtime
            prevtime = currtime
            print(f"elapsed time:\t{round(time_passed, 3)} second")
            print(f'total elapsed time:\t{round((currtime-start_time)/3600,3)} hour')
            time_left = (1-agent.progress) * (currtime-start_time)/(agent.progress - start_prog)
            print(f"time left:\t{round(time_left/3600, 3)} hour")
            
            print('-----------------------------------------------------------')
            sys.stdout.flush()
 
        if agent.progress >= next_save:
            next_save += save_interval
            agent.save_model(f"logs/{agent.name}/valid/{saves}/model.pt")
            agent.save_model(save_path)
            
            agent.model.eval()

            if isinstance(agent.model, models.OracleACNet):
                agent.model.extractor.is_oracle = False
                #tmp = agent.model.extractor.history_mode
                #agent.model.extractor.history_mode = 'prob'
                print('test mode')

            valid_runner.set_savedir(f'logs/{agent.name}/valid/{saves}')
            valid_runner.run_test('valid')
            test_runner.set_savedir(f'logs/{agent.name}/test/{saves}')
            test_runner.run_test('test')
            #test2_runner.set_savedir(f'logs/{agent.name}/test2/{saves}')
            #test2_runner.run_test('test2')

            if isinstance(agent.model, models.OracleACNet):
                agent.model.extractor.is_oracle = True
                #agent.model.extractor.history_mode = tmp

                #valid_runner.set_savedir(f'logs/{agent.name}/valid_oracle/{saves}')
                #valid_runner.run_test('valid_oracle')
                #test_runner.set_savedir(f'logs/{agent.name}/test_oracle/{saves}')
                #test_runner.run_test('test_oracle')
                #test2_runner.set_savedir(f'logs/{agent.name}/test2_oracle/{saves}')
                #test2_runner.run_test('test2_oracle')

            saves += 1
            
            agent.model.train()

    agent.save_model(save_path)
    vec_env.close()

def test(device, agent, env_name, train_dates, test_dates, env_dict, saveim=False, load=True, savenum=0, deterministic=False, is_cont=False):
    if load:
        if savenum == 0:
            savenum = get_best_valid_num(f'logs/{agent.name}')
        print('loading')
        save_path = f'logs/{agent.name}/valid/{savenum}/model.pt'
        print(save_path)
        agent.load_model(save_path)
    
    test_env_dict = env_dict
    test_env_dict['data_dates'] = test_dates
    test_env_dict['target_noise'] = False
    test_env_dict['time_noise'] = 0
    test_env_dict['window_mask_noise'] = 0
    #test_env_dict['start_noise'] = 0
    #test_env_dict['epilen_min'] = test_env_dict['epilen_max']
    test_env_dict['saveim'] = saveim
    
    if is_cont:
        test_env_dict['save_dir'] = f'logs/{agent.name}/test_cont_22_3_3/'
    else:
        test_env_dict['save_dir'] = f'logs/{agent.name}/test_custom/'

    if is_cont:
        test_train_env_dict = test_env_dict.copy()
        test_train_env_dict['save_dir'] = f'logs/{agent.name}/test_cont/train/'
        test_train_env_dict['data_dates'] = train_dates
        vec_env = l2vecenv(agent.nenvs, env_name, test_train_env_dict)
        test_runner = OnlineTestRunner(agent, env_name, vec_env, test_env_dict)
        test_runner.run_test('test')
    else:
        if isinstance(agent, models.OracleACNet):
            agent.extractor.is_oracle = False
        
        test_runner = TestRunner(agent, env_name, test_env_dict, deterministic=deterministic, date_divide=1, repeat=2)
        test_runner.run_test('test_custom')


    test_runner.close()


