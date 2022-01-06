import os
import time

import gpustat

from src.run.pipeline import send_email

if os.fork():  # parent
    print('\033[2J')
    while True:
        print('\033[0;0H')
        gpustat.print_gpustat()
        time.sleep(1)
else:  # child
    gpus = {}
    max_downtime = 300
    sleep_time = 5
    reset_every = 3
    reset_counter = 0
    while True:
        if reset_counter == reset_every:
            print('\033\143')
            reset_counter = 0
        reset_counter += 1

        time.sleep(sleep_time)
        for gpu in gpustat.new_query().jsonify()['gpus']:
            i = gpu['index']
            gpus.setdefault(i, {'downtime': 0, 'email_sent': False})
            if gpu['utilization.gpu'] <= 10:
                gpus[i]['downtime'] += sleep_time
            else:
                gpus[i]['downtime'] = 0

            if not gpus[i]['email_sent'] and gpus[i]['downtime'] >= max_downtime:
                send_email(
                    f'[ALERT] GPU {i} unutilized for {max_downtime} seconds EOM')
                gpus[i]['email_sent'] = True
