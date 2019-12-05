source experiments/common.sh

## Local SGD
ssh -o 'StrictHostKeyChecking=no' -o 'ControlMaster=auto' -o 'ControlPersist=2m' \
  -o 'ControlPath=~/.ssh/cm-%r@%h:%p' $u@r.psgd.morty-pg0.utah.cloudlab.us \
  python3 dist.py --config_file experiments/e1/sgd.json --local
##

## Parallel SGD 1
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e1/psgd1.json --worker_idx 0 ' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e1/psgd1.json --worker_idx 1' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e1/psgd1.json --worker_idx 2 ' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e1/psgd1.json --worker_idx 3' &
wait $master
##

## Parallel SGD 2
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e1/psgd2.json --worker_idx 0 ' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e1/psgd2.json --worker_idx 1' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e1/psgd2.json --worker_idx 2 ' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e1/psgd2.json --worker_idx 3' &
wait $master
##

## Parallel SGD 10
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e1/psgd10.json --worker_idx 0 ' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e1/psgd10.json --worker_idx 1' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e1/psgd10.json --worker_idx 2 ' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e1/psgd10.json --worker_idx 3' &
wait $master
##

bash experiments/retrieve.sh
