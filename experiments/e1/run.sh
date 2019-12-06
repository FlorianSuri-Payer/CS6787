source experiments/common.sh

## Local SGD
remote 'r' 'killall -9 python3'
remote 'r' 'python3 dist.py --config_file experiments/e1/sgd.json --local &> log.out'
##

## Parallel SGD 1
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e1/psgd1.json --worker_idx 0 &> log.out' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e1/psgd1.json --worker_idx 1 &> log.out' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e1/psgd1.json --worker_idx 2 &> log.out' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e1/psgd1.json --worker_idx 3 &> log.out' &
wait $master
##

## Parallel SGD 2
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e1/psgd2.json --worker_idx 0 &> log.out' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e1/psgd2.json --worker_idx 1 &> log.out' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e1/psgd2.json --worker_idx 2 &> log.out' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e1/psgd2.json --worker_idx 3 &> log.out' &
wait $master
##

## Parallel SGD 10
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e1/psgd10.json --worker_idx 0 >& log.out' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e1/psgd10.json --worker_idx 1 &> log.out' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e1/psgd10.json --worker_idx 2 &> log.out' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e1/psgd10.json --worker_idx 3 &> log.out' &
wait $master
##

bash experiments/retrieve.sh
