source experiments/common.sh

## Parallel SGD 1
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e2/psgd.json --worker_idx 0 &> log.out' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e2/psgd.json --worker_idx 1 &> log.out' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e2/psgd.json --worker_idx 2 &> log.out' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e2/psgd.json --worker_idx 3 &> log.out' &
wait $master
##

bash experiments/retrieve.sh
