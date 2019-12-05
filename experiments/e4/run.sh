source experiments/common.sh

if false; then
## Parallel SGD 1
remote 'r' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e4/psgd1.json --worker_idx 0 '
##

## Parallel SGD 2
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e4/psgd2.json --worker_idx 0 ' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e4/psgd2.json --worker_idx 1' &
wait $master
##

## Parallel SGD 3
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e4/psgd3.json --worker_idx 0 ' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e4/psgd3.json --worker_idx 1' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e4/psgd3.json --worker_idx 2' &
wait $master
##

## Parallel SGD 4
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e4/psgd4.json --worker_idx 0 ' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e4/psgd4.json --worker_idx 1' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e4/psgd4.json --worker_idx 2' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e4/psgd4.json --worker_idx 3' &
wait $master
##

## Parallel SGD 5
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'
remote 'client-0-3' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e4/psgd5.json --worker_idx 0 ' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e4/psgd5.json --worker_idx 1' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e4/psgd5.json --worker_idx 2' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e4/psgd5.json --worker_idx 3' &
sleep 0.7
remote 'client-0-3' 'python3 dist.py --config_file experiments/e4/psgd5.json --worker_idx 4' &
wait $master
##

## Parallel SGD 6
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'
remote 'client-0-3' 'killall -9 python3'
remote 'client-0-4' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e4/psgd6.json --worker_idx 0 ' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e4/psgd6.json --worker_idx 1' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e4/psgd6.json --worker_idx 2' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e4/psgd6.json --worker_idx 3' &
sleep 0.7
remote 'client-0-3' 'python3 dist.py --config_file experiments/e4/psgd6.json --worker_idx 4' &
sleep 0.7
remote 'client-0-4' 'python3 dist.py --config_file experiments/e4/psgd6.json --worker_idx 5' &
wait $master
##

## Parallel SGD 7
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'
remote 'client-0-3' 'killall -9 python3'
remote 'client-0-4' 'killall -9 python3'
remote 'client-0-5' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e4/psgd7.json --worker_idx 0 ' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e4/psgd7.json --worker_idx 1' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e4/psgd7.json --worker_idx 2' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e4/psgd7.json --worker_idx 3' &
sleep 0.7
remote 'client-0-3' 'python3 dist.py --config_file experiments/e4/psgd7.json --worker_idx 4' &
sleep 0.7
remote 'client-0-4' 'python3 dist.py --config_file experiments/e4/psgd7.json --worker_idx 5' &
sleep 0.7
remote 'client-0-5' 'python3 dist.py --config_file experiments/e4/psgd7.json --worker_idx 6' &
wait $master
##
fi

## Parallel SGD 8
remote 'r' 'killall -9 python3'
remote 'client-0-0' 'killall -9 python3'
remote 'client-0-1' 'killall -9 python3'
remote 'client-0-2' 'killall -9 python3'
remote 'client-0-3' 'killall -9 python3'
remote 'client-0-4' 'killall -9 python3'
remote 'client-0-5' 'killall -9 python3'
remote 'client-0-6' 'killall -9 python3'

remote 'r' 'python3 dist.py --config_file experiments/e4/psgd7.json --worker_idx 0 &> log.out' &
master=$!
sleep 0.7
remote 'client-0-0' 'python3 dist.py --config_file experiments/e4/psgd8.json --worker_idx 1' &
sleep 0.7
remote 'client-0-1' 'python3 dist.py --config_file experiments/e4/psgd8.json --worker_idx 2' &
sleep 0.7
remote 'client-0-2' 'python3 dist.py --config_file experiments/e4/psgd8.json --worker_idx 3' &
sleep 0.7
remote 'client-0-3' 'python3 dist.py --config_file experiments/e4/psgd8.json --worker_idx 4' &
sleep 0.7
remote 'client-0-4' 'python3 dist.py --config_file experiments/e4/psgd8.json --worker_idx 5' &
sleep 0.7
remote 'client-0-5' 'python3 dist.py --config_file experiments/e4/psgd8.json --worker_idx 6' &
sleep 0.7
remote 'client-0-6' 'python3 dist.py --config_file experiments/e4/psgd8.json --worker_idx 7' &
wait $master
##

bash experiments/retrieve.sh
