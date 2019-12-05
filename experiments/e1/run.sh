bash experiments/common.sh

ssh -o 'StrictHostKeyChecking=no' -o 'ControlMaster=auto' -o 'ControlPersist=2m' \
  -o 'ControlPath=~/.ssh/cm-%r@%h:%p' $u@r.psgd.morty-pg0.utah.cloudlab.us \
  python3 dist.py --config_file experiments/e1/sgd.json --local
