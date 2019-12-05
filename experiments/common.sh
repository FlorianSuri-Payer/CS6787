remote () {
ssh -o 'StrictHostKeyChecking=no' -o 'ControlMaster=auto' -o 'ControlPersist=2m' \
  -o 'ControlPath=~/.ssh/cm-%r@%h:%p' $u@$1.psgd.morty-pg0.utah.cloudlab.us \
  $2
}
