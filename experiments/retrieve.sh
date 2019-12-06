for s in r; do
rsync -avz $u@$s.psgd.morty-pg0.utah.cloudlab.us:experiments .
done
