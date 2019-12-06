for s in r client-0-0 client-0-1 client-0-2 client-0-3 client-0-4 client-0-5 client-0-6; do
rsync -avz $u@$s.psgd.morty-pg0.utah.cloudlab.us:experiments .
done
