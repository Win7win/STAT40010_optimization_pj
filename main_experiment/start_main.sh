python sparse_ridge_e2006.py \
  --train ../data/e2006/E2006.train.bz2 --test ../data/e2006/E2006.test.bz2 \
  --colnorm l2 --lambda2 0.01 --k-list 10 20 50 \
  --run-greedy --m-cand 5000 \
  --run-iht --iht-step auto --iht-debias --iht-max-iter 20 \
  --run-iht --iht-step niht --iht-debias --iht-max-iter 20 \
  --run-fista --fista-matchk --fista-max-iter 300 \
  --outdir out_main --seed 0