for mc in 1000 5000 10000 15000 20000; do
  python sparse_ridge_e2006.py \
    --train ../data/e2006/E2006.train.bz2 --test ../data/e2006/E2006.test.bz2 --colnorm l2 \
    --lambda2 0.01 --k-list 20 \
    --run-greedy --m-cand ${mc} \
    --outdir out_mcand_${mc} --seed 0
done


for l2 in 0.001 0.01 0.1 1.0 10.0; do
  python sparse_ridge_e2006.py \
    --train ../data/e2006/E2006.train.bz2 --test ../data/e2006/E2006.test.bz2 --colnorm l2 \
    --lambda2 ${l2} --k-list 20 \
    --run-greedy --m-cand 5000 \
    --run-iht --iht-step niht --iht-debias --iht-max-iter 20 \
    --outdir out_lambda2_${l2} --seed 0
done