
count_t_0=$(ls -l ~/Documents/Deep_Learning/pytorch_imagenet/Task_2/train/0 | wc -l)
count_v_0=$(ls -l ~/Documents/Deep_Learning/pytorch_imagenet/Task_2/val/0 | wc -l)
count_t_1=$(ls -l ~/Documents/Deep_Learning/pytorch_imagenet/Task_2/train/1 | wc -l)
count_v_1=$(ls -l ~/Documents/Deep_Learning/pytorch_imagenet/Task_2/val/1 | wc -l)

if [ $count_t_0 > $count_t_1 ]
then
    diff_t=$(($count_t_0-$count_t_1))
    diff_v=$(($count_v_0-$count_v_1))
    check=$(echo 1)
else
    diff_t=$(($count_t_1-$count_t_0))
    diff_v=$(($count_v_1-$count_v_0))
    check=$(echo 1)
fi


cd ~/Documents/Deep_Learning/pytorch_imagenet/Task_2_balanced

if [ $check > 0 ]
then
    echo $check
    random_t=$(ls ~/Documents/Deep_Learning/pytorch_imagenet/Task_2_balanced/train/0 |sort -R |tail -$diff_t)
    random_v=$(ls ~/Documents/Deep_Learning/pytorch_imagenet/Task_2_balanced/val/0|sort -R |tail -$diff_v)

    cd ~/Documents/Deep_Learning/pytorch_imagenet/Task_2_balanced/train/0
    for f in $random_t ; do
      rm "$f"
    done
    cd ~/Documents/Deep_Learning/pytorch_imagenet/Task_2_balanced/val/0
    for f in $random_v ; do
      rm "$f"
    done
else
    echo $check
    random_t=$(ls ~/Documents/Deep_Learning/pytorch_imagenet/Task_2_balanced/train/1 |sort -R |tail -$diff_t)
    random_v=$(ls ~/Documents/Deep_Learning/pytorch_imagenet/Task_2_balanced/val/1|sort -R |tail -$diff_v)
    cd ~/Documents/Deep_Learning/pytorch_imagenet/Task_2_balanced/train/1
    for f in $random_t ; do
      rm "$f"
    done
    cd ~/Documents/Deep_Learning/pytorch_imagenet/Task_2_balanced/val/1
    for f in $random_v ; do
      rm "$f"
    done
fi
# "~/Documents/Deep_Learning/pytorch_imagenet/random_sampling.sh"
