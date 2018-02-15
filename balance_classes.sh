
count_t_0=$(ls -l ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/train/0 | wc -l)
count_v_0=$(ls -l ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/val/0 | wc -l)
count_t_1=$(ls -l ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/train/1 | wc -l)
count_v_1=$(ls -l ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/val/1 | wc -l)

if [ $count_t_0 -gt $count_t_1 ] #greater than
then
    diff_t=$(($count_t_0-$count_t_1))
    diff_v=$(($count_v_0-$count_v_1))
    check=$(echo 0)
else
    diff_t=$(($count_t_1-$count_t_0))
    diff_v=$(($count_v_1-$count_v_0))
    check=$(echo 1)
fi


cd ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals

if [ $check -eq 0 ]
then
    #Remove from 0
    echo $check is 0
    random_t=$(ls ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/train/0 |sort -R |tail -$diff_t)
    random_v=$(ls ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/val/0|sort -R |tail -$diff_v)

    cd ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/train/0
    for f in $random_t ; do
      rm "$f"
    done
    cd ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/val/0
    for f in $random_v ; do
      rm "$f"
    done
else
    #Remove from 1
    echo $check is 1
    random_t=$(ls ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/train/1 |sort -R |tail -$diff_t)
    random_v=$(ls ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/val/1|sort -R |tail -$diff_v)
    cd ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/train/1
    for f in $random_t ; do
      rm "$f"
    done
    cd ~/Documents/Deep_Learning/pytorch_imagenet/Task_1_Animals/val/1
    for f in $random_v ; do
      rm "$f"
    done
fi
# "~/Documents/Deep_Learning/pytorch_imagenet/random_sampling.sh"
