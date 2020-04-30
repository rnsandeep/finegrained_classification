max_epoch=25



########## TEsting NEtwork #######################
for epoch in `seq 0 $max_epoch`
do
       f="_checkpoint.pth.tar"
       echo "epoch: ", $epoch, "resize_size: ", $size
       python3 test_resnet50.py ../output train_output_08_02_preloaded/$epoch$f 258 test_output_08_02_preloaded
#       python3 test_inceptionv3.py $dataset $exp/$epoch$f $meanfile 299 $size $classes out_$exp
done

