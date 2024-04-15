#!/bin/bash


# this is added as a prefix to the wandb run name
USER_PREFIX=HK

# address of data folder
PRE="/mlodata1/hokarami/tedam"
DATA_NAME="retweets_mc"

# hyperparameters. Use `python Main.py -h` for more information
COMMON=" -data_label multiclass  -epoch 50 -per 100    -ES_pat 100 -wandb -wandb_project TEEDAM_unsupervised_timeCat "
HPs="-batch_size 256  -lr 0.003 -lr_scheduler StepLR -weight_decay 0.1 " # -te_d_mark 8 -te_d_time 8 -te_d_inner 16 -te_d_k 8 -te_d_v 8 "


# coefficients for multi-objective loss function
COEFS="-w_sample_label 10000  -w_time 1 -w_event 1"


############################################################ Possible loss functions for TEE
# TEE with PP(MC) (equation 2 in the paper)
TE__pp_mc="-event_enc 1          -mod mc    -next_mark 1     -mark_detach 1      -sample_label 0"

# TEE with PP(ML) (equation 3 in the paper)
TE__pp_ml="-event_enc 1          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 0"

# TEE with AE loss
TE__nextmark="-event_enc 1          -mod none      -next_mark 1     -mark_detach 0      -sample_label 0"

# TEE with PP(single) (equation 4 in the paper)
TE__pp_single_mark="-event_enc 1          -mod single    -next_mark 1     -mark_detach 0      -sample_label 0"



# running the experiments for different masking parameters and splits
for i_diag in {-2..1} # this is the masking parameter (w) introduced in the paper. i_diag in [-2,-1,0,1] corresponds to w in [3,2,1,0] respectively
do

    # this dataset has only one split

    SETTING=" -data  $PRE/$DATA_NAME/ -diag_offset $i_diag " 
        
    
    # TE__nextmark with time concatenation
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-concat-d$i_diag]" -time_enc concat     

    # TE__nextmark with time summation
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__nextmark -user_prefix "[$USER_PREFIX-TE__nextmark-sum-d$i_diag]" -time_enc sum




    
    # TE__pp_single_mark with time concatenation
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-concat-d$i_diag]" -time_enc concat        

    # TE__pp_single_mark with time summation
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_single_mark -user_prefix "[$USER_PREFIX-TE__pp_single_mark-sum-d$i_diag]" -time_enc sum
    




    # TE__pp_mc with time concatenation        
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_mc -user_prefix "[$USER_PREFIX-TE__pp_mc-concat-d$i_diag]" -time_enc concat
    
    # TE__pp_mc with time summation
    python Main.py  $HPs $COEFS $SETTING $COMMON $TE__pp_mc -user_prefix "[$USER_PREFIX-TE__pp_mc-sum-d$i_diag]" -time_enc sum 

        




done