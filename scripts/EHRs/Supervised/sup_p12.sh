#!/bin/bash


# this is added as a prefix to the wandb run name
USER_PREFIX=H70-

# address of data folder
PRE="/mlodata1/hokarami/tedam"
DATA_NAME="p12"

# hyperparameters. Use `python Main.py -h` for more information
COMMON=" -demo -data_label multilabel  -epoch 50 -per 100    -ES_pat 100 -wandb -wandb_project TEEDAM_supervised "
HPs="-batch_size 128  -lr 0.01 -weight_decay 0.1 -w_pos_label 0.5 "


# coefficients for multi-objective loss function
COEFS="-w_sample_label 100  -w_time 1 -w_event 1"


############################################################ Possible TEE loss
# DAM + TEE with AE loss and label loss
TEDA__nextmark="-event_enc 1    -state          -mod none      -next_mark 1     -mark_detach 0      -sample_label 1"

# DAM + TEE with PP(single) and label loss(equation 4 in the paper)
TEDA__pp_single_mark="-event_enc 1    -state          -mod single    -next_mark 1     -mark_detach 0      -sample_label 1"

# DAM + TEE with PP(ML) and label loss(equation 3 in the paper)
TEDA__pp_ml="-event_enc 1    -state          -mod ml        -next_mark 1     -mark_detach 1      -sample_label 1"

# DAM + TEE with label loss only
TEDA__none="-event_enc 1    -state          -mod none        -next_mark 1     -mark_detach 1      -sample_label 1"

# Only DAM with label loss
DA__base="-event_enc 0    -state          -mod none      -next_mark 1     -mark_detach 1      -sample_label 1"



# for different splits (raindrop-same splits as raindro's paper)    
for i_split in {0..4}
do
    SETTING=" -data  $PRE/$DATA_NAME/ -setting raindrop -split $i_split " 


        # DA__base (DAM in Table 5)
        echo python Main.py  $HPs $COEFS $SETTING $COMMON $DA__base -user_prefix "[$USER_PREFIX-DA__base-concat]" -time_enc concat -wandb_tag RD75    

        # TEDA__none (TEE+DAM in Table 5)
        echo python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__none -user_prefix "[$USER_PREFIX-TEDA__none-concat]" -time_enc concat -wandb_tag RD75 

        # TEDA__nextmark (TEE+DAM (AE loss) in Table 5)            
        echo python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat]" -time_enc concat -wandb_tag RD75

        # TEDA__pp_single_mark (TEE+DAM (single) in Table 5)                        
        echo python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA__pp_single_mark-concat]" -time_enc concat -wandb_tag RD75    

        # TEDA__pp_ml (TEE+DAM (ML) in Table 5)            
        echo python Main.py  $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat]" -time_enc concat -wandb_tag RD75 


done



# Run this only after running unsupervised scripts, we develop the models using ## TRANSFER LEARNING ##


for i_split in {0..4}
do
    SETTING=" -data  $PRE/$DATA_NAME/ -setting raindrop -split $i_split " 


        # TEDA__nextmark ([TEE with AE] + DAM in Table 5)
        TL="-transfer_learning DO -freeze TE -tl_tag RD74-nextmark3"
        python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TEDA__nextmark -user_prefix "[$USER_PREFIX-TEDA__nextmark-concat]" -time_enc concat -wandb_tag RD75

        # TEDA__pp_single_mark ([TEE with PP(single)] + DAM in Table 5)
        TL="-transfer_learning DO -freeze TE -tl_tag RD74-single3"
        python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TEDA__pp_single_mark -user_prefix "[$USER_PREFIX-TEDA__pp_single_mark-concat]" -time_enc concat -wandb_tag RD75    

        # TEDA__pp_ml ([TEE with PP(ML)] + DAM in Table 5)
        TL="-transfer_learning DO -freeze TE -tl_tag RD74-ml3"
        python Main.py  $TL $HPs $COEFS $SETTING $COMMON $TEDA__pp_ml -user_prefix "[$USER_PREFIX-TEDA__pp_ml-concat]" -time_enc concat -wandb_tag RD75 


done





