#!/bin/bash


for s in {1..11}; do
    for o in {1..50}; do
        echo "Processing s${s}/o${o}..."
        
        python hands-segmentation-pytorch/main.py \
            --mode predict \
            --data_base_path "core50_128x128/s${s}/o${o}" \
            --model_checkpoint "checkpoint/checkpoint.ckpt" \
            --model_pretrained
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed s${s}/o${o}"
        else
            echo "✗ Error processing s${s}/o${o}"
            # Uncomment the next line if you want to stop on first error
            # exit 1
        fi
        
        echo "---"
    done
done
