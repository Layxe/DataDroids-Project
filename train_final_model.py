# ##################################################################################################
#
# This script contains the final procedure on how to train the best performing model for the
# sentence relation task. This scripts combines different snippets from the previous experiments.
#
# ##################################################################################################

from transformers import TrainingArguments, Trainer

from lib.dataloading import load_sentence_dataframe, SentenceRelationTransformerDataset
from lib.transformershandler import TransformerHandler

# Constants
# ##################################################################################################

TRANSFORMER_NAME = 'roberta'
OUTPUT_DIR       = './results/roberta-base'

LEARNING_RATE    = 1e-3
NUM_EPOCHS       = 5

# Note: This works for 6 GB of GPU memory
BATCH_SIZE = 4
USE_FP16   = True

# Main procedure
# ##################################################################################################

def main():
    # Load our transformer model for the task
    roberta_handler = TransformerHandler(TRANSFORMER_NAME)

    # Load the training data
    training_data, test_data = load_sentence_dataframe(use_training_data=True, only_english=False)

    # Convert the data into a PyTorch Dataset
    training_dataset = SentenceRelationTransformerDataset(training_data, roberta_handler.get_tokenizer())
    test_dataset     = SentenceRelationTransformerDataset(test_data, roberta_handler.get_tokenizer())

    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        learning_rate               = LEARNING_RATE,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        num_train_epochs            = NUM_EPOCHS,
        weight_decay                = 0.01,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        fp16                        = USE_FP16
    )

    trainer = Trainer(
        model           = roberta_handler.get_transformer(),
        args            = training_args,
        train_dataset   = training_dataset,
        eval_dataset    = test_dataset,
        compute_metrics = roberta_handler.evaluation_function
    )

    trainer.train()

if __name__ == "__main__":
    main()