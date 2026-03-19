import torch 

"""


"""
def simple_sft_loss(model, input_ids):
    """
    SFT loss
    """

    # Forward Pass 
    model_output = model(input_ids)
    logits = model_output.logits
    predictions = logits[:-1]


    # Shift for next token prediction 



    # Cross Entropy Loss 