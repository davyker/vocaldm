"""Utility functions for gradient-enabled training."""

import torch
from audioldm.latent_diffusion.util import *  # Import all original utility functions

# Create a training-specific checkpoint function that maintains gradient flow
def checkpoint_for_training(func, inputs, params, flag):
    """
    A checkpoint function that properly maintains gradient flow for training.
    Unlike the original, this version doesn't detach tensors during backward.
    
    Args:
        func: The function to evaluate
        inputs: Argument sequence to pass to func
        params: Parameters that func depends on
        flag: Whether to enable checkpointing
    """
    if not flag:
        return func(*inputs)
    
    # Simple wrapper for direct gradient flow
    def grad_preserving_wrapper(func, inputs, params):
        # Execute function directly - no detaching or re-computing in backward
        output = func(*inputs)
        return output

    all_inputs = list(inputs) + list(params)
    
    # For all tensors that should keep gradients, ensure requires_grad=True
    for i, inp in enumerate(all_inputs):
        if isinstance(inp, torch.Tensor) and not inp.requires_grad:
            if any(x is inp for x in params):  # Only set requires_grad on params that should have it
                all_inputs[i] = inp.requires_grad_(True)
    
    return grad_preserving_wrapper(func, inputs, params)

# A complete replacement for the CheckpointFunction that properly maintains gradient flow
class CheckpointFunctionForTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.length = length
        # Save the tensors but DON'T consume them - we'll need them in backward
        ctx.save_for_backward(*args)
        input_tensors = args[:length]
        # Run the function without no_grad so we can track which inputs affect the output
        output_tensors = run_function(*input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # Retrieve saved tensors - this is the proper way in PyTorch autograd.Function
        saved_tensors = ctx.saved_tensors
        input_tensors = saved_tensors[:ctx.length]
        input_params = saved_tensors[ctx.length:]
        
        # Get the run_function from context
        run_function = ctx.run_function
        
        # Critical difference: don't detach the input tensors
        # Instead, maintain the original tensors to preserve gradient flow
        with torch.enable_grad():
            # Make a forward pass with the original tensors
            output_tensors = run_function(*input_tensors)
        
        # Filter input_tensors + input_params to only those requiring gradients
        # This is to address "One of the differentiated Tensors does not require grad" error
        grad_inputs = []
        for x in input_tensors + input_params:
            if isinstance(x, torch.Tensor) and x.requires_grad:
                grad_inputs.append(x)
        
        # Calculate gradients with respect to filtered inputs
        if len(grad_inputs) > 0:
            input_grads = torch.autograd.grad(
                output_tensors,
                grad_inputs,
                output_grads,
                allow_unused=True,
            )
            
            # Create a list with None for non-differentiable tensors
            all_input_grads = []
            grad_idx = 0
            for x in input_tensors + input_params:
                if isinstance(x, torch.Tensor) and x.requires_grad:
                    all_input_grads.append(input_grads[grad_idx])
                    grad_idx += 1
                else:
                    all_input_grads.append(None)
        else:
            # If nothing requires gradients, return None for all inputs
            all_input_grads = [None] * (len(input_tensors) + len(input_params))
        
        return (None, None) + tuple(all_input_grads)

# A complete checkpoint function replacement that uses the training-compatible checkpointing
def checkpoint_full_replacement(func, inputs, params, flag):
    """
    Enhanced checkpoint function that properly maintains gradient flow.
    Completely replaces the original checkpoint function for training.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunctionForTraining.apply(func, len(inputs), *args)
    else:
        return func(*inputs)