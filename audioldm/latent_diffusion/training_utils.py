"""Utilities for training with gradient flow."""

import torch
from contextlib import contextmanager

@contextmanager
def training_ema_scope(model, context=None):
    """
    EMA scope that preserves gradients for training.
    Similar to the regular ema_scope but WITHOUT disabling gradients.
    
    Args:
        model: The model with ema
        context: Optional context string for logging
    """
    if model.use_ema:
        # Store the current model state
        original_training_state = {}
        for name, param in model.model.named_parameters():
            original_training_state[name] = param.requires_grad
            
        # Store the current parameters
        model.model_ema.store(model.model.parameters())
        
        try:
            # Copy EMA weights to the model (but don't wrap in no_grad)
            # Temporarily ensure all parameters requiring EMA are marked as requires_grad=True
            # to match the mapping in the EMA module
            for name, param in model.model.named_parameters():
                if name in model.model_ema.m_name2s_name:
                    param.requires_grad = True
                    
            # Now copy the EMA weights
            model.model_ema.copy_to(model.model)
            
            if context is not None:
                print(f"{context}: Switched to EMA weights for training (with gradients enabled)")
                
            yield None
        finally:
            # Restore the original parameters
            model.model_ema.restore(model.model.parameters())
            
            # Restore the original requires_grad state
            for name, param in model.model.named_parameters():
                if name in original_training_state:
                    param.requires_grad = original_training_state[name]
                    
            if context is not None:
                print(f"{context}: Restored training weights")
    else:
        # If not using EMA, just yield
        yield None