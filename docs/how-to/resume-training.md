# Resume training

Training loops in `deep4downscaling.deep.train` support resuming from a saved checkpoint.

## Standard training loop

Pass the checkpoint path to `resume_checkpoint` in `standard_training_loop`:

```python
from deep4downscaling.deep.train import standard_training_loop

history = standard_training_loop(
    model=model,
    model_name="deepesd_tas",
    model_path="./models",
    loss_function=loss_fn,
    optimizer=optimizer,
    num_epochs=100,
    device="cuda",
    train_data=train_loader,
    valid_data=valid_loader,
    resume_checkpoint="./models/deepesd_tas_checkpoint.pt",
    patience_early_stopping=10,
)
```

The loop restores model weights, optimizer state, and epoch counter from the checkpoint file.

## Checkpoint saving

Checkpoints are saved automatically when using early stopping. You can also save at regular intervals:

```python
history = standard_training_loop(
    ...,
    save_checkpoint_every=5,   # save every 5 epochs
    save_versions_every=10,    # keep versioned snapshots
)
```

## Adversarial and CGAN loops

`adversarial_training_loop` and `standard_cgan_training_loop` follow the same checkpoint conventions. See the [training API](../api/deep/train.md) for model-specific arguments.

## Tips

- Use the same `model_name` and `model_path` as the original run
- Ensure the model architecture matches the checkpoint (same `x_shape`, `y_shape`, and stochastic settings)
- If you change the optimizer or learning rate, verify that restored optimizer state is still appropriate

## Related

- [Training API](../api/deep/train.md)
- [Workflow](../user-guide/workflow.md)
