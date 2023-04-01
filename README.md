Maybe you will meet with this issue: https://github.com/microsoft/torchgeo/issues/1143
If so, and/or you want to use WandB logger, you need to do the following steps:
Go to `local_venv/lib/python3.10/site-packages/torchgeo/trainers/segmentation.py` 
and change the following lines (starting from line 206)
```python
    try:
        datamodule = self.trainer.datamodule
        batch["prediction"] = y_hat_hard
        for key in ["image", "mask", "prediction"]:
            batch[key] = batch[key].cpu()
        sample = unbind_samples(batch)[0]
        fig = datamodule.plot(sample)
        summary_writer = self.logger.experiment
        summary_writer.add_figure(
            f"image/{batch_idx}", fig, global_step=self.global_step
        )
        plt.close()
    except ValueError:
        pass
```

to this lines
```python
    try:
        datamodule = self.trainer.datamodule
        batch["prediction"] = y_hat_hard
        for key in ["image", "mask", "prediction"]:
            batch[key] = batch[key].cpu()
        sample = unbind_samples(batch)[0]
        fig = datamodule.plot(sample)
        logger = self.logger
        plt.savefig(f"image_{batch_idx}.png")
        logger.log_image(key=f"image_{batch_idx}", images=[f"image_{batch_idx}.png"])
        plt.close()
    except ValueError:
        pass
```

If not, you have to use the TensorBoardLogger as mentioned in the issue.