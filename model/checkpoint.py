from tensorflow.keras.callbacks import ModelCheckpoint
import os

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, **kwargs):
        super().__init__(filepath, **kwargs)
        self.filepath = filepath
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.model_dir = self.filepath
        model_path = os.path.join(self.model_dir, f"model_epoch{epoch:02d}.h5")
        print(f"ckpt filepath: {model_path}")
        self.filepath = model_path
        super().on_epoch_end(epoch, logs)
        
