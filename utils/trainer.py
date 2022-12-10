import jax
import jax.numpy as jnp
import numpy as np

from tqdm import tqdm

class FlaxTrainer:
    def __init__(self):
        pass

    def fit(self, flaxModule, train_dataloader, test_dataloader, epochs):
        train_state = flaxModule.configure_train_state()
    
        train_step = 0
        for epoch in range(1, epochs+1):
            # Train
            train_batch_loss = []
            train_batch_acc = []
            train_tqdm = tqdm(train_dataloader, desc=f'Train Epoch {epoch}', leave=False)
            for idx, batch in enumerate(train_tqdm):
                grads, (loss, acc) = flaxModule.training_step(train_state, batch, idx)
                train_state = train_state.apply_gradients(grads=grads)

                train_batch_loss.append(loss)
                train_batch_acc.append(acc)
                train_tqdm.set_postfix_str(f'loss={loss:.3f} acc={acc:.3f}')

                train_step += 1

            # Test
            test_batch_loss = []
            test_batch_acc = []
            test_tqdm = tqdm(test_dataloader, desc=f'Test Epoch {epoch}', leave=False)
            for idx, batch in enumerate(test_tqdm):
                loss, acc = flaxModule.validation_step(train_state, batch, idx)
                test_batch_loss.append(loss)
                test_batch_acc.append(acc)
                test_tqdm.set_postfix(f'loss={loss:.3f} acc={acc:.3f}')

            print('Epoch', epoch, np.mean(test_batch_loss))
            print('Epoch', epoch, np.mean(test_batch_acc))

        return train_state

    def predict(self, module, train_state, test_loader):
        prediction = []
        test_tqdm = tqdm(test_dataloader, desc=f'Predict Epoch {epoch}', leave=False)
        for idx, batch in enumerate(test_tqdm):
            prediction.append(flaxModule.predict_step(train_state, batch, idx))

        return np.concatenate(prediction)

