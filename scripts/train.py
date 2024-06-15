import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from unet.datasets import CarvanaDataset
from unet.model import UNet

if __name__ == "__main__":
    dataset = CarvanaDataset()
    train_dataset, val_dataset = dataset.split_dataset(split_ratio=0.8)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet()
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)

    model.train()

    epochs = 100
    batch_loss = 0

    for epoch in range(epochs):

        train_batch_loss = 0
        val_batch_loss = 0

        for image, label in tqdm(train_dataloader):
            optimizer.zero_grad()

            prediction = model(image)
            loss = loss_function(prediction, label)

            batch_loss += loss
            loss.backward()
            optimizer.step()
        _image_label_prediction = (
            torch.cat(
                [image[0], label[0].repeat(3, 1, 1), torch.nn.functional.sigmoid(prediction[0].detach().to("cpu")).repeat(3, 1, 1)],
                dim=-1,
            )
            .permute(1, 2, 0)
            .numpy()
        )
        plt.figure(figsize=(3, 1))
        plt.imshow(_image_label_prediction)
        plt.axis("off")
        plt.savefig(f"prediction_{epoch}.png")
        plt.close()

        print(f"Train Epoch loss: {batch_loss/len(train_dataloader)}")

        with torch.no_grad():
            model.eval()
            for image, label in tqdm(val_dataloader):

                prediction = model(image)
                loss = loss_function(prediction, label)

                val_batch_loss += loss

            _image_label_prediction = (
                torch.cat(
                    [image[0], label[0].repeat(3, 1, 1), torch.nn.functional.sigmoid(prediction[0].detach().to("cpu")).repeat(3, 1, 1)],
                    dim=-1,
                )
                .permute(1, 2, 0)
                .numpy()
            )
            plt.figure(figsize=(3, 1))
            plt.imshow(_image_label_prediction)
            plt.axis("off")
            plt.savefig(f"val_prediction_{epoch}.png")
            plt.close()
            print(f"Val Epoch loss: {val_batch_loss/len(val_dataloader)}")
