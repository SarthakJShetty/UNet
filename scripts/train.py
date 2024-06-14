import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from unet.datasets import CarvanaDataset
from unet.model import UNet

if __name__ == "__main__":
    dataset = CarvanaDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    model = UNet()
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)

    model.train()

    epochs = 100
    batch_loss = 0

    for epoch in range(epochs):

        for image, label in tqdm(dataloader):
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
        print(f"Epoch loss: {batch_loss/len(dataloader)}")
