import os
import torch
import torch.utils.data
from loguru import logger
from pytorch_transformers import BertConfig, BertForSequenceClassification
from tqdm import tqdm
from .data import SSTDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        loss, logits = model(batch, labels=labels)
        err = lossfn(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred_labels = torch.argmax(logits, dim=1)
        train_acc += (pred_labels == labels).sum().item()
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    return train_loss, train_acc

def evaluate_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)[0]
            err = lossfn(logits, labels)
            loss += err.item()
            pred_labels = torch.argmax(logits, dim=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc

def train(
        root=True,
        binary=False,
        bert='bert-large-uncased',
        epochs=30,
        batch_size=32,
        save=False,
):
    train_set = SSTDataset('train', root=root, binary=binary)
    dev_set = SSTDataset('dev', root=root, binary=binary)
    test_set = SSTDataset('test', root=root, binary=binary)

    config = BertConfig.from_pretrained(bert)
    if not binary:
        config.num_labels = 5
    model = BertForSequenceClassification.from_pretrained(bert, config=config)

    model = model.to(device)
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(1, epochs):
        train_loss, train_acc = train_one_epoch(
            model, lossfn, optimizer, train_set, batch_size=batch_size
        )
        val_loss, val_acc = evaluate_one_epoch(
            model, lossfn, optimizer, dev_set, batch_size=batch_size
        )
        test_loss, test_acc = evaluate_one_epoch(
            model, lossfn, optimizer, test_set, batch_size=batch_size
        )

        logger.info(f"epoch={epoch}")
        logger.info(
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}"
        )
        logger.info(
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}"
        )

        if save:
            label = 'binary' if binary else 'fine'
            nodes = 'root' if root else 'all'
            torch.save(model, f"{bert}_{nodes}_e{epoch}.pickle")

    logger.success('Finished.')