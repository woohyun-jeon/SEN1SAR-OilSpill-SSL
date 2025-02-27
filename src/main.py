import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import csv
import gc
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_supervised_transform, get_ssl_transform, S1OilDataset
from models import get_model, MoCo, SimCLR, get_moco_loss, get_simclr_loss
from utils import load_dataset_ids, load_config, set_seed, calculate_class_weights, calculate_metrics, save_metrics_to_csv


def pretrain_model(model, ssl_dataset, config, pretraining_type, device):
    model = model.to(device)
    pretrain_loader = DataLoader(ssl_dataset, batch_size=config['params']['batch_size'], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['params']['learning_rate'])
    scaler = GradScaler()
    best_loss = float('inf')
    best_model_path = None

    for epoch in range(config['params']['ssl_epochs']):
        model.train()
        running_loss = 0.0
        for images_i, images_j, _ in tqdm(pretrain_loader, desc=f"Pretraining {pretraining_type} Epoch [{epoch + 1}/{config['params']['ssl_epochs']}]"):
            images_i = images_i.to(device)
            images_j = images_j.to(device)

            optimizer.zero_grad()

            with autocast():
                if isinstance(model, MoCo):
                    logits, labels = model(images_i, images_j)
                    loss = get_moco_loss(logits, labels)
                elif isinstance(model, SimCLR):
                    features_i = model(images_i)
                    features_j = model(images_j)
                    features = torch.cat([features_i, features_j], dim=0)
                    loss = get_simclr_loss(features)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(pretrain_loader)
        print(f"Epoch [{epoch + 1}/{config['params']['ssl_epochs']}], Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(config['path']['out_path'], f"pretrained_{pretraining_type}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at {best_model_path}")

    return best_model_path


def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, config):
    model.to(device)
    scaler = GradScaler()
    best_loss = float('inf')
    best_model_state = None
    train_metrics_history = []
    val_metrics_history = []

    for epoch in range(config['params']['sup_epochs']):
        # training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['params']['sup_epochs']}"):
            images = images.to(device)
            masks = masks.float().to(device)

            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = probs.cpu().numpy()
                targets = masks.cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets)

        # calculate training metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        train_metrics = calculate_metrics(all_preds, all_targets)
        train_metrics['loss'] = running_loss / len(train_loader)
        train_metrics_history.append(train_metrics)

        # validation phase
        model.eval()
        val_metrics = evaluate_model(model, val_loader, device)
        val_metrics_history.append(val_metrics)

        # update scheduler
        scheduler.step(val_metrics['loss'])

        # save best model
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            best_model_state = model.state_dict()
            print(f"New best model saved! (val_loss: {best_loss:.4f})")

        # save metrics history
        save_metrics_to_csv(train_metrics_history, os.path.join(config['path']['out_path'], 'train_metrics.csv'))
        save_metrics_to_csv(val_metrics_history, os.path.join(config['path']['out_path'], 'val_metrics.csv'))

    return best_model_state, best_loss


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad(), autocast():
        for images, masks, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.float().to(device)

            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # get predictions
            probs = torch.sigmoid(outputs)
            preds = probs.cpu().numpy()
            targets = masks.cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

            # clear some memory
            del outputs, probs
            torch.cuda.empty_cache()

    # concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # calculate metrics
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = running_loss / len(dataloader)

    return metrics


def save_predictions(model, dataloader, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad(), autocast():
        for images, _, filenames in tqdm(dataloader, desc="Saving predictions"):
            images = images.to(device)
            outputs = model(images)

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            outputs = outputs.cpu().numpy()

            for j, output in enumerate(outputs):
                filename = os.path.splitext(filenames[j])[0] + '.png'
                save_path = os.path.join(save_dir, filename)
                output = output.squeeze()
                output = (output * 255).astype('uint8')
                img = Image.fromarray(output, mode='L')
                img.save(save_path, format='PNG', optimize=True)


def main():
    config_path = 'configs.yaml'
    cfgs = load_config(config_path)
    os.makedirs(cfgs['path']['out_path'], exist_ok=True)
    set_seed(seed=cfgs['params']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set model and encoder name from config
    model_name = cfgs['model'].get('name', 'UNet')
    encoder_name = cfgs['model'].get('encoder_name', 'resnet50')

    # validate model & encoder name
    assert model_name in ['UNet', 'DeepLabV3plus'], f"Model {model_name} is not supported"
    assert encoder_name in ['resnet50', 'resnet101'], f"Encoder {encoder_name} is not supported"

    print(f"Using {model_name} with {encoder_name} backbone")

    # create main output directory for this model and backbone combination
    model_dir = os.path.join(cfgs['path']['out_path'], f"{model_name}_{encoder_name}")
    os.makedirs(model_dir, exist_ok=True)

    # load datasets
    data_path = cfgs['path']['data_path']
    train_ids = load_dataset_ids(os.path.join(data_path, 'train.txt'))
    val_ids = load_dataset_ids(os.path.join(data_path, 'valid.txt'))
    test_ids = load_dataset_ids(os.path.join(data_path, 'test.txt'))

    # execute SSL pretraining
    all_ids = train_ids + val_ids
    ssl_dataset = S1OilDataset(data_path, all_ids, transform=get_ssl_transform(), load_labels=False)

    ssl_models = {}
    for ssl_type in ['moco', 'simclr']:
        print(f"\nPretraining {ssl_type.upper()} model with {encoder_name} backbone...")
        if ssl_type == 'moco':
            K = ((cfgs['params']['batch_size'] * 1000) // cfgs['params']['batch_size']) * cfgs['params']['batch_size']
            ssl_model = MoCo(dim=128, K=K, in_channels=1, encoder_name=encoder_name)
        else:
            ssl_model = SimCLR(feature_dim=128, in_channels=1, encoder_name=encoder_name)

        # create modified config to save SSL models directly in the model directory
        ssl_config = {**cfgs, 'path': {**cfgs['path'], 'out_path': model_dir}}

        # save SSL models in the main model directory
        ssl_model_path = os.path.join(model_dir, f"pretrained_{ssl_type}_best.pth")
        best_model_path = pretrain_model(ssl_model, ssl_dataset, ssl_config, ssl_type, device)

        ssl_models[ssl_type] = best_model_path
        torch.cuda.empty_cache()

    # get datasets for supervised training
    train_transform = get_supervised_transform()
    val_transform = get_supervised_transform()

    train_dataset = S1OilDataset(data_path, train_ids, transform=train_transform)
    val_dataset = S1OilDataset(data_path, val_ids, transform=val_transform)
    test_dataset = S1OilDataset(data_path, test_ids, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfgs['params']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfgs['params']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfgs['params']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

    # run experiments
    pretrained_types = [None, 'imagenet', 'moco', 'simclr']
    results = {}
    for pretrained in pretrained_types:
        # reset seed for each experiment to ensure fair comparison
        set_seed(seed=cfgs['params']['seed'])

        # get pretrain name for directory structure
        pretrain_dir_name = f"{pretrained}_pretrain" if pretrained else "no_pretrain"
        print(f"\nTraining {model_name} with {encoder_name} encoder and {pretrain_dir_name}")

        # create output directory for this experiment within the model directory
        output_dir = os.path.join(model_dir, pretrain_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        # model setup
        if pretrained in ['moco', 'simclr']:
            model = get_model(model_name, encoder_name=encoder_name, pretrained=pretrained,
                              n_classes=1, in_channels=1, ssl_weights_path=ssl_models[pretrained])
        else:
            model = get_model(model_name, encoder_name=encoder_name, pretrained=pretrained,
                              n_classes=1, in_channels=1)

        model = model.to(device)

        # training setup
        class_weights = calculate_class_weights(train_dataset)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(model.parameters(), lr=cfgs['params']['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # create modified config for train_model to save metrics in experiment directory
        train_config = {**cfgs, 'path': {**cfgs['path'], 'out_path': output_dir}}

        # training
        best_model_state, best_val_loss = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler, loss_fn=criterion,
            device=device, config=train_config
        )

        # load best model and evaluate
        model.load_state_dict(best_model_state)
        print("\nCalculating final test metrics...")
        test_metrics = evaluate_model(model, test_loader, device)

        # save results
        results[f"{model_name}_{encoder_name}_{pretrain_dir_name}"] = {'test': test_metrics}
        metrics_file = os.path.join(output_dir, 'test_metrics.csv')
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=test_metrics.keys())
            writer.writeheader()
            writer.writerow(test_metrics)

        # save model
        model_file = os.path.join(output_dir, f'model.pth')
        torch.save(model.state_dict(), model_file)

        # create predictions directory
        predictions_dir = os.path.join(output_dir, 'predictions')
        save_predictions(model, test_loader, predictions_dir, device)

        # clean up
        torch.cuda.empty_cache()
        gc.collect()

    # print final results
    print("\nFinal Results Summary:")
    for exp_name, metrics in results.items():
        print(f"\n{exp_name}:")
        for metric, value in metrics['test'].items():
            print(f"    {metric}: {value:.4f}")


if __name__ == "__main__":
    main()