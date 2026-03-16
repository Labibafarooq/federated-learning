from load_client_data import load_client_data

train_loader, val_loader = load_client_data(cid=0, data_dir="./client_data", batch_size=32)

# Print the shape of the first batch
images, labels = next(iter(train_loader))
print("Train batch shape:", images.shape)
print("Labels shape:", labels.shape)
